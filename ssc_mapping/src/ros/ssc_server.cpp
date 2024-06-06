#include "ssc_mapping/ros/ssc_server.h"

#include <iomanip>
#include <sstream>

#include <voxblox/core/common.h>
#include <voxblox/core/voxel.h>

#include "ssc_mapping/core/voxel.h"
#include "ssc_mapping/fusion/counting_fusion.h"
#include "ssc_mapping/fusion/log_odds_fusion.h"
#include "ssc_mapping/fusion/naive_fusion.h"
#include "ssc_mapping/fusion/occupancy_fusion.h"
#include "ssc_mapping/fusion/sc_fusion.h"
#include "ssc_mapping/utils/voxel_utils.h"
#include "ssc_mapping/visualization/visualization.h"

namespace voxblox {

SSCServer::SSCServer(const ros::NodeHandle &nh,
                     const ros::NodeHandle &nh_private,
                     const ssc_fusion::BaseFusion::Config &fusion_config,
                     const SSCMap::Config &config)
    : nh_(nh), nh_private_(nh_private), transformer_(nh, nh_private),
      integrate_point_cloud_(false), publish_pointclouds_on_update_(true),
      world_frame_("odom"), decay_weight_std_(fusion_config.decay_weight_std) {
  ssc_map_.reset(new SSCMap(config));

  if (fusion_config.fusion_strategy.compare(ssc_fusion::strategy::naive) == 0) {
    base_fusion_.reset(new ssc_fusion::NaiveFusion());
  } else if (fusion_config.fusion_strategy.compare(
                 ssc_fusion::strategy::occupancy_fusion) == 0) {
    base_fusion_.reset(new ssc_fusion::OccupancyFusion(fusion_config));
  } else if (fusion_config.fusion_strategy.compare(
                 ssc_fusion::strategy::log_odds) == 0) {
    base_fusion_.reset(new ssc_fusion::LogOddsFusion(fusion_config));
  } else if (fusion_config.fusion_strategy.compare(
                 ssc_fusion::strategy::counting) == 0) {
    base_fusion_.reset(new ssc_fusion::CountingFusion(fusion_config));
  } else if (fusion_config.fusion_strategy.compare(
                 ssc_fusion::strategy::sc_fusion) == 0) {
    base_fusion_.reset(new ssc_fusion::SCFusion(fusion_config));
  } else {
    LOG(WARNING)
        << "Wrong Fusion strategy provided. Using default fusion strategy";
    base_fusion_.reset(new ssc_fusion::OccupancyFusion(fusion_config));
  }

  // TEST: Evaluation utils.
  nh_private_.param("evaluate", evaluate_, evaluate_);
  nh_private_.param("evaluation_path", evaluation_path_, evaluation_path_);
  nh_private_.param("save_interval", save_interval_, save_interval_);
  if (evaluate_) {
    shutdown_timer_ = nh_private_.createTimer(ros::Duration(0.3),
                                              &SSCServer::timerCallback, this);
  }

  // integrate measured pointcloud
  nh_private_.param("integrate_pointcloud", integrate_point_cloud_,
                    integrate_point_cloud_);

  // publish voxelized probabilistic map
  nh_private_.param("publish_pointclouds", publish_pointclouds_on_update_,
                    publish_pointclouds_on_update_);

  // subscribe to SSC from node with 3D CNN
  ssc_map_sub_ = nh_.subscribe("ssc", evaluate_ ? 1000 : 50,
                               &SSCServer::sscCallback, this);
  save_map_srv_ = nh_private_.advertiseService(
      "save_map", &SSCServer::saveMapCallback, this);

  if (integrate_point_cloud_) {
    pointcloud_sub_ =
        nh_.subscribe("pointcloud", 10, &SSCServer::insertPointcloud, this);
  }

  if (publish_pointclouds_on_update_) {
    // publish fused maps as occupancy pointcloud
    ssc_pointcloud_pub_ =
        nh_private_.advertise<pcl::PointCloud<pcl::PointXYZRGB>>(
            "occupancy_pointcloud", 1, true);

    // publish fused maps as occupancy nodes - marker array
    occupancy_marker_pub_ =
        nh_private_.advertise<visualization_msgs::MarkerArray>(
            "ssc_occupied_nodes", 1, true);
  }
}

ssc_fusion::BaseFusion::Config
SSCServer::getFusionConfigROSParam(const ros::NodeHandle &nh,
                                   const ros::NodeHandle &nh_private) {
  ssc_fusion::BaseFusion::Config fusion_config;

  nh_private.param("fusion_pred_conf", fusion_config.pred_conf,
                   fusion_config.pred_conf);
  nh_private.param("fusion_max_weight", fusion_config.max_weight,
                   fusion_config.max_weight);
  nh_private.param("fusion_prob_occupied", fusion_config.prob_occupied,
                   fusion_config.prob_occupied);
  nh_private.param("fusion_prob_free", fusion_config.prob_free,
                   fusion_config.prob_free);
  nh_private.param("fusion_min_prob", fusion_config.min_prob,
                   fusion_config.min_prob);
  nh_private.param("fusion_max_prob", fusion_config.max_prob,
                   fusion_config.max_prob);
  nh_private.param("fusion_strategy", fusion_config.fusion_strategy,
                   fusion_config.fusion_strategy);
  nh_private.param("decay_weight_std", fusion_config.decay_weight_std,
                   fusion_config.decay_weight_std);

  return fusion_config;
}

SSCMap::Config
SSCServer::getSSCMapConfigFromRosParam(const ros::NodeHandle &nh_private) {
  SSCMap::Config ssc_map_config;
  double voxel_size = ssc_map_config.ssc_voxel_size;
  int voxels_per_side = ssc_map_config.ssc_voxels_per_side;
  nh_private.param("ssc_voxel_size", voxel_size, voxel_size);
  nh_private.param("ssc_voxels_per_side", voxels_per_side, voxels_per_side);
  if (!isPowerOfTwo(voxels_per_side)) {
    ROS_ERROR("voxels_per_side must be a power of 2, setting to default value");
    voxels_per_side = ssc_map_config.ssc_voxels_per_side;
  }

  ssc_map_config.ssc_voxel_size = static_cast<FloatingPoint>(voxel_size);
  ssc_map_config.ssc_voxels_per_side = voxels_per_side;

  return ssc_map_config;
}

bool SSCServer::saveMap(const std::string &file_path) {
  // Inheriting classes should add saving other layers to this function.
  return io::SaveLayer(ssc_map_->getSSCLayer(), file_path);
}

bool SSCServer::saveMapCallback(voxblox_msgs::FilePath::Request &request,
                                voxblox_msgs::FilePath::Response &) {
  return saveMap(request.file_path);
}

void SSCServer::timerCallback(const ros::TimerEvent & /* event */) {
  if (time_offset_ < 0.0) {
    return;
  }
  const double time_since_last_msg =
      ros::WallTime::now().toSec() - last_message_received_;
  if (time_since_last_msg > 3.0) {
    LOG(INFO) << "No more messages received. Shutting down.";
    ros::shutdown();
  }
}

void SSCServer::sscCallback(const ssc_msgs::SSCGrid::ConstPtr &msg) {
  // TEST: Evaluation
  if (evaluate_) {
    last_message_received_ = ros::WallTime::now().toSec();
    double current_time = msg->header.stamp.toSec();
    if (time_offset_ < 0) {
      time_offset_ = current_time;
    } else {
      current_time -= time_offset_;
      if (current_time >= next_save_time_) {
        next_save_time_ += save_interval_;
        std::stringstream ss;
        ss << std::setw(5) << std::setfill('0') << save_map_id_;
        saveMap(evaluation_path_ + ss.str() + ".ssc");
        LOG(INFO) << "Saving map " << save_map_id_ << " at time "
                  << current_time;
        save_map_id_++;
      }
    }
  }

  if (msg->origin_z <
      -1.5f) { // a check to print if there is a wrong pose/outlier received
    LOG(WARNING) << "Outlier pose detected with origin at " << msg->origin_z
                 << ". Skipping.";
    return;
  }

  auto exp_decay_weight = [](double x, double y, double z, double std_dev) {
    return exp(-0.5 / std_dev * std::sqrt((x * x) + (y * y) + (z * z)));
    // NOTE: Used to be exponential instead of gaussian (add root)
  };

  // todo - resize voxels to full size? Resize voxel grid from 64x36x64 to
  // 240x144x240 todo - update the  ssc_msgs::SSCGrid to contain the scale
  // instead of hard coding update - done - removed upsampling as its to slow
  // and not needed Layer<SSCOccupancyVoxel>
  // temp_layer(ssc_map_->getSSCLayerPtr()->voxel_size() * 4,
  //                                    (ssc_map_->getSSCLayerPtr()->block_size()/ssc_map_->getSSCLayerPtr()->voxel_size())
  //                                    / 4);
  // Note: Not needed anymore

  // completions are in odometry frame
  // Note: numpy flattens an array (x,y,z) such that
  // X
  //  \
    //   \
    //    + ------------------> Z
  //    |
  //    |
  //    v
  //    Y axis
  // Input from simulation are X,Y,Z. X forward, Z up and Y left.
  // Its converted to grid coordinate system for
  // input to network as Y,Z,X and similarly
  // Network does predictions in Y,Z,X coordinates.
  // predictions in Y,Z,X are flattened using numpy (for sending as array)
  // These flattened predictions have are formated as:
  // first 240 would have z from 0 to 239, y=0, z=0
  // and similarly next 240 would have y=1,x=0
  // and so on. if we read here in numpy way way then output would
  // be in same Y,Z,X form.
  // Solution:
  // Either convert here from Y,Z,X to X,Y,Z and load in the same format
  // as numpy saved
  // Or  convert Y,Z,X -> X,Y,Z before sending and send transpose
  // of X,Y,Z from Numpy and load here with the x being fastest axis.
  auto grid_origin_index = getGridIndexFromOriginPoint<GlobalIndex>(
      Point(msg->origin_x, msg->origin_y, msg->origin_z),
      ssc_map_->getSSCLayerPtr()->voxel_size_inv());

  for (size_t x = 0; x < msg->depth; x++) {
    for (size_t y = 0; y < msg->height; y++) {
      for (size_t z = 0; z < msg->width; z++) {
        size_t idx = x * msg->width * msg->height + y * msg->width + z;
        uint predicted_label = msg->data[idx];
        float free_space_confidence = msg->data[idx] - predicted_label;
        float occupied_confidence = 1.f - free_space_confidence;

        float weight = decay_weight_std_ > 0.f
                           ? exp_decay_weight(x, y, z, decay_weight_std_)
                           : 1.f;

        // transform from grid coordinate system to world coordinate system
        uint32_t world_orient_x =
            z; // still uses scale of grid though rotation is in world coords
        uint32_t world_orient_y = x;
        uint32_t world_orient_z = y;
        GlobalIndex voxelIdx(world_orient_x, world_orient_y, world_orient_z);

        // add origin so that new voxels are integrated wrt origin
        voxelIdx += grid_origin_index;

        SSCOccupancyVoxel *voxel =
            ssc_map_->getSSCLayerPtr()->getVoxelPtrByGlobalIndex(voxelIdx);

        // check if the block containing the voxel exists.
        if (voxel == nullptr) {
          BlockIndex block_idx = getBlockIndexFromGlobalVoxelIndex(
              voxelIdx, ssc_map_->getSSCLayerPtr()->voxels_per_side_inv());
          auto block =
              ssc_map_->getSSCLayerPtr()->allocateBlockPtrByIndex(block_idx);
          const VoxelIndex local_voxel_idx = getLocalFromGlobalVoxelIndex(
              voxelIdx, ssc_map_->getSSCLayerPtr()->voxels_per_side());
          voxel = &block->getVoxelByVoxelIndex(local_voxel_idx);
        }

        base_fusion_->fuse(voxel, predicted_label, occupied_confidence, weight);
      }
    }
  }

  if (publish_pointclouds_on_update_) {
    publishSSCOccupancyPoints();
    publishSSCOccupiedNodes();
  }
}

void SSCServer::insertPointcloud(
    const sensor_msgs::PointCloud2::Ptr &pointcloud_msg) {
  Transformation T_G_C;
  if (!transformer_.lookupTransform(pointcloud_msg->header.frame_id,
                                    world_frame_, pointcloud_msg->header.stamp,
                                    &T_G_C)) {
    return;
  }

  Pointcloud points_C;
  Colors colors;

  pcl::PointCloud<pcl::PointXYZRGB> pointcloud_pcl;
  // pointcloud_pcl is modified below:
  pcl::fromROSMsg(*pointcloud_msg, pointcloud_pcl);
  convertPointcloud(pointcloud_pcl, nullptr, &points_C, &colors);

  const Point &origin = T_G_C.getPosition();

  LongIndexSet free_cells;
  LongIndexSet occupied_cells;

  const Point start_scaled =
      origin * ssc_map_->getSSCLayerPtr()->voxel_size_inv();
  Point end_scaled = Point::Zero();

  for (size_t pt_idx = 0; pt_idx < points_C.size(); ++pt_idx) {
    const Point &point_C = points_C[pt_idx];
    const Point point_G = T_G_C * point_C;
    const Ray unit_ray = (point_G - origin).normalized();

    AlignedVector<GlobalIndex> global_voxel_indices;
    FloatingPoint ray_distance = (point_G - origin).norm();

    double max_ray_length = 5.0;
    double min_ray_length = 0.1;

    if (ray_distance < min_ray_length) {
      continue;
    } else if (ray_distance > max_ray_length) {
      // Simply clear up until the max ray distance in this case.
      end_scaled = (origin + max_ray_length * unit_ray) *
                   ssc_map_->getSSCLayerPtr()->voxel_size_inv();

      if (free_cells.find(getGridIndexFromPoint<GlobalIndex>(end_scaled)) ==
          free_cells.end()) {
        castRay(start_scaled, end_scaled, &global_voxel_indices);
        free_cells.insert(global_voxel_indices.begin(),
                          global_voxel_indices.end());
      }
    } else {
      end_scaled = point_G * ssc_map_->getSSCLayerPtr()->voxel_size_inv();
      if (occupied_cells.find(getGridIndexFromPoint<GlobalIndex>(end_scaled)) ==
          occupied_cells.end()) {
        castRay(start_scaled, end_scaled, &global_voxel_indices);

        if (global_voxel_indices.size() > 2) {
          free_cells.insert(global_voxel_indices.begin(),
                            global_voxel_indices.end() - 1);
          occupied_cells.insert(global_voxel_indices.back());
        }
      }
    }
  }

  // Clean up the lists: remove any occupied cells from free cells.
  for (const GlobalIndex &global_index : occupied_cells) {
    LongIndexSet::iterator cell_it = free_cells.find(global_index);
    if (cell_it != free_cells.end()) {
      free_cells.erase(cell_it);
    }
  }

  // Then actually update the occupancy voxels.
  BlockIndex last_block_idx = BlockIndex::Zero();
  Block<SSCOccupancyVoxel>::Ptr block;

  // bool occupied = false;
  for (const GlobalIndex &global_voxel_idx : free_cells) {
    SSCOccupancyVoxel *voxel =
        ssc_map_->getSSCLayerPtr()->getVoxelPtrByGlobalIndex(global_voxel_idx);

    // check if the block containing the voxel exists.
    if (voxel == nullptr) {
      // ssc_map_->getSSCLayerPtr()->a
      BlockIndex block_idx = getBlockIndexFromGlobalVoxelIndex(
          global_voxel_idx, ssc_map_->getSSCLayerPtr()->voxels_per_side_inv());
      auto block =
          ssc_map_->getSSCLayerPtr()->allocateBlockPtrByIndex(block_idx);
      const VoxelIndex local_voxel_idx = getLocalFromGlobalVoxelIndex(
          global_voxel_idx, ssc_map_->getSSCLayerPtr()->voxels_per_side());
      voxel = &block->getVoxelByVoxelIndex(local_voxel_idx);
    }
    voxel->probability_log = std::min(
        std::max(voxel->probability_log + voxblox::logOddsFromProbability(0.3),
                 voxblox::logOddsFromProbability(0.12f)),
        voxblox::logOddsFromProbability(0.97f));
    voxel->observed = true;

    // initialize label as the voxel is observed now
    if (voxel->label < 0) {
      voxel->label = 0;
    }
  }

  for (const GlobalIndex &global_voxel_idx : occupied_cells) {
    SSCOccupancyVoxel *voxel =
        ssc_map_->getSSCLayerPtr()->getVoxelPtrByGlobalIndex(global_voxel_idx);

    // check if the block containing the voxel exists.
    if (voxel == nullptr) {
      // ssc_map_->getSSCLayerPtr()->a
      BlockIndex block_idx = getBlockIndexFromGlobalVoxelIndex(
          global_voxel_idx, ssc_map_->getSSCLayerPtr()->voxels_per_side_inv());
      auto block =
          ssc_map_->getSSCLayerPtr()->allocateBlockPtrByIndex(block_idx);
      const VoxelIndex local_voxel_idx = getLocalFromGlobalVoxelIndex(
          global_voxel_idx, ssc_map_->getSSCLayerPtr()->voxels_per_side());
      voxel = &block->getVoxelByVoxelIndex(local_voxel_idx);
    }

    voxel->probability_log = std::min(
        std::max(voxel->probability_log + voxblox::logOddsFromProbability(0.7f),
                 voxblox::logOddsFromProbability(0.12f)),
        voxblox::logOddsFromProbability(0.97f));
    voxel->observed = true;

    // measured voxels are denoted by separate category
    if (voxel->label < 0) {
      voxel->label = 12;
    }
  }

  if (publish_pointclouds_on_update_) {
    publishSSCOccupancyPoints();
    publishSSCOccupiedNodes();
  }
}

void SSCServer::publishSSCOccupancyPoints() {
  if (ssc_pointcloud_pub_.getNumSubscribers() > 0) {
    // Create a pointcloud with distance = intensity.
    pcl::PointCloud<pcl::PointXYZRGB> pointcloud;
    createPointcloudFromSSCLayer(ssc_map_->getSSCLayer(), &pointcloud);
    pointcloud.header.frame_id = world_frame_;
    ssc_pointcloud_pub_.publish(pointcloud);
  }
}

void SSCServer::publishSSCOccupiedNodes() {
  if (occupancy_marker_pub_.getNumSubscribers() > 0) {
    // Create a pointcloud with distance = intensity.
    visualization_msgs::MarkerArray marker_array;
    createOccupancyBlocksFromSSCLayer(ssc_map_->getSSCLayer(), world_frame_,
                                      &marker_array);
    occupancy_marker_pub_.publish(marker_array);
  }
}

// function definitions
std::string SSCMap::Config::print() const {
  std::stringstream ss;
  // clang-format off
  ss << "====================== SSCMap Map Config ========================\n";
  ss << " - ssc_voxel_size:               " << ssc_voxel_size << "\n";
  ss << " - ssc_voxels_per_side:          " << ssc_voxels_per_side << "\n";
  ss << "==============================================================\n";
  // clang-format on
  return ss.str();
}

} // namespace voxblox
