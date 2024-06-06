#include "ssc_planning/map/ssc_voxblox_map.h"

#include <active_3d_planning_core/data/system_constraints.h>
#include <ssc_mapping/visualization/color_map.h>
#include <voxblox/core/common.h>
#include <voxblox_ros/ros_params.h>

namespace active_3d_planning {
namespace map {

ModuleFactoryRegistry::Registration<SSCVoxbloxOccupancyMap>
    SSCVoxbloxOccupancyMap::registration("SSCVoxbloxOccupancyMap");

SSCVoxbloxOccupancyMap::SSCVoxbloxOccupancyMap(PlannerI &planner)
    : OccupancyMap(planner) {}

voxblox::SSCServer &SSCVoxbloxOccupancyMap::getSSCServer() {
  return *ssc_server_;
}

voxblox::EsdfServer &SSCVoxbloxOccupancyMap::getESDFServer() {
  return *esdf_server_;
}

void SSCVoxbloxOccupancyMap::setupFromParamMap(Module::ParamMap *param_map) {
  voxblox::SSCMap::Config map_config;
  ssc_fusion::BaseFusion::Config fusion_config;

  setParam<float>(param_map, "ssc_confidence_threshold",
                  &ssc_confidence_threshold_, ssc_confidence_threshold_);

  // load ssc map config
  setParam<float>(param_map, "voxel_size", &map_config.ssc_voxel_size,
                  map_config.ssc_voxel_size);

  // load fusion config
  setParam<float>(param_map, "pred_conf", &fusion_config.pred_conf,
                  fusion_config.pred_conf);
  setParam<float>(param_map, "max_weight", &fusion_config.max_weight,
                  fusion_config.max_weight);
  setParam<float>(param_map, "prob_occupied", &fusion_config.prob_occupied,
                  fusion_config.prob_occupied);
  setParam<float>(param_map, "prob_free", &fusion_config.prob_free,
                  fusion_config.prob_free);
  setParam<float>(param_map, "min_prob", &fusion_config.min_prob,
                  fusion_config.min_prob);
  setParam<float>(param_map, "max_prob", &fusion_config.max_prob,
                  fusion_config.max_prob);
  setParam<float>(param_map, "decay_weight_std",
                  &fusion_config.decay_weight_std,
                  fusion_config.decay_weight_std);
  setParam<std::string>(param_map, "fusion_strategy",
                        &fusion_config.fusion_strategy,
                        fusion_config.fusion_strategy);
  setParam<bool>(param_map, "use_ssc_planning", &use_ssc_planning_, false);
  setParam<bool>(param_map, "use_ssc_information_planning",
                 &use_ssc_information_planning_, true);
  setParam<bool>(param_map, "use_voxblox_planning", &use_voxblox_planning_,
                 true);
  setParam<bool>(param_map, "use_voxblox_information_planning",
                 &use_voxblox_information_planning_, true);

  // setup ssc server
  ros::NodeHandle ssc_nh("ssc_map");
  ros::NodeHandle ssc_nh_private("~ssc_map");
  ssc_server_.reset(new voxblox::SSCServer(ssc_nh, ssc_nh_private,
                                           fusion_config, map_config));

  // setup esdf server.
  ros::NodeHandle nh("voxblox_map");
  ros::NodeHandle nh_private("~voxblox_map");
  auto esdf_config = voxblox::getEsdfMapConfigFromRosParam(nh_private);
  auto tsdf_config = voxblox::getTsdfMapConfigFromRosParam(nh_private);
  auto mesh_config = voxblox::getMeshIntegratorConfigFromRosParam(nh_private);
  auto esdf_integrator_config =
      voxblox::getEsdfIntegratorConfigFromRosParam(nh_private);
  auto tsdf_integrator_config =
      voxblox::getTsdfIntegratorConfigFromRosParam(nh_private);

  // TEST
  nh_ = ros::NodeHandle();
  vis_pub_ = nh_.advertise<visualization_msgs::Marker>("ssc_visualization", 10);
  vis_timer_ = nh_.createTimer(ros::Duration(0.5),
                               &SSCVoxbloxOccupancyMap::visCallback, this);

  // update tsdf voxel size to match ssc voxel map
  setParam<float>(param_map, "voxel_size", &tsdf_config.tsdf_voxel_size,
                  tsdf_config.tsdf_voxel_size);

  // setup ESDF Server
  esdf_server_.reset(new voxblox::EsdfServer(
      nh, nh_private, esdf_config, esdf_integrator_config, tsdf_config,
      tsdf_integrator_config, mesh_config));
  esdf_server_->setTraversabilityRadius(
      planner_.getSystemConstraints().collision_radius);

  // cache constants
  c_voxel_size_ = ssc_server_->getSSCMapPtr()->voxel_size();
  c_block_size_ = ssc_server_->getSSCMapPtr()->block_size();
  c_occupied_confidence_threshold_ =
      voxblox::logOddsFromProbability(0.5f + ssc_confidence_threshold_);
  c_free_confidence_threshold_ =
      voxblox::logOddsFromProbability(0.5f - ssc_confidence_threshold_);
}

void SSCVoxbloxOccupancyMap::visCallback(const ros::TimerEvent &e) {
  // Iterate over all unobserved ssc predictions and visualize them.

  // Setup msg.
  visualization_msgs::Marker msg;
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = "world";
  msg.action = visualization_msgs::Marker::MODIFY;
  msg.type = visualization_msgs::Marker::CUBE_LIST;
  msg.id = 0;
  msg.scale.x = 0.08;
  msg.scale.y = 0.08;
  msg.scale.z = 0.08;
  const float occ_prob = voxblox::logOddsFromProbability(0.5);
  const float alpha = 0.5;
  msg.pose.orientation.w = 1;

  const auto &layer = ssc_server_->getSSCMapPtr()->getSSCLayer();
  voxblox::BlockIndexList indices;
  layer.getAllAllocatedBlocks(&indices);
  const int num_voxels = layer.voxels_per_side() * layer.voxels_per_side() *
                         layer.voxels_per_side();
  const voxblox::SSCColorMap color_map;

  // Iterate over all voxels
  for (const auto &index : indices) {
    const auto &block = layer.getBlockByIndex(index);
    for (int i = 0; i < num_voxels; ++i) {
      const voxblox::SSCOccupancyVoxel &voxel = block.getVoxelByLinearIndex(i);
      if (!voxel.observed || voxel.probability_log < occ_prob) {
        continue;
      }
      // Check observed in measured map.
      const voxblox::Point position =
          block.computeCoordinatesFromLinearIndex(i);
      // if (position.z() < -0.1) {
      //   continue;
      // }
      const auto *tsdf_voxel = esdf_server_->getTsdfMapPtr()
                                   ->getTsdfLayer()
                                   .getVoxelPtrByCoordinates(position);
      if (tsdf_voxel) {
        if (tsdf_voxel->weight >= 1e-6) {
          continue;
        }
      }

      // Add point.
      geometry_msgs::Point point;
      point.x = position.x();
      point.y = position.y();
      point.z = position.z();
      msg.points.push_back(point);
      std_msgs::ColorRGBA color;
      voxblox::Color vox_color = color_map.colorLookup(voxel.label);
      color.r = vox_color.r / 255.f;
      color.g = vox_color.g / 255.f;
      color.b = vox_color.b / 255.f;
      color.a = alpha;
      msg.colors.push_back(color);
    }
  }
  vis_pub_.publish(msg);
}

bool SSCVoxbloxOccupancyMap::isTraversable(
    const Eigen::Vector3d &position, const Eigen::Quaterniond &orientation) {
  double collision_radius = planner_.getSystemConstraints().collision_radius;

  if (use_voxblox_planning_) {
    // first check from voxblox esdf
    double distance = 0.0;
    if (esdf_server_->getEsdfMapPtr()->getDistanceAtPosition(position,
                                                             &distance)) {
      // This means the voxel is observed
      return (distance > collision_radius);
    }
  }

  if (use_ssc_planning_) {
    // The voxel is not observed by voxblox tsdf map. In this case
    // check SSC Map
    // The criteria to use ssc map is met.
    std::vector<Eigen::Vector3d> neighbouring_points;
    const bool clearing =
        position.norm() <
        1.f; // NOTE: this is currently fixed for prototyping, clean this up.
    voxblox::utils::getSurroundingVoxelsSphere(
        position, c_voxel_size_, collision_radius, &neighbouring_points);
    for (auto point : neighbouring_points) {
      auto state = getVoxelState(point);
      if (state == OccupancyMap::OCCUPIED) {
        return false;
      } else if (state == OccupancyMap::UNKNOWN && !clearing) {
        // Double check clearing radius here since this is otherwise not
        // checked.
        return false;
      }
    }
    return true;
  }
  return false;
}

bool SSCVoxbloxOccupancyMap::isObserved(const Eigen::Vector3d &point) {
  if (use_voxblox_planning_) {
    if (esdf_server_->getEsdfMapPtr()->isObserved(point)) {
      return true;
    }
  }
  if (use_ssc_planning_) {
    if (ssc_confidence_threshold_ == 0.f) {
      return ssc_server_->getSSCMapPtr()->isObserved(point);
    } else {
      const voxblox::SSCOccupancyVoxel *voxel =
          ssc_server_->getSSCMapPtr()->getSSCLayer().getVoxelPtrByCoordinates(
              point.cast<voxblox::FloatingPoint>());
      if (!voxel) {
        return false;
      }
      return voxel->observed &&
             (voxel->probability_log >= c_occupied_confidence_threshold_ ||
              voxel->probability_log <= c_free_confidence_threshold_);
    }
  }
  return false;
}

double SSCVoxbloxOccupancyMap::getVoxelLogProb(const Eigen::Vector3d &point) {
  voxblox::Point voxblox_point(point.x(), point.y(), point.z());
  voxblox::Block<voxblox::SSCOccupancyVoxel>::Ptr block =
      ssc_server_->getSSCMapPtr()->getSSCLayerPtr()->getBlockPtrByCoordinates(
          voxblox_point);
  if (block) {
    voxblox::SSCOccupancyVoxel *ssc_voxel =
        block->getVoxelPtrByCoordinates(voxblox_point);
    if (ssc_voxel) {
      return ssc_voxel->probability_log;
    }
  }
  return 0.0;
}

// get occupancy
unsigned char
SSCVoxbloxOccupancyMap::getVoxelState(const Eigen::Vector3d &point) {
  double distance = 0.0;
  if (use_voxblox_information_planning_) {
    if (esdf_server_->getEsdfMapPtr()->getDistanceAtPosition(point,
                                                             &distance)) {
      // This means the voxel is observed by ESDF Map
      if (distance < c_voxel_size_) {
        return OccupancyMap::OCCUPIED;
      } else {
        return OccupancyMap::FREE;
      }
    }
  }

  if (use_ssc_information_planning_) {
    // voxel is not observed by ESDF Map. See if its observed by SSC Map.
    return getVoxelSSCState(point);
  }
  return OccupancyMap::UNKNOWN;
}

unsigned char
SSCVoxbloxOccupancyMap::getVoxelSSCState(const Eigen::Vector3d &point) {
  auto voxel =
      ssc_server_->getSSCMapPtr()->getSSCLayerPtr()->getVoxelPtrByCoordinates(
          point.cast<voxblox::FloatingPoint>());

  if (voxel == nullptr)
    return OccupancyMap::UNKNOWN;

  if (voxel->observed) {
    if (ssc_confidence_threshold_ == 0.f) {
      return voxel->probability_log > voxblox::logOddsFromProbability(0.5f)
                 ? OccupancyMap::OCCUPIED
                 : OccupancyMap::FREE;
    } else {
      if (voxel->probability_log >= c_occupied_confidence_threshold_) {
        return OccupancyMap::OCCUPIED;
      } else if (voxel->probability_log <= c_free_confidence_threshold_) {
        return OccupancyMap::FREE;
      }
    }
  }
  return OccupancyMap::UNKNOWN;
}

// get voxel size
double SSCVoxbloxOccupancyMap::getVoxelSize() { return c_voxel_size_; }

// get the center of a voxel from input point
bool SSCVoxbloxOccupancyMap::getVoxelCenter(Eigen::Vector3d *center,
                                            const Eigen::Vector3d &point) {
  voxblox::BlockIndex block_id = ssc_server_->getSSCMapPtr()
                                     ->getSSCLayerPtr()
                                     ->computeBlockIndexFromCoordinates(
                                         point.cast<voxblox::FloatingPoint>());
  *center = voxblox::getOriginPointFromGridIndex(block_id, c_block_size_)
                .cast<double>();
  voxblox::VoxelIndex voxel_id =
      voxblox::getGridIndexFromPoint<voxblox::VoxelIndex>(
          (point - *center).cast<voxblox::FloatingPoint>(),
          1.0 / c_voxel_size_);
  *center += voxblox::getCenterPointFromGridIndex(voxel_id, c_voxel_size_)
                 .cast<double>();
  return true;
}

} // namespace map
} // namespace active_3d_planning
