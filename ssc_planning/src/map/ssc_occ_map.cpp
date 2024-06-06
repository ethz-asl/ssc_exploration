#include "ssc_planning/map/ssc_occ_map.h"

#include <voxblox/core/common.h>
#include <voxblox_ros/ros_params.h>
#include <active_3d_planning_core/data/system_constraints.h>

namespace active_3d_planning {
namespace map {

ModuleFactoryRegistry::Registration<SSCOccupancyMap> SSCOccupancyMap::registration(
    "SSCOccupancyMap");

SSCOccupancyMap::SSCOccupancyMap(PlannerI& planner) : OccupancyMap(planner) {}

voxblox::SSCServer& SSCOccupancyMap::getSSCServer() { return *ssc_server_; }

void SSCOccupancyMap::setupFromParamMap(Module::ParamMap* param_map) {
  // create an esdf server
  ros::NodeHandle nh("");
  ros::NodeHandle nh_private("~");

  voxblox::SSCMap::Config map_config;
  ssc_fusion::BaseFusion::Config fusion_config;

  // load ssc map config
  setParam<float>(param_map, "voxel_size", &map_config.ssc_voxel_size, map_config.ssc_voxel_size);

  // load fusion config
  setParam<float>(param_map, "pred_conf", &fusion_config.pred_conf, fusion_config.pred_conf);
  setParam<float>(param_map, "max_weight", &fusion_config.max_weight, fusion_config.max_weight);
  setParam<float>(param_map, "prob_occupied", &fusion_config.prob_occupied, fusion_config.prob_occupied);
  setParam<float>(param_map, "prob_free", &fusion_config.prob_free, fusion_config.prob_free);
  setParam<float>(param_map, "min_prob", &fusion_config.min_prob, fusion_config.min_prob);
  setParam<float>(param_map, "max_prob", &fusion_config.max_prob, fusion_config.max_prob);
  setParam<float>(param_map, "decay_weight_std", &fusion_config.decay_weight_std, fusion_config.decay_weight_std);
  setParam<std::string>(param_map, "fusion_strategy", &fusion_config.fusion_strategy, fusion_config.fusion_strategy);
  ssc_server_.reset(new voxblox::SSCServer(nh, nh_private, fusion_config, map_config));
  //esdf_server_.reset(new voxblox::EsdfServer(nh, nh_private));

  // cache constants
  c_voxel_size_ = ssc_server_->getSSCMapPtr()->voxel_size();
  c_block_size_ = ssc_server_->getSSCMapPtr()->block_size();

//   ssc_pointcloud_pub_ =
//             nh_private.advertise<pcl::PointCloud<pcl::PointXYZRGB> >("missing_esdf_pointcloud", 1, true);
}

bool SSCOccupancyMap::isTraversable(const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation) {
    double collision_radius = planner_.getSystemConstraints().collision_radius;
    double clearing_radius = 0.2;

    // get enclosing sphere
    std::vector<Eigen::Vector3d> neighbouring_points;
    voxblox::utils::getSurroundingVoxelsSphere(position, c_voxel_size_, collision_radius, &neighbouring_points);

    for (auto point : neighbouring_points) {
        auto voxel_state = getVoxelState(point);

        bool is_within_clearing_radius = (planner_.getCurrentPosition() - position).norm()  < clearing_radius;
        
        // ignore unknown points within clearing radius
        if (voxel_state == OccupancyMap::UNKNOWN && is_within_clearing_radius)
            continue;

        if (voxel_state != OccupancyMap::FREE) {
            return false;
        }
    }
    return true;
}

bool SSCOccupancyMap::isObserved(const Eigen::Vector3d& point) {
  return ssc_server_->getSSCMapPtr()->isObserved(point);
}

// get occupancy
unsigned char SSCOccupancyMap::getVoxelState(const Eigen::Vector3d& point) {
    auto voxel =
        ssc_server_->getSSCMapPtr()->getSSCLayerPtr()->getVoxelPtrByCoordinates(point.cast<voxblox::FloatingPoint>());

    if (voxel == nullptr) 
      return OccupancyMap::UNKNOWN;

    if (voxel->observed) {
        if (voxel->probability_log > voxblox::logOddsFromProbability(0.5f)) {  // log(0.7/0.3) = 0.3679f
            return OccupancyMap::OCCUPIED;
        } else {
            return OccupancyMap::FREE;
        }
    } else {
        return OccupancyMap::UNKNOWN;
    }
}

// get voxel size
double SSCOccupancyMap::getVoxelSize() { return c_voxel_size_; }

// get the center of a voxel from input point
bool SSCOccupancyMap::getVoxelCenter(Eigen::Vector3d* center,
                                const Eigen::Vector3d& point) {
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

}  // namespace map
}  // namespace active_3d_planning
