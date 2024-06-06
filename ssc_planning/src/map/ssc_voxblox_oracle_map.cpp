#include "ssc_planning/map/ssc_voxblox_oracle_map.h"

#include <active_3d_planning_core/data/system_constraints.h>

namespace active_3d_planning {
namespace map {

ModuleFactoryRegistry::Registration<SSCVoxbloxOracleMap>
    SSCVoxbloxOracleMap::registration("SSCVoxbloxOracleMap");

SSCVoxbloxOracleMap::SSCVoxbloxOracleMap(PlannerI &planner)
    : SSCVoxbloxOccupancyMap(planner) {}

void SSCVoxbloxOracleMap::setupFromParamMap(Module::ParamMap *param_map) {
  // Load standard voxblox SSC map, then change the differences.
  SSCVoxbloxOccupancyMap::setupFromParamMap(param_map);

  // Params.
  std::string ground_truth_layer_path;
  setParam<std::string>(param_map, "ground_truth_layer_path",
                        &ground_truth_layer_path, ground_truth_layer_path);
  setParam<bool>(param_map, "quality_only", &p_quality_only_, false);

  // No SSC server needed.
  if (!p_quality_only_) {
    ssc_server_.reset();
  }

  // Load oracle GT layer.
  voxblox::io::LoadLayer<voxblox::TsdfVoxel>(ground_truth_layer_path,
                                             &ground_truth_layer_);

  // TEST tmp fix.
  c_voxel_size_ = 0.08;
  c_block_size_ = 0.08 * 16;
}

bool SSCVoxbloxOracleMap::isTraversable(const Eigen::Vector3d &position,
                                        const Eigen::Quaterniond &orientation) {
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
    // check oracle map.
    std::vector<Eigen::Vector3d> neighbouring_points;
    voxblox::utils::getSurroundingVoxelsSphere(
        position, c_voxel_size_, collision_radius, &neighbouring_points);
    for (auto point : neighbouring_points) {
      if (p_quality_only_) {
        // Non-clearing unobserved points are intraversable.
        if (point.norm() > 1.f &&
            !ssc_server_->getSSCMapPtr()->isObserved(point)) {
          return false;
        }
      }
      // Look-up in GT map.
      voxblox::Block<voxblox::TsdfVoxel>::Ptr block =
          ground_truth_layer_->getBlockPtrByCoordinates(
              point.cast<voxblox::FloatingPoint>());
      if (!block) {
        return false;
      }
      const voxblox::TsdfVoxel &voxel =
          block->getVoxelByCoordinates(point.cast<voxblox::FloatingPoint>());
      if (voxel.weight <= 1e-6 || voxel.distance < c_voxel_size_) {
        return false;
      }
    }
    return true;
  }
  return false;
}

bool SSCVoxbloxOracleMap::isObserved(const Eigen::Vector3d &point) {
  if (use_voxblox_planning_) {
    if (esdf_server_) {
      auto esdf_map = esdf_server_->getEsdfMapPtr();
      if (esdf_map) {
        if (esdf_map->isObserved(point)) {
          return true;
        }
      }
    }
  }
  if (use_ssc_planning_) {
    if (p_quality_only_) {
      return ssc_server_->getSSCMapPtr()->isObserved(point);
    }
    // Check GT Layer.
    voxblox::Block<voxblox::TsdfVoxel>::Ptr block =
        ground_truth_layer_->getBlockPtrByCoordinates(
            point.cast<voxblox::FloatingPoint>());
    if (block) {
      if (block->getVoxelByCoordinates(point.cast<voxblox::FloatingPoint>())
              .weight >= 1e-6) {
        return true;
      }
    }
  }
  return false;
}

double SSCVoxbloxOracleMap::getVoxelLogProb(const Eigen::Vector3d &point) {
  return voxblox::logOddsFromProbability(0.999999); // GT map.
}

// get occupancy
unsigned char SSCVoxbloxOracleMap::getVoxelState(const Eigen::Vector3d &point) {
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
SSCVoxbloxOracleMap::getVoxelSSCState(const Eigen::Vector3d &point) {
  if (p_quality_only_) {
    if (!ssc_server_->getSSCMapPtr()->isObserved(point)) {
      return OccupancyMap::UNKNOWN;
    }
  }

  // Look up GT map.
  voxblox::Block<voxblox::TsdfVoxel>::Ptr block =
      ground_truth_layer_->getBlockPtrByCoordinates(
          point.cast<voxblox::FloatingPoint>());
  if (block) {
    const voxblox::TsdfVoxel &voxel =
        block->getVoxelByCoordinates(point.cast<voxblox::FloatingPoint>());
    if (voxel.weight <= 1e-6) {
      return OccupancyMap::UNKNOWN;
    } else if (voxel.distance <= c_voxel_size_) {
      return OccupancyMap::OCCUPIED;
    } else {
      return OccupancyMap::FREE;
    }
  }
  return OccupancyMap::UNKNOWN;
}

// get the center of a voxel from input point
bool SSCVoxbloxOracleMap::getVoxelCenter(Eigen::Vector3d *center,
                                         const Eigen::Vector3d &point) {
  voxblox::BlockIndex block_id = esdf_server_->getEsdfMapPtr()
                                     ->getEsdfLayer()
                                     .computeBlockIndexFromCoordinates(
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
