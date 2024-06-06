#include "ssc_planning/trajectory_evaluator/ssc_exploration_evaluator.h"

#include <algorithm>
#include <vector>

namespace active_3d_planning {
namespace trajectory_evaluator {

ModuleFactoryRegistry::Registration<SSCExplorationEvaluator>
    SSCExplorationEvaluator::registration("SSCExplorationEvaluator");

SSCExplorationEvaluator::SSCExplorationEvaluator(PlannerI &planner)
    : SimulatedSensorEvaluator(planner) {}

void SSCExplorationEvaluator::setupFromParamMap(Module::ParamMap *param_map) {
  SimulatedSensorEvaluator::setupFromParamMap(param_map);
  setParam<double>(param_map, "predicted_occ_weight", &p_predicted_occ_weight_,
                   1.0);
  setParam<double>(param_map, "predicted_free_weight",
                   &p_predicted_free_weight_, 1.0);
  setParam<double>(param_map, "unobserved_weight", &p_unobserved_weight_, 1.0);
  setParam<bool>(param_map, "weight_by_confidence", &p_weight_by_confidence_,
                 false);

  // setup map
  map_ = dynamic_cast<map::SSCVoxbloxOccupancyMap *>(&(planner_.getMap()));
  if (!map_) {
    planner_.printError("'SSCExplorationEvaluator' requires a map of type "
                        "'SSCVoxbloxOccupancyMap'!");
  }

  // cache voxblox constants
  c_voxel_size_ = map_->getVoxelSize();
  c_gains_ = {0, p_predicted_occ_weight_, p_predicted_free_weight_,
              p_unobserved_weight_};
}

bool SSCExplorationEvaluator::storeTrajectoryInformation(
    TrajectorySegment *traj_in,
    const std::vector<Eigen::Vector3d> &new_voxels) {
  // Uses the default voxel info, not much gain from caching more info
  return SimulatedSensorEvaluator::storeTrajectoryInformation(traj_in,
                                                              new_voxels);
}

bool SSCExplorationEvaluator::computeGainFromVisibleVoxels(
    TrajectorySegment *traj_in) {
  traj_in->gain = 0.0;
  if (!traj_in->info) {
    return false;
  }
  SimulatedSensorInfo *info =
      reinterpret_cast<SimulatedSensorInfo *>(traj_in->info.get());

  for (const Eigen::Vector3d voxel : info->visible_voxels) {
    float weight = 1.f;
    if (p_weight_by_confidence_) {
      const voxblox::SSCOccupancyVoxel *ssc_voxel =
          map_->getSSCServer()
              .getSSCMapPtr()
              ->getSSCLayer()
              .getVoxelPtrByCoordinates(voxel.cast<voxblox::FloatingPoint>());
      if (ssc_voxel) {
        const float confidence =
            voxblox::probabilityFromLogOdds(ssc_voxel->probability_log);
        weight = 2.f * std::abs(confidence - 0.5f);
      }
    }
    traj_in->gain += c_gains_[getVoxelType(voxel)] * weight;
  }
  return true;
}

int SSCExplorationEvaluator::getVoxelType(const Eigen::Vector3d &voxel) {
  // The voxel is already observed, don't consider it in calculating gain.
  if (map_->getESDFServer().getEsdfMapPtr()->isObserved(voxel)) {
    return 0;
  }

  // voxel not observed in measured map
  unsigned char voxel_state = map_->getVoxelSSCState(voxel);
  if (voxel_state == map::OccupancyMap::OCCUPIED) {
    return 1;
  } else if (voxel_state == map::OccupancyMap::FREE) {
    return 2;
  }
  return 3;
}

void SSCExplorationEvaluator::visualizeTrajectoryValue(
    VisualizationMarkers *markers, const TrajectorySegment &trajectory) {
  // Display all voxels that contribute to the gain. max_impact-min_impact as
  // green-red, frontier voxels purple, unknwon voxels teal
  if (!trajectory.info) {
    return;
  }
  VisualizationMarker marker;
  marker.type = VisualizationMarker::CUBE_LIST;
  marker.scale.x() = c_voxel_size_;
  marker.scale.y() = c_voxel_size_;
  marker.scale.z() = c_voxel_size_;

  // points
  SimulatedSensorInfo *info =
      reinterpret_cast<SimulatedSensorInfo *>(trajectory.info.get());
  for (const Eigen::Vector3d voxel : info->visible_voxels) {
    int voxel_type = getVoxelType(voxel);
    if (c_gains_[voxel_type] > 0.f) {
      marker.points.push_back(voxel);
      Color color;
      if (voxel_type == 1) {
        color.r = 1.0;
        color.g = 0.0;
        color.b = 0.0;
        color.a = 0.5;
      } else if (voxel_type == 2) {
        color.r = 0.0;
        color.g = 0.75;
        color.b = 1.0;
        color.a = 0.1;
      } else {
        color.r = 1.0;
        color.g = 0.75;
        color.b = 0.0;
        color.a = 0.5;
      }
      marker.colors.push_back(color);
    }
  }
  markers->addMarker(marker);

  if (p_visualize_sensor_view_) {
    sensor_model_->visualizeSensorView(markers, trajectory);
  }
}

} // namespace trajectory_evaluator
} // namespace active_3d_planning
