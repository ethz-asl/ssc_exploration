#include "ssc_planning/trajectory_evaluator/ssc_voxel_evaluator.h"

#include <algorithm>
#include <vector>

namespace active_3d_planning {
namespace trajectory_evaluator {

ModuleFactoryRegistry::Registration<SSCVoxelEvaluator>
    SSCVoxelEvaluator::registration("SSCVoxelEvaluator");

SSCVoxelEvaluator::SSCVoxelEvaluator(PlannerI& planner)
    : FrontierEvaluator(planner) {}

void SSCVoxelEvaluator::setupFromParamMap(Module::ParamMap* param_map) {
  FrontierEvaluator::setupFromParamMap(param_map);
  setParam<double>(param_map, "frontier_voxel_weight",
                   &p_frontier_voxel_weight_, 1.5);
  setParam<double>(param_map, "min_impact_factor", &p_min_impact_factor_, 0.0);
  setParam<double>(param_map, "new_measured_voxel_weight", &p_new_measured_voxel_weight_, 0.2);
  setParam<double>(param_map, "new_voxel_weight", &p_new_voxel_weight_, 1.0);
  setParam<double>(param_map, "voxel_log_prob_weight", &p_log_prob_weight_, 0.2);
  setParam<double>(param_map, "max_log_prob", &p_max_log_prob_, voxblox::logOddsFromProbability(0.9f));

  // setup map
  map_ = dynamic_cast<map::SSCVoxbloxOccupancyMap*>(&(planner_.getMap()));
  if (!map_) {
    planner_.printError(
        "'SSCVoxelEvaluator' requires a map of type 'SSCVoxbloxOccupancyMap'!");
  }

  // cache voxblox constants
  c_voxel_size_ = map_->getVoxelSize();
}

bool SSCVoxelEvaluator::storeTrajectoryInformation(
    TrajectorySegment* traj_in,
    const std::vector<Eigen::Vector3d>& new_voxels) {
  // Uses the default voxel info, not much gain from caching more info
  return SimulatedSensorEvaluator::storeTrajectoryInformation(traj_in,
                                                              new_voxels);
}

bool SSCVoxelEvaluator::computeGainFromVisibleVoxels(
    TrajectorySegment* traj_in) {
  traj_in->gain = 0.0;
  if (!traj_in->info) {
    return false;
  }
  SimulatedSensorInfo* info =
      reinterpret_cast<SimulatedSensorInfo*>(traj_in->info.get());

  // just assume we take a single image from the last trajectory point here...
  Eigen::Vector3d origin = traj_in->trajectory.back().position_W;
  for (int i = 0; i < info->visible_voxels.size(); ++i) {
    traj_in->gain += getVoxelValue(info->visible_voxels[i], origin);
  }
  return true;
}

double SSCVoxelEvaluator::getVoxelValue(const Eigen::Vector3d& voxel, const Eigen::Vector3d& origin) {
    // The voxel is already observed, don't consider it in calculating gain.
    if (map_->getESDFServer().getEsdfMapPtr()->isObserved(voxel)) {
        return 0;
    }
    // voxel not observed in measured map
    unsigned char voxel_state = map_->getVoxelState(voxel);
    if (voxel_state != map::OccupancyMap::UNKNOWN) {
        // voxel is observed in predicted map.
        // Note:
        // map_->getVoxelState(voxel) checks in both maps but the voxel
        // is not observed in measured map, it must be either free or
        // or occupied in predicted map
        double gain = p_log_prob_weight_ * (p_max_log_prob_ - abs(map_->getVoxelLogProb(voxel)));
        gain = std::max(gain, 0.0);
        if (voxel_state == map::OccupancyMap::FREE) {
            gain += p_new_measured_voxel_weight_;
        }
        if (gain > p_min_impact_factor_) {
            return gain;
        }
    } else {
        // unknown voxel in both measured and predicted map
        if (p_frontier_voxel_weight_ > 0.0) {
            if (isFrontierVoxel(voxel)) {
                return p_frontier_voxel_weight_;
            }
        }
        return p_new_voxel_weight_;
    }
    return 0;
}

void SSCVoxelEvaluator::visualizeTrajectoryValue(
    VisualizationMarkers* markers, const TrajectorySegment& trajectory) {
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
  double value;
  Eigen::Vector3d origin = trajectory.trajectory.back().position_W;
  SimulatedSensorInfo* info =
      reinterpret_cast<SimulatedSensorInfo*>(trajectory.info.get());
  for (int i = 0; i < info->visible_voxels.size(); ++i) {
    value = getVoxelValue(info->visible_voxels[i], origin);
    if (value > 0.0) {
      marker.points.push_back(info->visible_voxels[i]);
      Color color;
      if (value == p_frontier_voxel_weight_) {
        color.r = 0.6;
        color.g = 0.4;
        color.b = 1.0;
        color.a = 0.5;
      } else if (value == p_new_voxel_weight_) {
        color.r = 0.0;
        color.g = 0.75;
        color.b = 1.0;
        color.a = 0.1;
      } else {
        double frac =
            (value - p_min_impact_factor_) / (1.0 - p_min_impact_factor_);
        color.r = std::min((0.5 - frac) * 2.0 + 1.0, 1.0);
        color.g = std::min((frac - 0.5) * 2.0 + 1.0, 1.0);
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

}  // namespace trajectory_evaluator
}  // namespace active_3d_planning
