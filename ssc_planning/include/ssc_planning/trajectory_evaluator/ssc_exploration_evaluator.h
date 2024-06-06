#ifndef SSC_PLANNING_TRAJECTORY_EVALUATOR_SSC_EXPLORATION_EVALUATOR_H_
#define SSC_PLANNING_TRAJECTORY_EVALUATOR_SSC_EXPLORATION_EVALUATOR_H_

#include <vector>

//#include <active_3d_planning_core/map/tsdf_map.h>
#include "ssc_planning/map/ssc_occ_map.h"
#include "ssc_planning/map/ssc_voxblox_map.h"
#include <active_3d_planning_core/module/trajectory_evaluator/simulated_sensor_evaluator.h>

namespace active_3d_planning {
namespace trajectory_evaluator {

// This evaluator computes a standard exploration gain, similar to the
// NaiveEvaluator, but taking into consideration various functions of the SSC
// map.
class SSCExplorationEvaluator : public SimulatedSensorEvaluator {
public:
  explicit SSCExplorationEvaluator(PlannerI &planner); // NOLINT

  // Override virtual methods
  void visualizeTrajectoryValue(VisualizationMarkers *markers,
                                const TrajectorySegment &trajectory) override;

  void setupFromParamMap(Module::ParamMap *param_map) override;

protected:
  static ModuleFactoryRegistry::Registration<SSCExplorationEvaluator>
      registration;

  // Override virtual methods
  bool storeTrajectoryInformation(
      TrajectorySegment *traj_in,
      const std::vector<Eigen::Vector3d> &new_voxels) override;

  bool computeGainFromVisibleVoxels(TrajectorySegment *traj_in) override;

  // map
  map::SSCVoxbloxOccupancyMap *map_;

  // params
  double p_predicted_occ_weight_; 
  double p_predicted_free_weight_;
  double p_unobserved_weight_;
  bool p_weight_by_confidence_;

  // constants
  double c_voxel_size_;
  std::vector<double> c_gains_;

  // methods
  int getVoxelType(const Eigen::Vector3d
                       &voxel); // 0-obs, 1-pred_occ, 2-pred_free, 3-unknown
};

} // namespace trajectory_evaluator
} // namespace active_3d_planning
#endif // SSC_PLANNING_TRAJECTORY_EVALUATOR_SSC_EXPLORATION_EVALUATOR_H_
