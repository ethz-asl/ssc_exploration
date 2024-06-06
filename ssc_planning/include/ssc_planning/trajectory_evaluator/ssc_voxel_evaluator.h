#ifndef SSC_PLANNING_VOXEL_EVALUATOR_H_
#define SSC_PLANNING_VOXEL_EVALUATOR_H_

#include <vector>

//#include <active_3d_planning_core/map/tsdf_map.h>
#include "ssc_planning/map/ssc_occ_map.h"
#include "ssc_planning/map/ssc_voxblox_map.h"
#include <active_3d_planning_core/module/trajectory_evaluator/frontier_evaluator.h>

namespace active_3d_planning {
namespace trajectory_evaluator {

// SSCVoxelEvaluator uses the voxel log prob to estimate how much they
// can still change with additional observations. Requires the SSC map to
// published to the planner intern voxblox ssc server. Uses frontier voxels to allow
// additional gain for exploration. Observed voxels contribute between [0, 1] per
// voxel, based on the expectd impact. Unobserved voxels contribute 1x
// new_voxel_weight per voxel.
class SSCVoxelEvaluator : public FrontierEvaluator {
 public:
  explicit SSCVoxelEvaluator(PlannerI& planner);  // NOLINT

  // Override virtual methods
  void visualizeTrajectoryValue(VisualizationMarkers* markers,
                                const TrajectorySegment& trajectory) override;

  void setupFromParamMap(Module::ParamMap* param_map) override;

 protected:
  static ModuleFactoryRegistry::Registration<SSCVoxelEvaluator> registration;

  // Override virtual methods
  bool storeTrajectoryInformation(
      TrajectorySegment* traj_in,
      const std::vector<Eigen::Vector3d>& new_voxels) override;

  bool computeGainFromVisibleVoxels(TrajectorySegment* traj_in) override;

  // map
  map::SSCVoxbloxOccupancyMap * map_;

  // params
  double p_new_measured_voxel_weight_; // weight for voxel observed in sscmap but not observed in measured tsdf map
  double p_min_impact_factor_;  // Minimum expected change, the gain is set at 0
  // here.
  double p_new_voxel_weight_;  // Multiply unobserved voxels by this weight to
  // balance quality/exploration
  double p_frontier_voxel_weight_;  // Multiply frontier voxels by this weight
  double p_max_log_prob_;  // Max log probability for a voxel, beyond which its not considered in gain
  // weight of log prob for gain computation
  double p_log_prob_weight_;

  // constants
  double c_voxel_size_;

  // methods
  double getVoxelValue(const Eigen::Vector3d& voxel,
                       const Eigen::Vector3d& origin);
};

}  // namespace trajectory_evaluator
}  // namespace active_3d_planning
#endif  // SSC_PLANNING_VOXEL_EVALUATOR_H_
