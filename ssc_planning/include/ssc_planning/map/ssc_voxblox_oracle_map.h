#ifndef SSC_VOXBLOX_ORACLE_PLANNING_MAP_H_
#define SSC_VOXBLOX_ORACLE_PLANNING_MAP_H_

#include <memory>

#include <active_3d_planning_core/map/occupancy_map.h>
#include <active_3d_planning_core/module/module_factory_registry.h>
#include <ssc_mapping/ros/ssc_server.h>
#include <ssc_mapping/utils/voxel_utils.h>
#include <voxblox_ros/esdf_server.h>

#include "ssc_planning/map/ssc_voxblox_map.h"

namespace active_3d_planning {
namespace map {

/**
 * SSC + Voxblox as a map representation. First use esdf map,
 * in case the voxel is not observed, falls back to SSC Map. This map uses the
 * ground truth SSC crovided by an 'oracle' for verification.
 */
class SSCVoxbloxOracleMap : public SSCVoxbloxOccupancyMap {
public:
  explicit SSCVoxbloxOracleMap(PlannerI &planner);

  // implement virtual methods
  void setupFromParamMap(Module::ParamMap *param_map) override;

  // check collision for a single pose
  bool isTraversable(const Eigen::Vector3d &position,
                     const Eigen::Quaterniond &orientation) override;

  // check whether point is part of the map
  bool isObserved(const Eigen::Vector3d &point) override;

  // get occupancy
  unsigned char getVoxelState(const Eigen::Vector3d &point) override;
  unsigned char getVoxelSSCState(const Eigen::Vector3d &point) override;

  // get the center of a voxel from input point
  bool getVoxelCenter(Eigen::Vector3d *center,
                      const Eigen::Vector3d &point) override;

  // get the voxel occupancy probability in LogOdds
  double getVoxelLogProb(const Eigen::Vector3d &point);

protected:
  static ModuleFactoryRegistry::Registration<SSCVoxbloxOracleMap> registration;

  // Ground truth map in-place of the SSC map
  voxblox::Layer<voxblox::TsdfVoxel>::Ptr ground_truth_layer_;

  bool p_quality_only_;
};

} // namespace map
} // namespace active_3d_planning

#endif // SSC_VOXBLOX_ORACLE_PLANNING_MAP_H_
