#ifndef SSC_VOXBLOX_3D_PLANNING_MAP_H_
#define SSC_VOXBLOX_3D_PLANNING_MAP_H_

#include <memory>

#include <active_3d_planning_core/map/occupancy_map.h>
#include <active_3d_planning_core/module/module_factory_registry.h>
#include <ssc_mapping/ros/ssc_server.h>
#include <ssc_mapping/utils/voxel_utils.h>
#include <voxblox_ros/esdf_server.h>

namespace active_3d_planning {
namespace map {

/**
 * SSC + Voxblox as a map representation. First use esdf map,
 * in case the voxel is not observed, falls back to SSC Map.
 */
class SSCVoxbloxOccupancyMap : public OccupancyMap {
public:
  explicit SSCVoxbloxOccupancyMap(PlannerI &planner);

  // implement virtual methods
  void setupFromParamMap(Module::ParamMap *param_map) override;

  // check collision for a single pose
  bool isTraversable(const Eigen::Vector3d &position,
                     const Eigen::Quaterniond &orientation) override;

  // check whether point is part of the map
  bool isObserved(const Eigen::Vector3d &point) override;

  // get occupancy
  unsigned char getVoxelState(const Eigen::Vector3d &point) override;
  virtual unsigned char getVoxelSSCState(const Eigen::Vector3d &point);

  // get voxel size
  double getVoxelSize() override;

  // get the center of a voxel from input point
  bool getVoxelCenter(Eigen::Vector3d *center,
                      const Eigen::Vector3d &point) override;

  // get the voxel occupancy probability in LogOdds
  double getVoxelLogProb(const Eigen::Vector3d &point);

  // accessor to the servers for specialized planners
  voxblox::SSCServer &getSSCServer();

  voxblox::EsdfServer &getESDFServer();

protected:
  static ModuleFactoryRegistry::Registration<SSCVoxbloxOccupancyMap>
      registration;

  // esdf server that contains the map, subscribe to external ESDF/TSDF updates
  std::unique_ptr<voxblox::SSCServer> ssc_server_;

  std::unique_ptr<voxblox::EsdfServer> esdf_server_;

  // TEST
  ros::NodeHandle nh_;
  ros::Publisher vis_pub_;
  ros::Timer vis_timer_;
  void visCallback(const ros::TimerEvent& e);

  // use ssc map for planning
  bool use_ssc_planning_;

  // use measured voxblox measured map for planning
  bool use_voxblox_planning_;

  // whether to use ssc map for information planning
  bool use_ssc_information_planning_;

  // use measured voxblox measured map for information planning
  bool use_voxblox_information_planning_;

  // Extra confidence required for a SSC voxel to be observed (min: 0 -> 0.5
  // max)
  float ssc_confidence_threshold_ = 0.f;

  // cache constants
  double c_voxel_size_;
  double c_block_size_;
  float c_occupied_confidence_threshold_;
  float c_free_confidence_threshold_;
};

} // namespace map
} // namespace active_3d_planning

#endif // SSC_VOXBLOX_3D_PLANNING_MAP_H_
