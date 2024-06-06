#ifndef SSC_VOXBLOX_3D_PLANNING_CRITERIA_MAP_H_
#define SSC_VOXBLOX_3D_PLANNING_CRITERIA_MAP_H_

#include <memory>

#include <active_3d_planning_core/module/module_factory_registry.h>
#include <ssc_mapping/ros/ssc_server.h>
#include <ssc_mapping/utils/voxel_utils.h>
#include <voxblox_ros/esdf_server.h>
#include <active_3d_planning_core/map/occupancy_map.h>
#include "ssc_planning/map/ssc_voxblox_map.h"

namespace active_3d_planning {
namespace map {

namespace ssc_utilization_criterias {
  std::string confidence = "confidence";
}

/**
 * Base class to check for criteria to match inorder to use
 * predicted map
 */
class BaseCriteria {
   public:
    virtual bool criteriaVerify(const voxblox::SSCMap & ssc_map, const Eigen::Vector3d& position) = 0;
};

class ConfidenceCriteria : public BaseCriteria {
   public:
    ConfidenceCriteria(const float confidence_threshold) : confidence_threshold_(confidence_threshold) {}

    virtual bool criteriaVerify(const voxblox::SSCMap & ssc_map, const Eigen::Vector3d& position);

   private:
    float confidence_threshold_;
};


/**
 * SSC + Voxblox as a map representation. Use SSC Predicted map if
 * criteria is met else use measured map
 */
class SSCVoxbloxCriteriaMap : public SSCVoxbloxOccupancyMap {
   public:
    explicit SSCVoxbloxCriteriaMap(PlannerI& planner);

    // implement virtual methods
    void setupFromParamMap(Module::ParamMap* param_map) override;

    // check collision for a single pose
    bool isTraversable(const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation) override;

    // check whether point is part of the map
    bool isObserved(const Eigen::Vector3d& point) override;

    // get occupancy
    unsigned char getVoxelState(const Eigen::Vector3d& point) override;

   protected:
    static ModuleFactoryRegistry::Registration<SSCVoxbloxCriteriaMap> registration;

    // use criteria for utilizing predicted ssc map
    std::unique_ptr<BaseCriteria> ssc_utilization_criteria_;
};

}  // namespace map
}  // namespace active_3d_planning

#endif  // SSC_VOXBLOX_3D_PLANNING_CRITERIA_MAP_H_
