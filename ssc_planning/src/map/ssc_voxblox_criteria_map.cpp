#include "ssc_planning/map/ssc_voxblox_criteria_map.h"
#include <voxblox/core/common.h>
#include <voxblox_ros/ros_params.h>
#include <active_3d_planning_core/data/system_constraints.h>

namespace active_3d_planning {
namespace map {

ModuleFactoryRegistry::Registration<SSCVoxbloxCriteriaMap> SSCVoxbloxCriteriaMap::registration(
    "SSCVoxbloxCriteriaMap");

SSCVoxbloxCriteriaMap::SSCVoxbloxCriteriaMap(PlannerI& planner) : SSCVoxbloxOccupancyMap(planner) {}

void SSCVoxbloxCriteriaMap::setupFromParamMap(Module::ParamMap* param_map) {
    // create an esdf server
    ros::NodeHandle nh("");
    ros::NodeHandle nh_private("~");

    voxblox::SSCMap::Config map_config;
    ssc_fusion::BaseFusion::Config fusion_config;
    
    // ssc criteria params
    std::string ssc_criteria("confidence");
    float ssc_criteria_threshold = 0.92f;

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
    setParam<std::string>(param_map, "ssc_criteria", &ssc_criteria, ssc_criteria);
    setParam<float>(param_map, "criteria_threshold", &ssc_criteria_threshold, ssc_criteria_threshold);

    // setup ssc server
    ssc_server_.reset(new voxblox::SSCServer(nh, nh_private, fusion_config, map_config));

    // setup esdf server
    auto esdf_config = voxblox::getEsdfMapConfigFromRosParam(nh_private);
    auto tsdf_config = voxblox::getTsdfMapConfigFromRosParam(nh_private);
    auto mesh_config = voxblox::getMeshIntegratorConfigFromRosParam(nh_private);
    auto esdf_integrator_config = voxblox::getEsdfIntegratorConfigFromRosParam(nh_private);
    auto tsdf_integrator_config = voxblox::getTsdfIntegratorConfigFromRosParam(nh_private);

    // update tsdf voxel size to match ssc voxel map
    setParam<float>(param_map, "voxel_size", &tsdf_config.tsdf_voxel_size, tsdf_config.tsdf_voxel_size);

    // setup ESDF Server
    esdf_server_.reset(new voxblox::EsdfServer(nh, nh_private, esdf_config, esdf_integrator_config, tsdf_config,
                                               tsdf_integrator_config, mesh_config));
    esdf_server_->setTraversabilityRadius(planner_.getSystemConstraints().collision_radius);

    //setup criteria for ssc
    if (ssc_criteria.compare(ssc_utilization_criterias::confidence) == 0) {
        ssc_utilization_criteria_.reset(new ConfidenceCriteria(ssc_criteria_threshold));
    } else {
        LOG(WARNING) << "Wrong Criteria provided. Using default confidence criteria";
        ssc_utilization_criteria_.reset(new ConfidenceCriteria(ssc_criteria_threshold));
    }

    // cache constants
    c_voxel_size_ = ssc_server_->getSSCMapPtr()->voxel_size();
    c_block_size_ = ssc_server_->getSSCMapPtr()->block_size();
}

bool SSCVoxbloxCriteriaMap::isTraversable(const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation) {
    double collision_radius = planner_.getSystemConstraints().collision_radius;

    // the criteria to use ssc map is not met. Using measured map instead
    double distance = 0.0;
    if (esdf_server_->getEsdfMapPtr()->getDistanceAtPosition(position, &distance)) {
        // This means the voxel is observed
        return (distance > collision_radius);
    } else if (ssc_utilization_criteria_->criteriaVerify(*ssc_server_->getSSCMapPtr(), position)) {
        // The criteria to use ssc map is met.
        std::vector<Eigen::Vector3d> neighbouring_points;
        voxblox::utils::getSurroundingVoxelsSphere(position, c_voxel_size_, collision_radius, &neighbouring_points);

        for (auto point : neighbouring_points) {
            if (getVoxelState(point) == OccupancyMap::OCCUPIED) {
                return false;
            }
        }
        return true;
    }

    return false;
}

bool SSCVoxbloxCriteriaMap::isObserved(const Eigen::Vector3d& point) {
    bool observed = false;

    if (ssc_utilization_criteria_->criteriaVerify(*ssc_server_->getSSCMapPtr(), point)) {
        observed = ssc_server_->getSSCMapPtr()->isObserved(point);
    } else {
        observed = esdf_server_->getEsdfMapPtr()->isObserved(point);
    }
    return observed;
}

// get occupancy
unsigned char SSCVoxbloxCriteriaMap::getVoxelState(const Eigen::Vector3d& point) {
    
    if (ssc_utilization_criteria_->criteriaVerify(*ssc_server_->getSSCMapPtr(), point)) {
        return OccupancyMap::OCCUPIED;
    } else {
        double distance = 0.0;
        if (esdf_server_->getEsdfMapPtr()->getDistanceAtPosition(point, &distance)) {
            // This means the voxel is observed by ESDF Map
            if (distance < c_voxel_size_) {
                return OccupancyMap::OCCUPIED;
            } else {
                return OccupancyMap::FREE;
            }
        }
    }
    return OccupancyMap::UNKNOWN;
}


// criteria functions
bool ConfidenceCriteria::criteriaVerify(const voxblox::SSCMap & ssc_map, const Eigen::Vector3d& position) {
   // Get the block.
    auto block_ptr = ssc_map.getSSCLayer().getBlockPtrByCoordinates(position.cast<float>());
    if (block_ptr) {
        const auto& voxel = block_ptr->getVoxelByCoordinates(position.cast<float>());
        return voxel.probability_log > voxblox::logOddsFromProbability(confidence_threshold_);
    }
    return false;
}

}  // namespace map
}  // namespace active_3d_planning
