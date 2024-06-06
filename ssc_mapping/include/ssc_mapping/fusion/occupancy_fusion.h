#ifndef SSC_OCCUPANCY_FUSION_H_
#define SSC_OCCUPANCY_FUSION_H_

#include <voxblox/core/common.h>

#include "ssc_mapping/fusion/base_fusion.h"

namespace ssc_fusion {
/**
 * Log odds based occupancy fusion using SCFusion like fixed probabilities
 * for free and occupied space. Semantics are fused naively, also
 * like SCFusion.
 */
class OccupancyFusion : public BaseFusion {
   public:
    OccupancyFusion(const BaseFusion::Config& config);

    OccupancyFusion(float pred_conf, float max_weight, float prob_occupied, float prob_free, float prob_min, float prob_max);

    virtual void fuse(voxblox::SSCOccupancyVoxel* voxel, uint predicted_label, float confidence = 0.f, float weight=0.0f) override;

   private:
    float min_log_prob_; // minumim log probability
    float max_log_prob_; // maximum threshold of log probability
    float pred_conf_; // weight for a single semantic prediction - ref: scfusion - uses confidence as weight for a semantic weight
    float max_weight_; // max aggregated label semantic weight
    float prob_occupied_; // constant probability to fuse occupied voxels
    float prob_free_;// constant probability to fuse free voxels

    // Detailed Calibration.
    std::vector<float> calibration_weights_;
};
}  // namespace ssc_fusion

#endif  // SSC_OCCUPANCY_FUSION_H_