#ifndef SSC_LOG_ODDS_FUSION_H_
#define SSC_LOG_ODDS_FUSION_H_

#include <voxblox/core/common.h>

#include "ssc_mapping/fusion/base_fusion.h"

namespace ssc_fusion {
/**
 * Log odds based occupancy fusion using Network Confidence as aprobabilitis and
 *  SCFusion like semantics are fused naively, also
 * like SCFusion.
 */
class LogOddsFusion : public BaseFusion {
public:
  LogOddsFusion(const BaseFusion::Config &config);

  LogOddsFusion(float pred_conf, float max_weight, float prob_min,
                float prob_max);

  void fuse(voxblox::SSCOccupancyVoxel *voxel, uint predicted_label,
            float confidence = 0.f, float weight = 0.0f) override;

private:
  float min_log_prob_;  // minumim log probability
  float max_log_prob_; // maximum threshold of log probability
  float pred_conf_; // weight for a single semantic prediction - ref: scfusion -
                    // uses confidence as weight for a semantic weight
  float max_weight_; // max aggregated label semantic weight
};
} // namespace ssc_fusion

#endif // SSC_LOG_ODDS_FUSION_H_