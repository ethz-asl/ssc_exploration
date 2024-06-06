#ifndef SSC_SC_FUSION_H_
#define SSC_SC_FUSION_H_

#include <voxblox/core/common.h>

#include "ssc_mapping/fusion/base_fusion.h"

namespace ssc_fusion {
/**
 * Log odds based occupancy fusion using SCFusion like fixed probabilities
 * foroccupied space. Semantics are fused like SCFusion.
 */
class SCFusion : public BaseFusion {
public:
  SCFusion(const BaseFusion::Config &config) {}

  virtual void fuse(voxblox::SSCOccupancyVoxel *voxel, uint predicted_label,
                    float confidence = 0.f, float weight = 0.0f) override;

private:
  // NOTE: currently use the fixed values of SCFusion, i.e. not setting them by
  // param/config.
  float min_log_prob_ =
      voxblox::logOddsFromProbability(0.12); // minumim log probability
  float max_log_prob_ = voxblox::logOddsFromProbability(
      0.97); // maximum threshold of log probability
  float pred_conf_ =
      0.75; // weight for a single semantic prediction - ref: scfusion - uses
            // confidence as weight for a semantic weight
  float max_weight_ = 30.f;     // max aggregated label semantic weight
  float prob_occupied_ = 0.51f; // constant probability to fuse occupied voxels
};
} // namespace ssc_fusion

#endif // SSC_SC_FUSION_H_