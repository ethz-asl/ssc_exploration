#ifndef SSC_BASE_FUSION_H_
#define SSC_BASE_FUSION_H_

#include "ssc_mapping/core/voxel.h"

namespace ssc_fusion {

namespace strategy {
const std::string naive = "naive";
const std::string occupancy_fusion = "occupancy_fusion";
const std::string log_odds = "log_odds";
const std::string counting = "counting";
const std::string sc_fusion = "sc_fusion";
} // namespace strategy

class BaseFusion {
public:
  struct Config {
    // default confidence of a semantic label
    float pred_conf = 0.75f;

    // max weight for semantic label. Each measurement adds default prediction
    // score. The max weight is clipped to this value.
    float max_weight = 50.0f;

    // Negative exponential weight decay standard deviatiaon for fusing far away
    // completions
    float decay_weight_std = 0.0f;

    // default probability value for occupied voxelss
    float prob_occupied = 0.675f;

    // default probability value for free voxels
    float prob_free = 0.45f;

    // fusion strategy to fuse new measurements into a voxel
    std::string fusion_strategy = strategy::log_odds;

    float min_prob = 0.12f;

    float max_prob = 0.97f;
  };

  virtual void fuse(voxblox::SSCOccupancyVoxel *voxel, uint predicted_label,
                    float confidence = 0.51f, float weight = 0.0f) = 0;
};
} // namespace ssc_fusion

#endif // SSC_BASE_FUSION_H_