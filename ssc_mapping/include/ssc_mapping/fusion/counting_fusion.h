#ifndef SSC_COUNTING_FUSION_H_
#define SSC_COUNTING_FUSION_H_

#include <voxblox/core/common.h>

#include "ssc_mapping/fusion/base_fusion.h"

namespace ssc_fusion {
/**
 * Counting Fusion
 */
class CountingFusion : public BaseFusion {
public:
  CountingFusion(const BaseFusion::Config &config) {}

  void fuse(voxblox::SSCOccupancyVoxel *voxel, uint predicted_label,
            float confidence = 0.f, float weight = 0.0f) override;
};
} // namespace ssc_fusion

#endif // SSC_COUNTING_FUSION_H_