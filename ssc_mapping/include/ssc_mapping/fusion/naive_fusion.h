#ifndef SSC_NAIVE_FUSION_H_
#define SSC_NAIVE_FUSION_H_

#include "ssc_mapping/fusion/base_fusion.h"
#include <voxblox/core/common.h>

namespace ssc_fusion {
class NaiveFusion : public BaseFusion {
public:
  virtual void fuse(voxblox::SSCOccupancyVoxel *voxel, uint predicted_label,
                    float confidence = 0.9f, float weight = 0.0f) override;

};
} // namespace ssc_fusion

#endif // SSC_NAIVE_FUSION_H_