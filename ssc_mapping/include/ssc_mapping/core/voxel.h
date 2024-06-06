
#ifndef SSC_VOXEL_H_
#define SSC_VOXEL_H_
#include <string>
#include <voxblox/core/voxel.h>

namespace voxblox {

struct SSCOccupancyVoxel {
    float probability_log = 0.0f;
    bool observed = false;
    int label = -1;
    float label_weight = 0.0f;
};

namespace voxel_types {
const std::string kSSCOccupancy = "ssc";
}  // namespace voxel_types

template <>
inline std::string getVoxelType<SSCOccupancyVoxel>() {
  return voxel_types::kSSCOccupancy;
}

}  // namespace voxblox
#endif //SSC_VOXEL_H_