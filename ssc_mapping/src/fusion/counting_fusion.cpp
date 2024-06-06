#include "ssc_mapping/fusion/counting_fusion.h"

namespace ssc_fusion {

void CountingFusion::fuse(voxblox::SSCOccupancyVoxel *voxel,
                          uint predicted_label, float confidence,
                          float weight) {
  // We just abuse the voxels label and label_weight fields for occuppied and
  // free counts, respectively.
  voxel->observed = true;
  if (voxel->label < 0) {
    voxel->label = 0;
  }
  if (predicted_label == 0) {
    // This is free space.
    voxel->label_weight += 1.f;
  } else {
    voxel->label += 1;
  }
  voxel->probability_log = voxblox::logOddsFromProbability(
      voxel->label / (voxel->label + voxel->label_weight));
}

} // namespace ssc_fusion
