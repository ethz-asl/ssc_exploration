#include "ssc_mapping/fusion/naive_fusion.h"

namespace ssc_fusion {

void NaiveFusion::fuse(voxblox::SSCOccupancyVoxel *voxel, uint predicted_label,
                       float confidence, float weight) {
  // Overwrite everything with the latest prediction.
  voxel->observed = true;
  voxel->label = predicted_label;
  voxel->probability_log = voxblox::logOddsFromProbability(confidence);
  voxel->label_weight = weight;
}

} // namespace ssc_fusion
