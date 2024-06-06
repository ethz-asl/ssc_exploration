#include "ssc_mapping/fusion/sc_fusion.h"
#include <iostream>
namespace ssc_fusion {

// SCFusion::SCFusion(const BaseFusion::Config &config)
//     : SCFusion(config.pred_conf, config.max_weight, config.prob_occupied,
//                config.prob_free, config.min_prob, config.max_prob) {}

void SCFusion::fuse(voxblox::SSCOccupancyVoxel *voxel, uint predicted_label,
                    float confidence, float weight) {

  // First, the prediction is discarded if classified as empty
  if (predicted_label == 0) {
    return;
  }

  // or if its corresponding voxel in the global map is in the empty state
  if (voxel->observed && voxel->label == 0) {
    return;
  }

  // Second, the predicted semantic label is instead fused in the global map if
  // the corresponding voxel is in either the unknown or occupied state.
  if (!voxel->observed || voxel->label > 0) {
    if (predicted_label == voxel->label) {
      voxel->label_weight =
          std::min(voxel->label_weight + pred_conf_, max_weight_);
    } else if (voxel->label_weight < pred_conf_) {
      voxel->label_weight = pred_conf_ - voxel->label_weight;
      voxel->label = predicted_label;
    } else {
      voxel->label_weight = voxel->label_weight - pred_conf_;
    }
  }

  // a voxel predicted as occupied by the network is fused only if the
  // corresponding voxel in the global map is in the unknown state
  if (!voxel->observed) {
    voxel->probability_log = std::min(
        std::max(voxel->probability_log + voxblox::logOddsFromProbability(0.51),
                 min_log_prob_),
        max_log_prob_); // voxblox::logOddsFromProbability(prob_occupied_);
    voxel->observed = true;
  }

  // question:
  // how to fuse free space or consective occupancy as we dont have a global map
  // here
}

} // namespace ssc_fusion
