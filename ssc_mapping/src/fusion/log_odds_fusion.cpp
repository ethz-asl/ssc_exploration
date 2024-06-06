#include "ssc_mapping/fusion/log_odds_fusion.h"

namespace ssc_fusion {

LogOddsFusion::LogOddsFusion(float pred_conf, float max_weight, float min_prob, float max_prob)
    : pred_conf_(pred_conf),
      max_weight_(max_weight),
      min_log_prob_(voxblox::logOddsFromProbability(min_prob)),
      max_log_prob_(voxblox::logOddsFromProbability(max_prob)){};

LogOddsFusion::LogOddsFusion(const BaseFusion::Config& config)
    : LogOddsFusion(config.pred_conf, config.max_weight, config.min_prob, config.max_prob) {}

void LogOddsFusion::fuse(voxblox::SSCOccupancyVoxel* voxel, uint predicted_label, float confidence, float weight) {
    voxel->observed = true;
    //=================================
    // Fuse Semantics - Like SCFusion
    //=================================
    if (predicted_label > 0) {
        if (predicted_label == voxel->label) {
            voxel->label_weight = std::min(voxel->label_weight + pred_conf_, max_weight_);
        } else if (voxel->label_weight < pred_conf_) {
            voxel->label_weight = pred_conf_ - voxel->label_weight;
            voxel->label = predicted_label;
        } else {
            voxel->label_weight = voxel->label_weight - pred_conf_;
        }
    }

    //==================================================================
    // Fuse Occupancy - Network occupancy confidence as log probability
    //==================================================================
    // if voxel is predicted as occupied
    float log_odds_update = voxblox::logOddsFromProbability(confidence);

    voxel->probability_log =
        std::min(std::max(voxel->probability_log + log_odds_update, min_log_prob_), max_log_prob_);
}

}  // namespace ssc_fusion
