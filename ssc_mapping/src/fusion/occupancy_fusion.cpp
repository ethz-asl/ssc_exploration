#include "ssc_mapping/fusion/occupancy_fusion.h"

namespace ssc_fusion {

OccupancyFusion::OccupancyFusion(float pred_conf, float max_weight,
                                 float prob_occupied, float prob_free,
                                 float min_prob, float max_prob)
    : pred_conf_(pred_conf), max_weight_(max_weight),
      prob_occupied_(prob_occupied), prob_free_(prob_free),
      min_log_prob_(voxblox::logOddsFromProbability(min_prob)),
      max_log_prob_(voxblox::logOddsFromProbability(max_prob)) {
  // TEST
  calibration_weights_ = {0.00861387f, 0.22477943f, 0.40927035f, 0.36088128f,
                          0.f,         0.31986936f, 0.45573225f, 0.55741808f,
                          0.31225102f, 0.f,         0.30001431f, 0.33003762f};
  for (float &w : calibration_weights_) {
    w = 0.5f + w * 0.5f;
  }
};

OccupancyFusion::OccupancyFusion(const BaseFusion::Config &config)
    : OccupancyFusion(config.pred_conf, config.max_weight, config.prob_occupied,
                      config.prob_free, config.min_prob, config.max_prob) {}

void OccupancyFusion::fuse(voxblox::SSCOccupancyVoxel *voxel,
                           uint predicted_label, float confidence,
                           float weight) {
  voxel->observed = true;
  // SCFusion for semantics
  if (predicted_label > 0) {
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

  // if voxel is predicted as occupied
  float prob_new;
  if (predicted_label > 0) {
    // occupied voxel
    constexpr bool detailed_calibration = false;
    if (detailed_calibration) {
      prob_new =
          ((calibration_weights_[predicted_label] - 0.5f) * weight) + 0.5f;
    } else {
      prob_new = ((prob_occupied_ - 0.5f) * weight) + 0.5f;
    }
  } else {
    // free voxel
    prob_new = ((prob_free_ - 0.5f) * weight) + 0.5f;
  }

  voxel->probability_log =
      std::min(std::max(voxel->probability_log +
                            voxblox::logOddsFromProbability(prob_new),
                        min_log_prob_),
               max_log_prob_);
}

} // namespace ssc_fusion
