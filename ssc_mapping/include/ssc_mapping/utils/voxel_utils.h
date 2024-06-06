#ifndef SSC_VOXEL_UTILS_H_
#define SSC_VOXEL_UTILS_H_

#include <voxblox/core/block.h>
#include <voxblox/interpolator/interpolator.h>
#include <voxblox/utils/evaluation_utils.h>
#include <voxblox/utils/voxel_utils.h>

#include "ssc_mapping/core/voxel.h"
#include "ssc_mapping/visualization/visualization.h"

namespace voxblox {

// part1 - used during upsampling. just pick largest class
// labels among voxels. Use default class prediction.
// like scfusion uses 0.51 for new predictions irespective
// of probs predicted by the network.
template <>
inline SSCOccupancyVoxel
Interpolator<SSCOccupancyVoxel>::interpVoxel(const InterpVector &q_vector,
                                             const SSCOccupancyVoxel **voxels) {
  const int MAX_CLASSES = 12;
  SSCOccupancyVoxel voxel;

  uint32_t count_observed = 0;
  std::vector<uint16_t> preds(MAX_CLASSES, 0);
  for (int i = 0; i < q_vector.size(); ++i) {
    if (voxels[i]->observed) {
      count_observed++;
      preds[voxels[i]->label]++;
    }
  }

  // if at least of half of voxels are observed,
  // assign the maximum class among the voxels
  if (count_observed > q_vector.size() / 2) {
    auto max = std::max_element(preds.begin(), preds.end());
    auto idx = std::distance(preds.begin(), max);
    voxel.label = idx;
    voxel.label_weight = 1.0f;
    voxel.observed = true;
  }

  return voxel;
}

// part 2 merging upsampled temp layer into voxel of map layer.
// Note; updated to use log probs and label fusion like in
// scfusion
// namespace utils
template <>
inline void mergeVoxelAIntoVoxelB(const SSCOccupancyVoxel &voxel_A,
                                  SSCOccupancyVoxel *voxel_B) {
  voxel_B->label = voxel_A.label;
  voxel_B->label_weight = voxel_A.label_weight;
  voxel_B->observed = voxel_A.observed;
  voxel_B->probability_log = voxel_A.probability_log;
}

template <>
inline void Block<SSCOccupancyVoxel>::serializeToIntegers(
    std::vector<uint32_t> *data) const {
  CHECK_NOTNULL(data);
  constexpr size_t kNumDataPacketsPerVoxel = 4u;
  data->clear();
  data->reserve(num_voxels_ * kNumDataPacketsPerVoxel);
  for (size_t voxel_idx = 0u; voxel_idx < num_voxels_; ++voxel_idx) {
    const SSCOccupancyVoxel &voxel = voxels_[voxel_idx];

    const uint32_t *bytes_1_ptr =
        reinterpret_cast<const uint32_t *>(&voxel.probability_log);
    data->push_back(*bytes_1_ptr);
    data->push_back(static_cast<uint32_t>(voxel.observed));

    const uint32_t *bytes_3_ptr =
        reinterpret_cast<const uint32_t *>(&voxel.label);

    data->push_back(*bytes_3_ptr);

    const uint32_t *bytes_4_ptr =
        reinterpret_cast<const uint32_t *>(&voxel.label_weight);
    data->push_back(*bytes_4_ptr);
  }
  CHECK_EQ(num_voxels_ * kNumDataPacketsPerVoxel, data->size());
}

template <>
inline void Block<SSCOccupancyVoxel>::deserializeFromIntegers(
    const std::vector<uint32_t> &data) {
  constexpr size_t kNumDataPacketsPerVoxel = 4u;
  const size_t num_data_packets = data.size();
  CHECK_EQ(num_voxels_ * kNumDataPacketsPerVoxel, num_data_packets);
  for (size_t voxel_idx = 0u, data_idx = 0u;
       voxel_idx < num_voxels_ && data_idx < num_data_packets;
       ++voxel_idx, data_idx += kNumDataPacketsPerVoxel) {
    const uint32_t bytes_1 = data[data_idx];
    const uint32_t bytes_2 = data[data_idx + 1u];
    const uint32_t bytes_3 = data[data_idx + 2u];
    const uint32_t bytes_4 = data[data_idx + 3u];

    SSCOccupancyVoxel &voxel = voxels_[voxel_idx];

    memcpy(&(voxel.probability_log), &bytes_1, sizeof(bytes_1));
    voxel.observed = static_cast<bool>(bytes_2 & 0x000000FF);
    memcpy(&(voxel.label), &bytes_3, sizeof(bytes_3));
    memcpy(&(voxel.label_weight), &bytes_4, sizeof(bytes_4));
  }
}

namespace utils {
template <> inline bool isObservedVoxel(const SSCOccupancyVoxel &voxel) {
  return voxel.observed;
}

inline bool isOccupied(const voxblox::TsdfVoxel &voxel, float voxel_size) {
  constexpr float kMinWeight = 1e-3;
  if (voxel.weight > kMinWeight && voxel.distance <= voxel_size) {
    return true;
  }
  return false;
}

inline bool isOccupied(const voxblox::SSCOccupancyVoxel &voxel,
                       float /* voxel_size */) {
  return voxel.observed && voxel.probability_log > logOddsFromProbability(0.5f);
  // && voxel.label > 0;  // NOTE: this incldue classes, original version of
  // Mansoor.
}

// Note: should be changed to setUnObserved.
// avoiding to break changes.
inline void setUnOccupied(voxblox::SSCOccupancyVoxel *voxel) {
  voxel->observed = false;
}

inline void setUnOccupied(voxblox::TsdfVoxel *voxel) { voxel->weight = 0; }

// get all the voxels at the radius from point within voxel size
inline void getSurroundingVoxelsSphere(const Eigen::Vector3d &point,
                                       double voxel_size, double radius,
                                       std::vector<Eigen::Vector3d> *points) {
  for (float x = -radius; x <= radius; x += voxel_size) {
    for (float y = -radius; y <= radius; y += voxel_size) {
      for (float z = -radius; z <= radius; z += voxel_size) {
        Eigen::Vector3d offset(x, y, z);
        if (offset.norm() <= radius) {
          // NOTE: Could also just iterate over sphere shell instead of volume
          points->emplace_back(point + offset);
        }
      }
    }
  }
}

} // namespace utils

} // namespace voxblox

#endif // SSC_VOXEL_UTILS_H_