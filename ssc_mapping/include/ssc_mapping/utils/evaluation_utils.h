#ifndef SSC_UTILS_H_
#define SSC_UTILS_H_

// ros/pcl includes
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>

// voxblox includes
#include <voxblox/core/block.h>
#include <voxblox/core/common.h>
#include <voxblox/core/layer.h>
#include <voxblox/core/tsdf_map.h>
#include <voxblox/core/voxel.h>
#include <voxblox/io/layer_io.h>

#include <ssc_mapping/utils/voxel_utils.h>
#include <ssc_mapping/visualization/visualization.h>

// Get a voxel index from a global 3D point
voxblox::GlobalIndex
get_voxel_index_from_point(const voxblox::Point &point,
                           voxblox::FloatingPoint voxel_size_inv) {
  return voxblox::getGridIndexFromPoint<voxblox::GlobalIndex>(point,
                                                              voxel_size_inv);
}

// returns the state of a voxel, 0-free 1-occupied 2-unknown
int get_voxel_state(const voxblox::GlobalIndex &index,
                    const voxblox::Layer<voxblox::TsdfVoxel> &layer) {
  voxblox::BlockIndex block_idx;
  voxblox::VoxelIndex voxel_idx;
  voxblox::getBlockAndVoxelIndexFromGlobalVoxelIndex(
      index, layer.voxels_per_side(), &block_idx, &voxel_idx);
  const auto block = layer.getBlockPtrByIndex(block_idx);
  if (block) {
    const voxblox::TsdfVoxel &voxel = block->getVoxelByVoxelIndex(voxel_idx);
    if (voxel.weight > 1e-6) {
      if (voxel.distance > layer.voxel_size() / 2) {
        return 0;
      } else {
        return 1;
      }
    }
  }
  return 2;
}

// start free space explroation from start point and
// mark all reachable free voxels.
// Returns a list of global indices of free voxels, and
// also a list of voxels that are immediate next voxels
// which should be occupied voxels as they mark the boundary
// of free space
void compute_free_occupied_space_frontier(
    const voxblox::Layer<voxblox::TsdfVoxel> &layer,
    const voxblox::Point &initial_point, voxblox::GlobalIndexVector *voxels,
    voxblox::GlobalIndexVector *obstacles) {

  auto t_start = std::chrono::high_resolution_clock::now();
  typedef voxblox::GlobalIndex GlobalIndex;

  GlobalIndex kNeighborOffsets[26] = {
      GlobalIndex(1, 0, 0),   GlobalIndex(1, 1, 0),   GlobalIndex(1, -1, 0),
      GlobalIndex(1, 0, 1),   GlobalIndex(1, 1, 1),   GlobalIndex(1, -1, 1),
      GlobalIndex(1, 0, -1),  GlobalIndex(1, 1, -1),  GlobalIndex(1, -1, -1),
      GlobalIndex(0, 1, 0),   GlobalIndex(0, -1, 0),  GlobalIndex(0, 0, 1),
      GlobalIndex(0, 1, 1),   GlobalIndex(0, -1, 1),  GlobalIndex(0, 0, -1),
      GlobalIndex(0, 1, -1),  GlobalIndex(0, -1, -1), GlobalIndex(-1, 0, 0),
      GlobalIndex(-1, 1, 0),  GlobalIndex(-1, -1, 0), GlobalIndex(-1, 0, 1),
      GlobalIndex(-1, 1, 1),  GlobalIndex(-1, -1, 1), GlobalIndex(-1, 0, -1),
      GlobalIndex(-1, 1, -1), GlobalIndex(-1, -1, -1)};

  // Cache submap data.
  voxblox::FloatingPoint voxel_size = layer.voxel_size();
  CHECK_GT(voxel_size, 0.f);
  voxblox::FloatingPoint voxel_size_inv = 1.f / voxel_size;

  // Setup search.
  voxblox::LongIndexSet closed_list;
  std::stack<GlobalIndex> open_stack;

  open_stack.push(get_voxel_index_from_point(initial_point, voxel_size_inv));

  // Search all frontiers.
  while (!open_stack.empty()) {
    // 'current', including the initial point, traverse observed free space.
    auto current = open_stack.top();
    open_stack.pop();

    // Check all neighbors for frontiers and free space.
    for (auto offset : kNeighborOffsets) {
      auto candidate = current + offset;
      if (closed_list.find(candidate) != closed_list.end()) {
        // Only consider voxels that were not yet checked.
        continue;
      }
      switch (get_voxel_state(candidate, layer)) {
      case 0:   // free
      case 2: { // unknown
        // Adjacent free space to continue the search.
        open_stack.push(candidate);
        closed_list.insert(candidate);

        break;
      }
      case 1:
      default:
        // We hit an obstacle.
        obstacles->emplace_back(candidate);
        break;
      }
    }
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  std::copy(closed_list.begin(), closed_list.end(),
            std::back_inserter(*voxels));
}

// Refine an observed layer by ignoring all voxels in observed layer that
// are not observed in ground truth layer.
template <typename VoxelTypeA, typename VoxelTypeB>
void refine_observed_layer(
    const voxblox::Layer<VoxelTypeA> &gt_layer,
    std::shared_ptr<voxblox::Layer<VoxelTypeB>> observed_layer) {

  CHECK(gt_layer.voxel_size() == observed_layer->voxel_size())
      << "Layers should have same voxel size.";
  CHECK(gt_layer.voxels_per_side() == observed_layer->voxels_per_side())
      << "Layers should have same block size.";

  size_t vps = gt_layer.voxels_per_side();
  size_t num_voxels_per_block = vps * vps * vps;

  voxblox::BlockIndexList blocks;
  observed_layer->getAllAllocatedBlocks(&blocks);
  for (const voxblox::BlockIndex &block_idx : blocks) {
    auto block = observed_layer->getBlockPtrByIndex(block_idx);

    for (size_t linear_index = 0; linear_index < num_voxels_per_block;
         ++linear_index) {
      // voxblox::Point coord =
      // block.computeCoordinatesFromLinearIndex(linear_index);
      auto voxel = &block->getVoxelByLinearIndex(linear_index);
      voxblox::VoxelIndex voxel_idx =
          block->computeVoxelIndexFromLinearIndex(linear_index);
      voxblox::GlobalIndex global_voxel_idx =
          voxblox::getGlobalVoxelIndexFromBlockAndVoxelIndex(
              block_idx, voxel_idx, gt_layer.voxels_per_side());

      // see if this voxel is observed in other layer
      auto gt_voxel = gt_layer.getVoxelPtrByGlobalIndex(global_voxel_idx);

      if (!gt_voxel) {
        // voxel not observed in gt_map so ignore voxels that are not observed
        // in gt map
        voxblox::utils::setUnOccupied(voxel);
      } else if (!voxblox::utils::isObservedVoxel(*gt_voxel)) {
        voxblox::utils::setUnOccupied(voxel);
      }
    }
  }
}

// find all the observed occupied and free voxels.
// Return the global indices array of these free and occupied voxels.
void get_free_and_occupied_voxels_from_layer(
    const voxblox::Layer<voxblox::TsdfVoxel> &layer,
    voxblox::LongIndexSet *occ_voxels, voxblox::LongIndexSet *free_voxels,
    float ssc_confidence_threshold = 0.f) {

  size_t vps = layer.voxels_per_side();
  size_t num_voxels_per_block = vps * vps * vps;
  voxblox::BlockIndexList blocks;
  layer.getAllAllocatedBlocks(&blocks);
  for (const voxblox::BlockIndex &block_idx : blocks) {
    const voxblox::Block<voxblox::TsdfVoxel> &block =
        layer.getBlockByIndex(block_idx);

    for (size_t linear_index = 0; linear_index < num_voxels_per_block;
         ++linear_index) {
      // voxblox::Point coord =
      // block.computeCoordinatesFromLinearIndex(linear_index);
      auto voxel = block.getVoxelByLinearIndex(linear_index);
      voxblox::VoxelIndex voxel_idx =
          block.computeVoxelIndexFromLinearIndex(linear_index);
      voxblox::GlobalIndex global_voxel_idx =
          voxblox::getGlobalVoxelIndexFromBlockAndVoxelIndex(
              block_idx, voxel_idx, layer.voxels_per_side());

      if (voxblox::utils::isObservedVoxel(voxel)) {
        if (voxblox::utils::isOccupied(voxel, layer.voxel_size())) {
          occ_voxels->insert(global_voxel_idx);
        } else {
          free_voxels->insert(global_voxel_idx);
        }
      }
    }
  }
}

void get_free_and_occupied_voxels_from_layer(
    const voxblox::Layer<voxblox::SSCOccupancyVoxel> &layer,
    voxblox::LongIndexSet *occ_voxels, voxblox::LongIndexSet *free_voxels,
    float ssc_confidence_threshold = 0.f) {

  size_t vps = layer.voxels_per_side();
  size_t num_voxels_per_block = vps * vps * vps;
  const float occupancy_threshold =
      voxblox::logOddsFromProbability(0.5f + ssc_confidence_threshold);
  const float free_threshold =
      voxblox::logOddsFromProbability(0.5f - ssc_confidence_threshold);

  voxblox::BlockIndexList blocks;
  layer.getAllAllocatedBlocks(&blocks);
  for (const voxblox::BlockIndex &block_idx : blocks) {
    const voxblox::Block<voxblox::SSCOccupancyVoxel> &block =
        layer.getBlockByIndex(block_idx);

    for (size_t linear_index = 0; linear_index < num_voxels_per_block;
         ++linear_index) {
      // voxblox::Point coord =
      // block.computeCoordinatesFromLinearIndex(linear_index);
      auto voxel = block.getVoxelByLinearIndex(linear_index);
      voxblox::VoxelIndex voxel_idx =
          block.computeVoxelIndexFromLinearIndex(linear_index);
      voxblox::GlobalIndex global_voxel_idx =
          voxblox::getGlobalVoxelIndexFromBlockAndVoxelIndex(
              block_idx, voxel_idx, layer.voxels_per_side());

      if (voxblox::utils::isObservedVoxel(voxel)) {
        if (voxel.probability_log > occupancy_threshold) {
          occ_voxels->insert(global_voxel_idx);
        } else if (voxel.probability_log <= free_threshold) {
          free_voxels->insert(global_voxel_idx);
        }
      }
    }
  }
}

// creates a refined ground truth layer filled with free space observed
// voxels and a single voxel layer occupancy around it.
void refine_gt_layer(const voxblox::Layer<voxblox::TsdfVoxel> &gt_layer,
                     voxblox::Layer<voxblox::TsdfVoxel>::Ptr refined_layer,
                     const voxblox::Point start = voxblox::Point(0.0, 0.0,
                                                                 0.0)) {
  voxblox::GlobalIndexVector gt_occ_voxels, gt_free_voxels;
  compute_free_occupied_space_frontier(gt_layer, start, &gt_free_voxels,
                                       &gt_occ_voxels);

  // add these voxels to refined layer
  for (auto global_voxel_idx : gt_occ_voxels) {

    voxblox::TsdfVoxel *refined_occ_voxel =
        refined_layer->getVoxelPtrByGlobalIndex(global_voxel_idx);
    const voxblox::TsdfVoxel *gt_voxel =
        gt_layer.getVoxelPtrByGlobalIndex(global_voxel_idx);

    // check if the block containing the voxel exists.
    if (refined_occ_voxel == nullptr) {
      auto block_idx = voxblox::getBlockIndexFromGlobalVoxelIndex(
          global_voxel_idx, refined_layer->voxels_per_side_inv());
      auto block = refined_layer->allocateBlockPtrByIndex(block_idx);
      const voxblox::VoxelIndex local_voxel_idx =
          voxblox::getLocalFromGlobalVoxelIndex(
              global_voxel_idx, refined_layer->voxels_per_side());
      refined_occ_voxel = &block->getVoxelByVoxelIndex(local_voxel_idx);
    }

    // occ_voxel->weight = 1.0;
    // occ_voxel->distance = 0.0f;
    *refined_occ_voxel = *gt_voxel;
  }

  for (auto global_voxel_idx : gt_free_voxels) {
    voxblox::TsdfVoxel *free_voxel =
        refined_layer->getVoxelPtrByGlobalIndex(global_voxel_idx);

    // check if the block containing the voxel exists.
    if (free_voxel == nullptr) {
      // ssc_map_->getSSCLayerPtr()->a
      auto block_idx = voxblox::getBlockIndexFromGlobalVoxelIndex(
          global_voxel_idx, refined_layer->voxels_per_side_inv());
      auto block = refined_layer->allocateBlockPtrByIndex(block_idx);
      const voxblox::VoxelIndex local_voxel_idx =
          voxblox::getLocalFromGlobalVoxelIndex(
              global_voxel_idx, refined_layer->voxels_per_side());
      free_voxel = &block->getVoxelByVoxelIndex(local_voxel_idx);
    }
    free_voxel->weight = 1.0f;
    free_voxel->distance = 100.0f; // refined_layer->voxel_size();
  }
}

// fills all reachable free space wih obeserved voxels
void fill_gt_layer(voxblox::Layer<voxblox::TsdfVoxel>::Ptr gt_layer,
                   const voxblox::Point start = voxblox::Point(0.0, 0.0, 0.0)) {
  voxblox::GlobalIndexVector gt_occ_voxels, gt_free_voxels;
  compute_free_occupied_space_frontier(*gt_layer, start, &gt_free_voxels,
                                       &gt_occ_voxels);

  for (auto global_voxel_idx : gt_free_voxels) {
    voxblox::TsdfVoxel *free_voxel =
        gt_layer->getVoxelPtrByGlobalIndex(global_voxel_idx);

    // check if the block containing the voxel exists.
    if (free_voxel == nullptr) {
      // ssc_map_->getSSCLayerPtr()->a
      auto block_idx = voxblox::getBlockIndexFromGlobalVoxelIndex(
          global_voxel_idx, gt_layer->voxels_per_side_inv());
      auto block = gt_layer->allocateBlockPtrByIndex(block_idx);
      const voxblox::VoxelIndex local_voxel_idx =
          voxblox::getLocalFromGlobalVoxelIndex(global_voxel_idx,
                                                gt_layer->voxels_per_side());
      free_voxel = &block->getVoxelByVoxelIndex(local_voxel_idx);
    }
    free_voxel->weight = 1.0f;
    free_voxel->distance = 100.0f; // refined_layer->voxel_size();
  }
}

// finds which occupied voxels in layer are also occupied in other layer. The
// intersecting occupied voxels in otherLayer are added to intersection and the
// unobserved and "OBSERVED" voxels > voxel_size are added to difference.
template <typename VoxelTypeA, typename VoxelTypeB>
void calculate_Intersection_difference(
    const voxblox::Layer<VoxelTypeA> &layer,
    const voxblox::Layer<VoxelTypeB> &otherLayer,
    voxblox::GlobalIndexVector *intersection,
    voxblox::GlobalIndexVector *difference) {
  size_t vps = layer.voxels_per_side();
  size_t num_voxels_per_block = vps * vps * vps;

  voxblox::BlockIndexList blocks;
  layer.getAllAllocatedBlocks(&blocks);
  for (const voxblox::BlockIndex &index : blocks) {
    // Iterate over all voxels in said blocks.
    const voxblox::Block<voxblox::TsdfVoxel> &block =
        layer.getBlockByIndex(index);

    for (size_t linear_index = 0; linear_index < num_voxels_per_block;
         ++linear_index) {
      // voxblox::Point coord =
      // block.computeCoordinatesFromLinearIndex(linear_index);
      auto gt_voxel = block.getVoxelByLinearIndex(linear_index);

      if (voxblox::utils::isOccupied(gt_voxel, layer.voxel_size())) {
        // voxel is observed in first layer. check if it exists in other layer

        // get global voxel index
        voxblox::VoxelIndex voxel_idx =
            block.computeVoxelIndexFromLinearIndex(linear_index);
        voxblox::BlockIndex block_idx = index;
        voxblox::GlobalIndex global_voxel_idx =
            voxblox::getGlobalVoxelIndexFromBlockAndVoxelIndex(
                block_idx, voxel_idx, layer.voxels_per_side());

        // see if this voxel is observed in other layer
        auto observed_voxel =
            otherLayer.getVoxelPtrByGlobalIndex(global_voxel_idx);

        if (observed_voxel != nullptr &&
            voxblox::utils::isOccupied(*observed_voxel,
                                       otherLayer.voxel_size())) {
          intersection->emplace_back(global_voxel_idx);
        } else {
          difference->emplace_back(global_voxel_idx);
        }
      }
    }
  }
}

// remove voxel indices that correspont to voxels that
// are inside walls or beneath surface. Such voxels have
// negative sign distance
template <typename VoxelType>
void prune_inside_voxels(const voxblox::Layer<VoxelType> &layer,
                         voxblox::GlobalIndexVector *voxels) {
  voxels->erase(std::remove_if(voxels->begin(), voxels->end(),
                               [&](auto voxel_idx) -> bool {
                                 auto voxel =
                                     layer.getVoxelPtrByGlobalIndex(voxel_idx);
                                 if (voxel != nullptr && voxel->weight > 1e-6 &&
                                     voxel->distance < 0)
                                   return true;
                                 return false;
                               }),
                voxels->end());
}

// compute the world bounds from ground truth map and then  discard all
// voxel indices that are outside bounds
template <typename VoxelType>
void prune_outlier_voxels(const voxblox::Layer<VoxelType> &layer,
                          voxblox::GlobalIndexVector *voxels) {
  voxblox::Point min_coords, max_coords;
  min_coords.setZero();
  max_coords.setZero();

  size_t vps = layer.voxels_per_side();
  size_t num_voxels_per_block = vps * vps * vps;

  voxblox::BlockIndexList blocks;
  layer.getAllAllocatedBlocks(&blocks);
  for (const voxblox::BlockIndex &index : blocks) {
    const voxblox::Block<voxblox::TsdfVoxel> &block =
        layer.getBlockByIndex(index);

    for (size_t linear_index = 0; linear_index < num_voxels_per_block;
         ++linear_index) {
      // voxblox::Point coord =
      // block.computeCoordinatesFromLinearIndex(linear_index);
      auto voxel = block.getVoxelByLinearIndex(linear_index);

      if (voxel.weight > 1e-6) {
        voxblox::VoxelIndex voxel_idx =
            block.computeVoxelIndexFromLinearIndex(linear_index);
        voxblox::BlockIndex block_idx = index;
        voxblox::GlobalIndex global_voxel_idx =
            voxblox::getGlobalVoxelIndexFromBlockAndVoxelIndex(
                block_idx, voxel_idx, layer.voxels_per_side());
        auto voxel_coords = voxblox::getCenterPointFromGridIndex(
            global_voxel_idx, layer.voxel_size());

        min_coords = min_coords.cwiseMin(voxel_coords);
        max_coords = max_coords.cwiseMax(voxel_coords);
      }
    }

    // auto block_origin = block.origin();
    // min_coords = min_coords.cwiseMin(block_origin);
    // max_coords = max_coords.cwiseMax(block_origin);
  }

  voxels->erase(
      std::remove_if(voxels->begin(), voxels->end(),
                     [&](auto voxel_idx) -> bool {
                       auto voxel = layer.getVoxelPtrByGlobalIndex(voxel_idx);
                       //  if (min_coords.cwiseMin(voxel_idx) != min_coords ||
                       //      max_coords.cwiseMax(voxel_idx) != max_coords)
                       //      return true;
                       auto voxel_coords = getOriginPointFromGridIndex(
                           voxel_idx, layer.voxel_size());
                       // voxblox::Point new_min_coords =
                       // min_coords.cwiseMin(voxel_coords);
                       if (min_coords.cwiseMin(voxel_coords) != min_coords ||
                           max_coords.cwiseMax(voxel_coords) != max_coords)
                         return true;
                       return false;
                     }),
      voxels->end());
}

// Creates a list of obeserved and unobserved voxels (in the input layer) from a
// list of voxels. The results are filled in out_observed_voxels and
// out_UnObserved_voxels.
void split_observed_unobserved_voxels(
    const voxblox::Layer<voxblox::TsdfVoxel> &layer,
    const voxblox::LongIndexSet &in_voxels,
    voxblox::LongIndexSet *out_observed_voxels,
    voxblox::LongIndexSet *out_un_observed_voxels,
    float ssc_confidence_threshold=0.f) {

  const float occupancy_threshold =
      voxblox::logOddsFromProbability(0.5f + ssc_confidence_threshold);
  const float free_threshold =
      voxblox::logOddsFromProbability(0.5f - ssc_confidence_threshold);

  for (auto voxel_idx : in_voxels) {
    auto voxel = layer.getVoxelPtrByGlobalIndex(voxel_idx);

    if (voxel == nullptr) {
      // voxel does not exist, so its un observed
      out_un_observed_voxels->insert(voxel_idx);
      continue;
    }
    bool is_observed = voxblox::utils::isObservedVoxel(*voxel);
    if (is_observed) {
      // voxel exists and is observed
      out_observed_voxels->insert(voxel_idx);
    } else {
      // voxel exists, but is not observed
      out_un_observed_voxels->insert(voxel_idx);
    }
  }
}

void split_observed_unobserved_voxels(
    const voxblox::Layer<voxblox::SSCOccupancyVoxel> &layer,
    const voxblox::LongIndexSet &in_voxels,
    voxblox::LongIndexSet *out_observed_voxels,
    voxblox::LongIndexSet *out_un_observed_voxels,
    float ssc_confidence_threshold=0.f) {

  const float occupancy_threshold =
      voxblox::logOddsFromProbability(0.5f + ssc_confidence_threshold);
  const float free_threshold =
      voxblox::logOddsFromProbability(0.5f - ssc_confidence_threshold);

  for (auto voxel_idx : in_voxels) {
    auto voxel = layer.getVoxelPtrByGlobalIndex(voxel_idx);

    if (voxel == nullptr) {
      // voxel does not exist, so its un observed
      out_un_observed_voxels->insert(voxel_idx);
      continue;
    }
    bool is_observed = voxblox::utils::isObservedVoxel(*voxel);
    if (voxel->probability_log <= occupancy_threshold &&
        voxel->probability_log > free_threshold) {
      is_observed = false;
    }

    if (is_observed) {
      // voxel exists and is observed
      out_observed_voxels->insert(voxel_idx);
    } else {
      // voxel exists, but is not observed
      out_un_observed_voxels->insert(voxel_idx);
    }
  }
}

using IndexSet = voxblox::LongIndexSet;
typedef std::tuple<IndexSet, IndexSet, IndexSet, IndexSet, IndexSet, IndexSet,
                   IndexSet, IndexSet>
    VoxelEvalData;

template <typename VoxelTypeA, typename VoxelTypeB>
VoxelEvalData get_voxel_data_from_layer(
    std::shared_ptr<voxblox::Layer<VoxelTypeA>> ground_truth_layer,
    std::shared_ptr<voxblox::Layer<VoxelTypeB>> observed_layer,
    bool refine_ob_layer, float ssc_confidence_threshold = 0.f) {
  CHECK(ground_truth_layer->voxel_size() == observed_layer->voxel_size())
      << "Error! Observed Layer and groundtruth layers should have same "
         "voxel "
         "size!";

  if (refine_ob_layer) {
    LOG(INFO) << "Updating observed layer by discarding voxels not observed in "
                 "ground truth.";
    refine_observed_layer(*ground_truth_layer, observed_layer);
  }

  IndexSet gt_occ_voxels, gt_free_voxels;
  get_free_and_occupied_voxels_from_layer(*ground_truth_layer, &gt_occ_voxels,
                                          &gt_free_voxels);

  IndexSet map_obs_occ_voxels, map_obs_free_voxels;
  get_free_and_occupied_voxels_from_layer(*observed_layer, &map_obs_occ_voxels,
                                          &map_obs_free_voxels,
                                          ssc_confidence_threshold);

  IndexSet gt_obs_occ, gt_unobs_occ;
  split_observed_unobserved_voxels(*observed_layer, gt_occ_voxels, &gt_obs_occ,
                                   &gt_unobs_occ, ssc_confidence_threshold);

  IndexSet gt_obs_free, gt_unobs_free;
  split_observed_unobserved_voxels(*observed_layer, gt_free_voxels,
                                   &gt_obs_free, &gt_unobs_free,
                                   ssc_confidence_threshold);

  return {gt_occ_voxels,       gt_free_voxels, map_obs_occ_voxels,
          map_obs_free_voxels, gt_obs_occ,     gt_obs_free,
          gt_unobs_occ,        gt_unobs_free};
}
#endif // SSC_UTILS_H_