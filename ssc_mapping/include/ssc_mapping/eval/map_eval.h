
#ifndef SSC_MAPPING_EVAL_H_
#define SSC_MAPPING_EVAL_H_

// general includes
#include <assert.h>
#include <glog/logging.h>

#include <fstream>
#include <iostream>

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

#include "ssc_mapping/utils/evaluation_utils.h"
#include "ssc_mapping/visualization/visualization.h"

namespace ssc_mapping {
namespace evaluation {

struct QualityMetrics {
  double precision_occ;
  double precision_free;
  double precision_overall;
  double recall_occ;
  double recall_free;
  double IoU_occ;
  double IoU_free;

  void print() const {
    printf("------ Quality Metrics Evaluation --------\n");
    printf("IoU_occ: %0.2lf \n", IoU_occ);
    printf("IoU_free: %0.2lf \n", IoU_free);
    printf("Precision_occ: %0.2lf \n", precision_occ);
    printf("Precision_free: %0.2lf \n", precision_free);
    printf("Precision_overall: %0.2lf \n", precision_overall);
    printf("Recall_occ: %0.2lf \n", recall_occ);
    printf("Recall_free: %0.2lf \n", recall_free);
    printf("-----------------------------------------\n");
  }
};

struct CoverageMetrics {
  double explored_occ;
  double explored_free;
  double explored_overall;
  double coverage_occ;
  double coverage_free;
  double coverage_overall;

  void print() const {
    printf("------ Coverage Metrics Evaluation --------\n");
    printf("Explored_occ: %0.2lf \n", explored_occ);
    printf("Explored_free: %0.2lf \n", explored_free);
    printf("Explored_overall: %0.2lf \n", explored_overall);

    printf("Coverage_occ: %0.2lf \n", coverage_occ);
    printf("Coverage_free: %0.2lf \n", coverage_free);
    printf("Coverage_overall: %0.2lf \n", coverage_overall);
    printf("------------------------------------------\n");
  }
};

// return the ration of sizes
float operator/(const voxblox::LongIndexSet &i1,
                const voxblox::LongIndexSet &i2) {
  return i1.size() / float(i2.size());
}

// finds intersection of two unordered sets (hashmaps with O(1) search)
voxblox::LongIndexSet
set_intersection(const voxblox::LongIndexSet &index_set_a,
                 const voxblox::LongIndexSet &index_set_b) {
  voxblox::LongIndexSet intersect;
  for (auto idx : index_set_a) {
    if (index_set_b.find(idx) != index_set_b.end()) {
      intersect.insert(idx);
    }
  }
  return intersect;
}

// finds difference of two unordered set. Indexes that are present in set a but
// absent in set b
voxblox::LongIndexSet set_difference(const voxblox::LongIndexSet &index_set_a,
                                     const voxblox::LongIndexSet &index_set_b) {
  voxblox::LongIndexSet difference;
  for (auto idx : index_set_a) {
    if (index_set_b.find(idx) == index_set_b.end()) {
      difference.insert(idx);
    }
  }
  return difference;
}

// return union of two unordered hash maps.
voxblox::LongIndexSet set_union(const voxblox::LongIndexSet &index_set_a,
                                const voxblox::LongIndexSet &index_set_b) {
  voxblox::LongIndexSet union_elements = index_set_a;
  for (auto idx : index_set_b) {
    union_elements.insert(idx);
  }
  return union_elements;
}

QualityMetrics
calculate_quality_metrics(const voxblox::LongIndexSet &gt_occ_all,
                          const voxblox::LongIndexSet &gt_obs_occ,
                          const voxblox::LongIndexSet &gt_free_all,
                          const voxblox::LongIndexSet &gt_obs_free,
                          const voxblox::LongIndexSet &map_obs_occ,
                          const voxblox::LongIndexSet &map_obs_free) {
  // calculate counts
  auto map_obs_occ_correct = set_intersection(map_obs_occ, gt_occ_all);
  auto map_obs_free_correct = set_intersection(map_obs_free, gt_free_all);
  auto map_obs_correct = set_union(map_obs_occ_correct, map_obs_free_correct);

  auto gt_obs_occ_correct = set_intersection(map_obs_occ, gt_obs_occ);
  auto gt_obs_free_correct = set_intersection(map_obs_free, gt_obs_free);

  // calculate metrics
  QualityMetrics metrics;
  metrics.precision_occ = map_obs_occ_correct / map_obs_occ;
  metrics.precision_free = map_obs_free_correct / map_obs_free;
  metrics.precision_overall =
      map_obs_correct / set_union(map_obs_occ, map_obs_free);

  metrics.recall_occ = gt_obs_occ_correct / gt_obs_occ;
  metrics.recall_free = gt_obs_free_correct / gt_obs_free;

  metrics.IoU_occ = map_obs_occ_correct / set_union(map_obs_occ, gt_obs_occ);
  metrics.IoU_free =
      map_obs_free_correct / set_union(map_obs_free, gt_obs_free);

  return metrics;
}

CoverageMetrics
calculate_coverage_metrics(const voxblox::LongIndexSet &gt_occ_all,
                           const voxblox::LongIndexSet &gt_obs_occ,
                           const voxblox::LongIndexSet &gt_free_all,
                           const voxblox::LongIndexSet &gt_obs_free,
                           const voxblox::LongIndexSet &map_obs_occ,
                           const voxblox::LongIndexSet &map_obs_free) {
  auto gt_obs = set_union(gt_obs_occ, gt_obs_free);
  auto gt_all = set_union(gt_occ_all, gt_free_all);
  auto gt_obs_occ_correct = set_intersection(gt_obs_occ, map_obs_occ);
  auto gt_obs_free_correct = set_intersection(gt_obs_free, map_obs_free);
  auto gt_obs_correct = set_union(gt_obs_occ_correct, gt_obs_free_correct);

  // calculate metrics
  CoverageMetrics metrics;
  metrics.explored_occ = gt_obs_occ / gt_occ_all;
  metrics.explored_free = gt_obs_free / gt_free_all;
  metrics.explored_overall = gt_obs / gt_all;

  metrics.coverage_occ = gt_obs_occ_correct / gt_occ_all;
  metrics.coverage_free = gt_obs_free_correct / gt_free_all;
  metrics.coverage_overall = gt_obs_correct / gt_all;

  return metrics;
}

} // namespace evaluation

namespace tests {

void fill_dummy_data(
    std::shared_ptr<voxblox::Layer<voxblox::TsdfVoxel>> ground_truth_layer,
    std::shared_ptr<voxblox::Layer<voxblox::TsdfVoxel>> observed_layer) {
  // add a block at origin
  voxblox::Point point_in_0_0_0(0, 0, 0);
  ground_truth_layer->allocateNewBlockByCoordinates(point_in_0_0_0);
  observed_layer->allocateNewBlockByCoordinates(point_in_0_0_0);

  // add another block
  voxblox::Point point_in_10_0_0(ground_truth_layer->voxels_per_side(), 0, 0);
  ground_truth_layer->allocateNewBlockByCoordinates(point_in_10_0_0);

  {
    int gt_x_min = 4;
    int gt_x_max = 12;
    int gt_y_min = 3;
    int gt_y_max = 6;

    // write free space
    for (size_t x = 0; x < 2 * ground_truth_layer->voxels_per_side(); x++) {
      for (size_t y = 0; y < ground_truth_layer->voxels_per_side(); y++) {
        // global voxel by global index
        // auto voxel =
        voxblox::GlobalIndex voxelIdx(x, y, 0);

        voxblox::TsdfVoxel *voxel =
            ground_truth_layer->getVoxelPtrByGlobalIndex(voxelIdx);
        voxel->weight = 1;
        voxel->distance = 100;
      }
    }

    // write occupancy
    for (size_t x = gt_x_min; x < gt_x_max; x++) {
      for (size_t y = gt_y_min; y < gt_y_max; y++) {
        // global voxel by global index
        // auto voxel =
        voxblox::GlobalIndex voxelIdx(x, y, 0);

        voxblox::TsdfVoxel *voxel =
            ground_truth_layer->getVoxelPtrByGlobalIndex(voxelIdx);
        voxel->weight = 1;
        voxel->distance = 0.0f;
      }
    }

    int observed_x_min = 5;
    int observed_x_max = 8;
    int observed_y_min = 2;
    int observed_y_max = 4;

    // write free space
    for (size_t x = 0; x < observed_layer->voxels_per_side(); x++) {
      for (size_t y = 0; y < observed_layer->voxels_per_side(); y++) {
        // global voxel by global index
        // auto voxel =
        voxblox::GlobalIndex voxelIdx(x, y, 0);

        voxblox::TsdfVoxel *voxel =
            observed_layer->getVoxelPtrByGlobalIndex(voxelIdx);
        voxel->weight = 1;
        voxel->distance = 100;
      }
    }

    // add voxels to observed map
    for (size_t x = observed_x_min; x < observed_x_max; x++) {
      for (size_t y = observed_y_min; y < observed_y_max; y++) {
        // global voxel by global index
        // auto voxel =
        voxblox::GlobalIndex voxelIdx(x, y, 0);

        voxblox::TsdfVoxel *voxel =
            observed_layer->getVoxelPtrByGlobalIndex(voxelIdx);
        voxel->weight = 1;
        voxel->distance = 0.0f;
      }
    }
  }
}
// unit_test
void test_eval_metrics() {
  voxblox::TsdfMap::Ptr ground_truth_map;
  voxblox::TsdfMap::Ptr observed_map;
  voxblox::TsdfMap::Config config;
  config.tsdf_voxel_size = 1;
  config.tsdf_voxels_per_side = 8u;

  voxblox::Layer<voxblox::TsdfVoxel>::Ptr ground_truth_layer;
  voxblox::Layer<voxblox::TsdfVoxel>::Ptr observed_layer;

  ground_truth_layer =
      std::make_shared<voxblox::Layer<voxblox::TsdfVoxel>>(1, 8u);
  observed_layer = std::make_shared<voxblox::Layer<voxblox::TsdfVoxel>>(1, 8u);

  fill_dummy_data(ground_truth_layer, observed_layer);

  // test map intersection
  voxblox::GlobalIndexVector intersection_gt, difference_gt;
  calculate_Intersection_difference(*ground_truth_layer, *observed_layer,
                                    &intersection_gt, &difference_gt);

  CHECK(intersection_gt.size() == 3)
      << "Error in Intersection Algorithm! Recheck Implementation.";
  CHECK(difference_gt.size() == 21)
      << "Error in DIfference Algorithm! Recheck Implementation.";

  // test evaluation
  // todo - add quality metrics
  // update - depracated - added test to
  // ssc_map_eval_test_node
}
} // namespace tests
} // namespace ssc_mapping

#endif // SSC_MAPPING_EVAL_H_
