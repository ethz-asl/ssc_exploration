
#include "ssc_mapping/eval/map_eval.h"
#include "ssc_mapping/utils/evaluation_utils.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "ssc_mapping_eval");
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    google::ParseCommandLineFlags(&argc, &argv, false);

    voxblox::Layer<voxblox::TsdfVoxel>::Ptr ground_truth_layer;
    voxblox::Layer<voxblox::TsdfVoxel>::Ptr observed_layer;

    ground_truth_layer = std::make_shared<voxblox::Layer<voxblox::TsdfVoxel>>(1, 8u);
    observed_layer = std::make_shared<voxblox::Layer<voxblox::TsdfVoxel>>(1, 8u);

    ssc_mapping::tests::fill_dummy_data(ground_truth_layer, observed_layer);

    CHECK(ground_truth_layer->voxel_size() == observed_layer->voxel_size())
        << "Error! Observed Layer and groundtruth layers should have same voxel size!";


    voxblox::LongIndexSet gt_occ_voxels, gt_free_voxels;
    get_free_and_occupied_voxels_from_layer(*ground_truth_layer, &gt_occ_voxels, &gt_free_voxels );

   voxblox::LongIndexSet map_obs_occ_voxels, map_obs_free_voxels;
    get_free_and_occupied_voxels_from_layer(*observed_layer, &map_obs_occ_voxels, &map_obs_free_voxels);
    
    voxblox::LongIndexSet gt_obs_occ, temp;
    split_observed_unobserved_voxels(*observed_layer, gt_occ_voxels, &gt_obs_occ, &temp);

    voxblox::LongIndexSet gt_obs_free;
    split_observed_unobserved_voxels(*observed_layer, gt_free_voxels, &gt_obs_free, &temp);

    auto quality_metrics = ssc_mapping::evaluation::calculate_quality_metrics(gt_occ_voxels, gt_obs_occ, gt_free_voxels, gt_obs_free, map_obs_occ_voxels, map_obs_free_voxels);
    auto coverage_metrics = ssc_mapping::evaluation::calculate_coverage_metrics(gt_occ_voxels, gt_obs_occ, gt_free_voxels, gt_obs_free, map_obs_occ_voxels, map_obs_free_voxels);
    
    //print metrics
    quality_metrics.print();
    coverage_metrics.print();

    // check quality metrics
    CHECK_NEAR(quality_metrics.precision_occ, 3.0/6.0, 1e-7);
    CHECK_NEAR(quality_metrics.precision_free, 49.0/58.0, 1e-7);
    CHECK_NEAR(quality_metrics.precision_overall, 52.0/64.0, 1e-7);

    CHECK_NEAR(quality_metrics.recall_occ, 3.0/12.0, 1e-7);
    CHECK_NEAR(quality_metrics.recall_free, 49.0/52.0, 1e-7);

    CHECK_NEAR(quality_metrics.IoU_occ, 3.0/15.0, 1e-7);
    CHECK_NEAR(quality_metrics.IoU_free, 49.0/61.0, 1e-7);

    //check exploration metrics
    CHECK_NEAR(coverage_metrics.explored_occ, 12.0/24.0, 1e-7);
    CHECK_NEAR(coverage_metrics.explored_free, 52.0/104.0, 1e-7);
    CHECK_NEAR(coverage_metrics.explored_overall, 64.0/128.0, 1e-7);

    CHECK_NEAR(coverage_metrics.coverage_occ, 3.0/24.0, 1e-7);
    CHECK_NEAR(coverage_metrics.coverage_free, 49.0/104.0, 1e-7);
    CHECK_NEAR(coverage_metrics.coverage_overall, 52.0/128.0, 1e-7);
    return 0;
}
