/**
 * Note:
 * Calculates coverage and quality metrics for a observed or predicted layer.
 * Accepts a target layer and a ground truth layer. The quality and coverage
 * metrics are appended to the input csv files. This code was also tested with
 * eval_plots.py for automating plots across runs and calcuating aggregated
 * metrics. I The code expects the file path provided already exists. For
 * calculating quality metrics specifically predicted layer while considering
 * observed tsdf voxels, use ssc_map_eval_quality_node.
 */
#include "ssc_mapping/eval/map_eval.h"
#include "ssc_mapping/utils/evaluation_utils.h"
#include <tuple>

std::string get_base_file_name(std::string path) {
  return path.substr(path.find_last_of("/\\") + 1,
                     path.find_last_of(".") - path.find_last_of("/\\") - 1);
}

std::string getMapIdFromFileName(std::string path) {
  // The ID is the last 5 digits of the name.
  const std::string name = get_base_file_name(path);
  constexpr size_t length = 5;
  if (length >= name.size()) {
    return name;
  }
  return name.substr(name.size() - length);
}

bool endsWith(std::string const &input, std::string const &ending) {
  if (input.length() >= ending.length()) {
    return (input.compare(input.length() - ending.length(), ending.length(),
                          ending) == 0);
  } else {
    return false;
  }
}

bool saveMetrics(
    const ssc_mapping::evaluation::QualityMetrics &quality_metrics,
    const ssc_mapping::evaluation::CoverageMetrics &coverage_metrics,
    const std::string &output_file_name, const std::string &map_id,
    bool verbose) {
  // Print metrics
  if (verbose) {
    quality_metrics.print();
    coverage_metrics.print();
    std::cout << "Adding metrics to " << output_file_name << std::endl;
  }

  // write metrics to file
  std::ofstream file(output_file_name, std::ios::app);

  if (file.is_open()) {
    file << map_id << "," << quality_metrics.precision_occ << ","
         << quality_metrics.precision_free << ","
         << quality_metrics.precision_overall << ","
         << quality_metrics.recall_occ << "," << quality_metrics.recall_free
         << "," << quality_metrics.IoU_occ << "," << quality_metrics.IoU_free
         << "," << coverage_metrics.explored_occ << ","
         << coverage_metrics.explored_free << ","
         << coverage_metrics.explored_overall << ","
         << coverage_metrics.coverage_occ << ","
         << coverage_metrics.coverage_free << ","
         << coverage_metrics.coverage_overall << std::endl;
    file.close();
  } else {
    std::cout << "Unable to open '" << output_file_name << "'!";
    return false;
  }
  return true;
}

IndexSet removeObservedVoxels(const IndexSet &input,
                              const VoxelEvalData &observed_map_data) {
  IndexSet result = input;
  result = ssc_mapping::evaluation::set_difference(
      result, std::get<2>(observed_map_data));
  result = ssc_mapping::evaluation::set_difference(
      result, std::get<3>(observed_map_data));
  return result;
}

int main(int argc, char **argv) {

  // init ros and google logging
  ros::init(argc, argv, "ssc_mapping_eval");
  google::InitGoogleLogging(argv[0]);
  google::SetCommandLineOption("GLOG_minloglevel", "0");
  google::InstallFailureSignalHandler();
  google::ParseCommandLineFlags(&argc, &argv, false);
  ros::NodeHandle nh("");
  ros::NodeHandle nh_private("~");

  // parse arguments
  std::string gt_layer_path;
  std::string tsdf_layer_path;
  std::string ssc_layer_path;
  std::string output_path;
  std::string eval_type; // tsdf, ssc, hierarchical, ssc_unobs, all
  std::string ssc_prefix;
  bool publish_visualization;
  bool refine_ob_layer;
  bool verbose;
  float
      ssc_confidence_threshold; // Threshold (in probability) for the SSC map to
                                // count a voxel as free or occupied. e.g. 0:
                                // [0,.5]->free, [.5,1]->occ, 0.1: [0,.4]->free,
                                // [.4,.6]->unknown, [.6,1]->occ.
  nh_private.getParam("gt_layer_path", gt_layer_path);
  nh_private.param("tsdf_layer_path", tsdf_layer_path, std::string());
  nh_private.param("ssc_layer_path", ssc_layer_path, std::string());
  nh_private.param("output_path", output_path, std::string());
  nh_private.param("ssc_prefix", ssc_prefix, std::string());
  nh_private.param("eval_type", eval_type, std::string("all"));
  nh_private.param("publish_visualization", publish_visualization, false);
  nh_private.param("refine_ob_layer", refine_ob_layer, true);
  nh_private.param("verbose", verbose, false);
  nh_private.param("ssc_confidence_threshold", ssc_confidence_threshold, 0.f);

  std::string ssc_suffix = "";
  if (ssc_confidence_threshold != 0.f) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << ssc_confidence_threshold;
    std::string s = ss.str();
    ssc_suffix = "_" + s.substr(2, 2);
  }

  // Load data.
  voxblox::Layer<voxblox::TsdfVoxel>::Ptr ground_truth_layer;
  voxblox::Layer<voxblox::TsdfVoxel>::Ptr tsdf_layer;
  voxblox::Layer<voxblox::SSCOccupancyVoxel>::Ptr ssc_layer;
  voxblox::io::LoadLayer<voxblox::TsdfVoxel>(gt_layer_path,
                                             &ground_truth_layer);
  VoxelEvalData tsdf_eval_data;
  VoxelEvalData ssc_eval_data;

  if (eval_type != "ssc") {
    voxblox::io::LoadLayer<voxblox::TsdfVoxel>(tsdf_layer_path, &tsdf_layer);
    tsdf_eval_data = get_voxel_data_from_layer(ground_truth_layer, tsdf_layer,
                                               refine_ob_layer);
  }
  if (eval_type != "tsdf") {
    voxblox::io::LoadLayer<voxblox::SSCOccupancyVoxel>(ssc_layer_path,
                                                       &ssc_layer);
    ssc_eval_data =
        get_voxel_data_from_layer(ground_truth_layer, ssc_layer,
                                  refine_ob_layer, ssc_confidence_threshold);
  }

  //##########################################
  // Evaluation Metrics
  //##########################################

  IndexSet gt_occ_voxels, gt_free_voxels, map_obs_occ_voxels,
      map_obs_free_voxels, gt_obs_occ, gt_obs_free, gt_unobs_occ, gt_unobs_free;

  if (eval_type == "tsdf" || eval_type == "all") {
    // TSDF metrics.
    std::tie(gt_occ_voxels, gt_free_voxels, map_obs_occ_voxels,
             map_obs_free_voxels, gt_obs_occ, gt_obs_free, gt_unobs_occ,
             gt_unobs_free) = tsdf_eval_data;

    auto quality_metrics = ssc_mapping::evaluation::calculate_quality_metrics(
        gt_occ_voxels, gt_obs_occ, gt_free_voxels, gt_obs_free,
        map_obs_occ_voxels, map_obs_free_voxels);
    auto coverage_metrics = ssc_mapping::evaluation::calculate_coverage_metrics(
        gt_occ_voxels, gt_obs_occ, gt_free_voxels, gt_obs_free,
        map_obs_occ_voxels, map_obs_free_voxels);

    saveMetrics(quality_metrics, coverage_metrics,
                output_path + "/metrics/tsdf.csv",
                getMapIdFromFileName(tsdf_layer_path), verbose);
  }

  if (eval_type == "ssc" || eval_type == "all") {
    // SSC metrics.
    std::tie(gt_occ_voxels, gt_free_voxels, map_obs_occ_voxels,
             map_obs_free_voxels, gt_obs_occ, gt_obs_free, gt_unobs_occ,
             gt_unobs_free) = ssc_eval_data;

    auto quality_metrics = ssc_mapping::evaluation::calculate_quality_metrics(
        gt_occ_voxels, gt_obs_occ, gt_free_voxels, gt_obs_free,
        map_obs_occ_voxels, map_obs_free_voxels);
    auto coverage_metrics = ssc_mapping::evaluation::calculate_coverage_metrics(
        gt_occ_voxels, gt_obs_occ, gt_free_voxels, gt_obs_free,
        map_obs_occ_voxels, map_obs_free_voxels);

    saveMetrics(quality_metrics, coverage_metrics,
                output_path + "/metrics/" + ssc_prefix + "ssc" + ssc_suffix +
                    ".csv",
                getMapIdFromFileName(ssc_layer_path), verbose);
  }

  if (eval_type == "hierarchical" || eval_type == "all") {
    // Hierarchical metrics.
    std::tie(gt_occ_voxels, gt_free_voxels, map_obs_occ_voxels,
             map_obs_free_voxels, gt_obs_occ, gt_obs_free, gt_unobs_occ,
             gt_unobs_free) = tsdf_eval_data;
    IndexSet ssc_obs_occ_voxels = std::get<2>(ssc_eval_data);
    IndexSet ssc_obs_free_voxels = std::get<3>(ssc_eval_data);

    // Remove observed voxels from SSC.
    ssc_obs_occ_voxels =
        removeObservedVoxels(ssc_obs_occ_voxels, tsdf_eval_data);
    ssc_obs_free_voxels =
        removeObservedVoxels(ssc_obs_free_voxels, tsdf_eval_data);

    // Add the SSC voxels to the observed map.
    map_obs_occ_voxels = ssc_mapping::evaluation::set_union(map_obs_occ_voxels,
                                                            ssc_obs_occ_voxels);
    map_obs_free_voxels = ssc_mapping::evaluation::set_union(
        map_obs_free_voxels, ssc_obs_free_voxels);
    IndexSet observed = ssc_mapping::evaluation::set_union(map_obs_occ_voxels,
                                                           map_obs_free_voxels);
    // Update the observed/unobserved GT.
    gt_obs_occ =
        ssc_mapping::evaluation::set_intersection(gt_occ_voxels, observed);
    gt_obs_free =
        ssc_mapping::evaluation::set_intersection(gt_free_voxels, observed);
    gt_unobs_occ =
        ssc_mapping::evaluation::set_intersection(gt_occ_voxels, gt_obs_occ);
    gt_unobs_free =
        ssc_mapping::evaluation::set_intersection(gt_free_voxels, gt_obs_free);

    // Evaluate and store the data.
    auto quality_metrics = ssc_mapping::evaluation::calculate_quality_metrics(
        gt_occ_voxels, gt_obs_occ, gt_free_voxels, gt_obs_free,
        map_obs_occ_voxels, map_obs_free_voxels);
    auto coverage_metrics = ssc_mapping::evaluation::calculate_coverage_metrics(
        gt_occ_voxels, gt_obs_occ, gt_free_voxels, gt_obs_free,
        map_obs_occ_voxels, map_obs_free_voxels);
    saveMetrics(quality_metrics, coverage_metrics,
                output_path + "/metrics/" + ssc_prefix + "hierarchical" +
                    ssc_suffix + ".csv",
                getMapIdFromFileName(tsdf_layer_path), verbose);
  }

  if (eval_type == "ssc_unobs" || eval_type == "all") {
    // SSC unobserved by TSDF metrics.
    std::tie(gt_occ_voxels, gt_free_voxels, map_obs_occ_voxels,
             map_obs_free_voxels, gt_obs_occ, gt_obs_free, gt_unobs_occ,
             gt_unobs_free) = ssc_eval_data;

    // discard voxels that are already measured fomr the ssc data and gt.
    map_obs_occ_voxels =
        removeObservedVoxels(map_obs_occ_voxels, tsdf_eval_data);
    map_obs_free_voxels =
        removeObservedVoxels(map_obs_free_voxels, tsdf_eval_data);
    gt_obs_occ = removeObservedVoxels(gt_obs_occ, tsdf_eval_data);
    gt_obs_free = removeObservedVoxels(gt_obs_free, tsdf_eval_data);

    // Evaluate and store the data.
    auto quality_metrics = ssc_mapping::evaluation::calculate_quality_metrics(
        gt_occ_voxels, gt_obs_occ, gt_free_voxels, gt_obs_free,
        map_obs_occ_voxels, map_obs_free_voxels);
    auto coverage_metrics = ssc_mapping::evaluation::calculate_coverage_metrics(
        gt_occ_voxels, gt_obs_occ, gt_free_voxels, gt_obs_free,
        map_obs_occ_voxels, map_obs_free_voxels);
    saveMetrics(quality_metrics, coverage_metrics,
                output_path + "/metrics/" + ssc_prefix + "ssc_unobs" +
                    ssc_suffix + ".csv",
                getMapIdFromFileName(tsdf_layer_path), verbose);
  }

  //#####################################
  // ROS Layer Visualizations
  //#####################################
  if (publish_visualization) {

    // publish
    std::cout << "Publishing voxels visualization!" << std::endl;

    auto missed_occupancy_observed_voxels_pub =
        nh_private.advertise<pcl::PointCloud<pcl::PointXYZRGB>>(
            "occupancy_pointcloud_observed_diff", 1, true);
    auto missed_occupancy_un_observed_voxels_pub =
        nh_private.advertise<pcl::PointCloud<pcl::PointXYZRGB>>(
            "occupancy_pointcloud_un_observed_diff", 1, true);
    auto correct_occupied_voxels_observed_pub =
        nh_private.advertise<pcl::PointCloud<pcl::PointXYZRGB>>(
            "occupancy_pointcloud_inter", 1, true);
    auto false_positive_observations_pub =
        nh_private.advertise<pcl::PointCloud<pcl::PointXYZRGB>>(
            "false_positive_observations", 1, true);
    auto free_gt_voxels_pub =
        nh_private.advertise<pcl::PointCloud<pcl::PointXYZRGB>>(
            "gt_free_voxels", 1, true);
    auto occ_gt_voxels_pub =
        nh_private.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("gt_occ_voxels",
                                                                1, true);
    auto unobserved_free_voxels_pub =
        nh_private.advertise<pcl::PointCloud<pcl::PointXYZRGBA>>(
            "unobserved_free_voxels", 1, true);

    // publish voxels that are occupied in ground truth but are unoccupied and
    // "observed" in observed map
    pcl::PointCloud<pcl::PointXYZRGB> pointcloud_observed_diff;
    ssc_mapping::createPointCloudFromVoxelIndices(
        ssc_mapping::evaluation::set_intersection(map_obs_free_voxels,
                                                  gt_occ_voxels),
        &pointcloud_observed_diff, voxblox::Color::Red());
    missed_occupancy_observed_voxels_pub.publish(pointcloud_observed_diff);

    // publish voxels that are occupied in ground truth but are unoccupied and
    // "not observed" in observed map
    pcl::PointCloud<pcl::PointXYZRGB> pointcloud_un_observed_occupied;
    ssc_mapping::createPointCloudFromVoxelIndices(
        gt_unobs_occ, &pointcloud_un_observed_occupied,
        voxblox::Color::Orange());
    missed_occupancy_un_observed_voxels_pub.publish(
        pointcloud_un_observed_occupied);

    // correctly observed occupied voxels
    pcl::PointCloud<pcl::PointXYZRGB> pointcloud_correct_occ;
    ssc_mapping::createPointCloudFromVoxelIndices(
        ssc_mapping::evaluation::set_intersection(map_obs_occ_voxels,
                                                  gt_occ_voxels),
        &pointcloud_correct_occ, voxblox::Color::Green());
    correct_occupied_voxels_observed_pub.publish(pointcloud_correct_occ);

    // false positive occupancy observations
    pcl::PointCloud<pcl::PointXYZRGB> pointcloud_observed_occupancy_fp;
    ssc_mapping::createPointCloudFromVoxelIndices(
        ssc_mapping::evaluation::set_difference(map_obs_occ_voxels,
                                                gt_occ_voxels),
        &pointcloud_observed_occupancy_fp, voxblox::Color::Yellow());
    false_positive_observations_pub.publish(pointcloud_observed_occupancy_fp);

    // free voxels in ground truth map
    pcl::PointCloud<pcl::PointXYZRGB> pointcloud_free_gt_voxels;
    ssc_mapping::createPointCloudFromVoxelIndices(
        gt_free_voxels, &pointcloud_free_gt_voxels, voxblox::Color::White());
    free_gt_voxels_pub.publish(pointcloud_free_gt_voxels);

    // occupied voxels in ground truth map
    pcl::PointCloud<pcl::PointXYZRGB> pointcloud_occ_gt_voxels;
    ssc_mapping::createPointCloudFromVoxelIndices(
        gt_occ_voxels, &pointcloud_occ_gt_voxels, voxblox::Color::Gray());
    occ_gt_voxels_pub.publish(pointcloud_occ_gt_voxels);

    ros::spin();
  }

  return 0;
}
