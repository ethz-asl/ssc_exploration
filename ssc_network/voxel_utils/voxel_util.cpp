#include <math.h>
#include <stdio.h>

#include <algorithm>
#include <cstring>

void depth2Grid(double* cam_info, double* vox_info, double* depth_data, double* vox_binary,
                double* depth2voxel_idx) {
    // Get camera information
    int frame_width = cam_info[0];
    int frame_height = cam_info[1];
    double cam_K[9];
    for (int i = 0; i < 9; ++i) cam_K[i] = cam_info[i + 2];
    double cam_pose[16];
    for (int i = 0; i < 16; ++i) cam_pose[i] = cam_info[i + 11];

    // Get voxel volume parameters
    double vox_unit = vox_info[0];
    // double vox_margin = vox_info[1];
    int vox_size[3];
    for (int i = 0; i < 3; ++i) vox_size[i] = vox_info[i + 2];
    double vox_origin[3];
    for (int i = 0; i < 3; ++i) vox_origin[i] = vox_info[i + 5];

    // Get point in world coordinate
    #pragma omp parallel for
    for (size_t pixel_x = 0; pixel_x < frame_width; pixel_x++) {
        for (size_t pixel_y = 0; pixel_y < frame_height; pixel_y++) {
            double point_depth = depth_data[pixel_y * frame_width + pixel_x];

            double point_cam[3] = {0};
            point_cam[0] = (pixel_x - cam_K[2]) * point_depth / cam_K[0];
            point_cam[1] = (pixel_y - cam_K[5]) * point_depth / cam_K[4];
            point_cam[2] = point_depth;

            double point_base[3] = {0};

            point_base[0] = cam_pose[0 * 4 + 0] * point_cam[0] + cam_pose[0 * 4 + 1] * point_cam[1] +
                            cam_pose[0 * 4 + 2] * point_cam[2];
            point_base[1] = cam_pose[1 * 4 + 0] * point_cam[0] + cam_pose[1 * 4 + 1] * point_cam[1] +
                            cam_pose[1 * 4 + 2] * point_cam[2];
            point_base[2] = cam_pose[2 * 4 + 0] * point_cam[0] + cam_pose[2 * 4 + 1] * point_cam[1] +
                            cam_pose[2 * 4 + 2] * point_cam[2];

            point_base[0] = point_base[0] + cam_pose[0 * 4 + 3];
            point_base[1] = point_base[1] + cam_pose[1 * 4 + 3];
            point_base[2] = point_base[2] + cam_pose[2 * 4 + 3];

            // World coordinate to grid coordinate
            int z = (int)floor((point_base[0] - vox_origin[0]) / vox_unit);
            int x = (int)floor((point_base[1] - vox_origin[1]) / vox_unit);
            int y = (int)floor((point_base[2] - vox_origin[2]) / vox_unit);

            // mark vox_binary
            if (x >= 0 && x < vox_size[0] && y >= 0 && y < vox_size[1] && z >= 0 && z < vox_size[2]) {
                int vox_idx = z * vox_size[0] * vox_size[1] + y * vox_size[0] + x;
                vox_binary[vox_idx] = double(1.0);
                depth2voxel_idx[pixel_y * frame_width + pixel_x] = vox_idx;
            }
        }
    }
}

void tsdfTransform(double* vox_info, double* vox_tsdf) {
    int vox_size[3];

    // initialize voxel size
    for (int i = 0; i < 3; ++i) {
        vox_size[i] = vox_info[i + 2];
    }

    size_t num_voxels = vox_size[0] * vox_size[1] * vox_size[2];

    #pragma omp parallel for
    for (size_t vox_idx = 0; vox_idx < num_voxels; ++vox_idx) {
        double value = double(vox_tsdf[vox_idx]);

        double sign;
        if (abs(value) < 0.001)
            sign = 1;
        else
            sign = value / abs(value);

        vox_tsdf[vox_idx] = sign * (std::max(0.001, (1.0 - abs(value))));
    }
}

void squaredDistanceTransform(double* cam_info, double* vox_info, double* depth_data, double* vox_binary,
                              double* vox_tsdf) {
    // Get voxel volume parameters
    double vox_unit = vox_info[0];
    double vox_margin = vox_info[1];
    int vox_size[3];
    double vox_origin[3];

    // load voxel size from parameter array
    for (int i = 0; i < 3; ++i) {
        vox_size[i] = vox_info[i + 2];
    }

    // load voxel origin from parameter array
    for (int i = 0; i < 3; ++i) {
        vox_origin[i] = vox_info[i + 5];
    }

    int frame_width = cam_info[0];
    int frame_height = cam_info[1];

    // load camera parameters
    const unsigned int CAMERA_INTRINSIC_MATRIX_SIZE = 9;
    const unsigned int CAMERA_POSE_MATRIX_SIZE = 16;

    double cam_K[CAMERA_INTRINSIC_MATRIX_SIZE];
    double cam_pose[CAMERA_POSE_MATRIX_SIZE];

    for (int i = 0; i < CAMERA_INTRINSIC_MATRIX_SIZE; ++i) {
        cam_K[i] = cam_info[i + 2];
    }

    for (int i = 0; i < CAMERA_POSE_MATRIX_SIZE; ++i) {
        cam_pose[i] = cam_info[i + 11];
    }

    // Total voxels in a fixed 3d volume of voxel_size
    size_t num_voxels = vox_size[0] * vox_size[1] * vox_size[2];

    #pragma omp parallel for
    for (size_t vox_idx = 0; vox_idx < num_voxels; ++vox_idx) {
        int z = double((vox_idx / (vox_size[0] * vox_size[1])) % vox_size[2]);
        int y = double((vox_idx / vox_size[0]) % vox_size[1]);
        int x = double(vox_idx % vox_size[0]);
        int search_region = (int)round(vox_margin / vox_unit);

        if (vox_binary[vox_idx] > 0) {
            vox_tsdf[vox_idx] = 0;
            continue;
        }

        // Get point in world coordinates (XYZ) from grid coordinates (YZX)
        double point_base[3] = {0};
        point_base[0] = double(z) * vox_unit + vox_origin[0];
        point_base[1] = double(x) * vox_unit + vox_origin[1];
        point_base[2] = double(y) * vox_unit + vox_origin[2];

        // Get point in current camera coordinates
        double point_cam[3] = {0};
        point_base[0] = point_base[0] - cam_pose[0 * 4 + 3];
        point_base[1] = point_base[1] - cam_pose[1 * 4 + 3];
        point_base[2] = point_base[2] - cam_pose[2 * 4 + 3];
        point_cam[0] = cam_pose[0 * 4 + 0] * point_base[0] + cam_pose[1 * 4 + 0] * point_base[1] +
                       cam_pose[2 * 4 + 0] * point_base[2];
        point_cam[1] = cam_pose[0 * 4 + 1] * point_base[0] + cam_pose[1 * 4 + 1] * point_base[1] +
                       cam_pose[2 * 4 + 1] * point_base[2];
        point_cam[2] = cam_pose[0 * 4 + 2] * point_base[0] + cam_pose[1 * 4 + 2] * point_base[1] +
                       cam_pose[2 * 4 + 2] * point_base[2];
        if (point_cam[2] <= 0) {
            continue;
        }

        // Project point to 2D
        int pixel_x = roundf(cam_K[0] * (point_cam[0] / point_cam[2]) + cam_K[2]);
        int pixel_y = roundf(cam_K[4] * (point_cam[1] / point_cam[2]) + cam_K[5]);
        if (pixel_x < 0 || pixel_x >= frame_width || pixel_y < 0 || pixel_y >= frame_height) {  // outside FOV
            continue;
        }

        // Get depth
        double point_depth = depth_data[pixel_y * frame_width + pixel_x];
        if (point_depth < double(0.5f) || point_depth > double(8.0f)) {
            continue;
        }
        if (roundf(point_depth) == 0) {  // mising depth
            vox_tsdf[vox_idx] = double(-1.0);
            continue;
        }

        // Get depth difference
        double sign;
        if (abs(point_depth - point_cam[2]) < 0.0001) {
            sign = 1;  // avoid NaN
        } else {
            sign = (point_depth - point_cam[2]) / abs(point_depth - point_cam[2]);
        }
        vox_tsdf[vox_idx] = double(sign);
        for (int iix = std::max(0, x - search_region); iix < std::min((int)vox_size[0], x + search_region + 1); iix++) {
            for (int iiy = std::max(0, y - search_region); iiy < std::min((int)vox_size[1], y + search_region + 1);
                 iiy++) {
                for (int iiz = std::max(0, z - search_region); iiz < std::min((int)vox_size[2], z + search_region + 1);
                     iiz++) {
                    int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
                    if (vox_binary[iidx] > 0) {
                        double xd = std::abs(x - iix);
                        double yd = std::abs(y - iiy);
                        double zd = std::abs(z - iiz);
                        double tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd) / (double)search_region;
                        if (tsdf_value < std::abs(vox_tsdf[vox_idx])) {
                            vox_tsdf[vox_idx] = double(tsdf_value * sign);
                        }
                    }
                }
            }
        }
    }
}

void ComputeTSDF(double* cam_info, double* vox_info, double* depth_data, double* vox_tsdf,
                 double* depth_mapping_idxs, double* occupancy) {
    int frame_width = cam_info[0];
    int frame_height = cam_info[1];
    int vox_size[3];

    for (int i = 0; i < 3; ++i) {
        vox_size[i] = vox_info[i + 2];
    }
    int num_crop_voxels = vox_size[0] * vox_size[1] * vox_size[2];

    // allocate voxel occupancy
    double* vox_binary = (double*)malloc((int)(num_crop_voxels * sizeof(double)));
    std::memset(vox_binary, 0, num_crop_voxels * sizeof(double));

    // from depth map to binaray voxel representation
    depth2Grid(cam_info, vox_info, depth_data, vox_binary, depth_mapping_idxs);
    squaredDistanceTransform(cam_info, vox_info, depth_data, vox_binary, vox_tsdf);
    tsdfTransform(vox_info, vox_tsdf);  // invert TSDF

    // copy computed TSDF back
    memcpy(occupancy, vox_binary, num_crop_voxels * sizeof(double));
    free(vox_binary);
}

void calculate_occupancy_prob(double* cam_info, double* vox_info, double* depth_data, double* vox_log) {
    // Get camera information
    int frame_width = cam_info[0];
    int frame_height = cam_info[1];
    double cam_K[9];
    for (int i = 0; i < 9; ++i) cam_K[i] = cam_info[i + 2];
    double cam_pose[16];
    for (int i = 0; i < 16; ++i) cam_pose[i] = cam_info[i + 11];

    // Get voxel volume parameters
    double vox_unit = vox_info[0];
    // double vox_margin = vox_info[1];
    int vox_size[3];
    for (int i = 0; i < 3; ++i) vox_size[i] = vox_info[i + 2];
    double vox_origin[3];
    for (int i = 0; i < 3; ++i) vox_origin[i] = vox_info[i + 5];

    // Get point in world coordinate
    #pragma omp parallel for
    for (size_t pixel_x = 0; pixel_x < frame_width; pixel_x++) {
        for (size_t pixel_y = 0; pixel_y < frame_height; pixel_y++) {
            double point_depth = depth_data[pixel_y * frame_width + pixel_x];

            double point_cam[3] = {0};
            point_cam[0] = (pixel_x - cam_K[2]) * point_depth / cam_K[0];
            point_cam[1] = (pixel_y - cam_K[5]) * point_depth / cam_K[4];
            point_cam[2] = point_depth;

            double point_base[3] = {0};

            point_base[0] = cam_pose[0 * 4 + 0] * point_cam[0] + cam_pose[0 * 4 + 1] * point_cam[1] +
                            cam_pose[0 * 4 + 2] * point_cam[2];
            point_base[1] = cam_pose[1 * 4 + 0] * point_cam[0] + cam_pose[1 * 4 + 1] * point_cam[1] +
                            cam_pose[1 * 4 + 2] * point_cam[2];
            point_base[2] = cam_pose[2 * 4 + 0] * point_cam[0] + cam_pose[2 * 4 + 1] * point_cam[1] +
                            cam_pose[2 * 4 + 2] * point_cam[2];

            point_base[0] = point_base[0] + cam_pose[0 * 4 + 3];
            point_base[1] = point_base[1] + cam_pose[1 * 4 + 3];
            point_base[2] = point_base[2] + cam_pose[2 * 4 + 3];

            // World coordinate to grid coordinate
            int z = (int)floor((point_base[0] - vox_origin[0]) / vox_unit);
            int x = (int)floor((point_base[1] - vox_origin[1]) / vox_unit);
            int y = (int)floor((point_base[2] - vox_origin[2]) / vox_unit);

            double prob_occ = 0.7f;
            double max_prob = 0.97f;

            if (x >= 0 && x < vox_size[0] && y >= 0 && y < vox_size[1] && z >= 0 && z < vox_size[2]) {
                int vox_idx = z * vox_size[0] * vox_size[1] + y * vox_size[0] + x;
                vox_log[vox_idx] = std::min(vox_log[vox_idx] + std::log(prob_occ / (1 - prob_occ)),
                                                std::log(max_prob / (1 - max_prob)));
            }
        }
    }
}

void calculateOccupancyProb(double* cam_info, double* vox_info, double* depth_data,
                            double* log_odds_occupancy) {
    int frame_width = cam_info[0];
    int frame_height = cam_info[1];
    int vox_size[3];

    for (int i = 0; i < 3; ++i) {
        vox_size[i] = vox_info[i + 2];
    }
    int num_crop_voxels = vox_size[0] * vox_size[1] * vox_size[2];

    // allocate voxel occupancy
    double* vox_prob = (double*)malloc((int)(num_crop_voxels * sizeof(double)));
    std::memset(vox_prob, 0, num_crop_voxels * sizeof(double));

    // from depth map to binary voxel representation
    calculate_occupancy_prob(cam_info, vox_info, depth_data, vox_prob);

    // copy computed log odds back to CPU
    memcpy(log_odds_occupancy, vox_prob, num_crop_voxels * sizeof(double));

    // deallocation
    free(vox_prob);
}
