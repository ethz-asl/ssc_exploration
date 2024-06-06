#ifndef SSC_VISUALIZATION_H_
#define SSC_VISUALIZATION_H_

#include <visualization_msgs/MarkerArray.h>

#include <voxblox/core/common.h>
#include <voxblox/core/layer.h>
#include <voxblox/core/voxel.h>
#include <voxblox/utils/color_maps.h>

#include <voxblox_ros/mesh_vis.h>
#include <voxblox_ros/ptcloud_vis.h>

#include "ssc_mapping/visualization/color_map.h"
#include "ssc_mapping/core/voxel.h"

namespace voxblox {

bool visualizeSSCOccupancyVoxels(const SSCOccupancyVoxel& voxel, const Point& /*coord*/, Color* color);

void createPointcloudFromSSCLayer(const Layer<SSCOccupancyVoxel>& layer,
                                  pcl::PointCloud<pcl::PointXYZRGB>* pointcloud);

template <typename VoxelType>
void createOccupancyBlocksFromLayer(const Layer<VoxelType>& layer,
                                    const ShouldVisualizeVoxelColorFunctionType<VoxelType>& vis_function,
                                    const std::string& frame_id, visualization_msgs::MarkerArray* marker_array);

void createOccupancyBlocksFromSSCLayer(const Layer<SSCOccupancyVoxel>& layer, const std::string& frame_id,
                                       visualization_msgs::MarkerArray* marker_array);

template <typename VoxelType>
void createOccupancyBlocksFromLayer(const Layer<VoxelType>& layer,
                                    const ShouldVisualizeVoxelColorFunctionType<VoxelType>& vis_function,
                                    const std::string& frame_id, visualization_msgs::MarkerArray* marker_array) {
    CHECK_NOTNULL(marker_array);
    // Cache layer settings.
    size_t vps = layer.voxels_per_side();
    size_t num_voxels_per_block = vps * vps * vps;
    FloatingPoint voxel_size = layer.voxel_size();

    visualization_msgs::Marker block_marker;
    block_marker.header.frame_id = frame_id;
    block_marker.ns = "occupied_voxels";
    block_marker.id = 0;
    block_marker.type = visualization_msgs::Marker::CUBE_LIST;
    block_marker.scale.x = block_marker.scale.y = block_marker.scale.z = voxel_size;
    block_marker.action = visualization_msgs::Marker::ADD;

    BlockIndexList blocks;
    layer.getAllAllocatedBlocks(&blocks);
    for (const BlockIndex& index : blocks) {
        // Iterate over all voxels in said blocks.
        const Block<VoxelType>& block = layer.getBlockByIndex(index);

        for (size_t linear_index = 0; linear_index < num_voxels_per_block; ++linear_index) {
            Point coord = block.computeCoordinatesFromLinearIndex(linear_index);
            Color color;
            if (vis_function(block.getVoxelByLinearIndex(linear_index), coord, &color)) {
                geometry_msgs::Point cube_center;
                cube_center.x = coord.x();
                cube_center.y = coord.y();
                cube_center.z = coord.z();
                block_marker.points.push_back(cube_center);
                std_msgs::ColorRGBA color_msg;
                colorVoxbloxToMsg(color, &color_msg);
                block_marker.colors.push_back(color_msg);
            }
        }
    }
    marker_array->markers.push_back(block_marker);
}                                       
}  // namespace voxblox

namespace ssc_mapping {
    // Creates a pointcloud from voxel indices
template <typename T>
void createPointCloudFromVoxelIndices(const T& voxels,
                                      pcl::PointCloud<pcl::PointXYZRGB>* pointcloud, const voxblox::Color& color, const float voxel_size = 0.08) {
    for (auto voxel : voxels) {
        pcl::PointXYZRGB point;
        point.x = voxel.x() * voxel_size + (voxel_size/2);
        point.y = voxel.y() * voxel_size + (voxel_size/2);
        point.z = voxel.z() * voxel_size + (voxel_size/2);
        point.r = color.r;
        point.g = color.g;
        point.b = color.b;
        pointcloud->push_back(point);
    }
    pointcloud->header.frame_id = "world";
}

// Creates a pointcloud from voxel indices
template <typename T>
void createPointCloudFromVoxelIndices(const T& voxels,
                                      pcl::PointCloud<pcl::PointXYZRGBA>* pointcloud, const voxblox::Color& color, const float voxel_size = 0.08) {
    for (auto voxel : voxels) {
        pcl::PointXYZRGBA point;
        point.x = voxel.x() * voxel_size + (voxel_size/2);
        point.y = voxel.y() * voxel_size + (voxel_size/2);
        point.z = voxel.z() * voxel_size + (voxel_size/2);
        point.r = color.r;
        point.g = color.g;
        point.b = color.b;
        point.a = color.a;
        pointcloud->push_back(point);
    }
    pointcloud->header.frame_id = "world";
}
}

#endif  // SSC_VISUALIZATION_H_