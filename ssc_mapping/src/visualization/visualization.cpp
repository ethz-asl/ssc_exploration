

#include "ssc_mapping/visualization/visualization.h"
#include "ssc_mapping/utils/voxel_utils.h"

namespace voxblox {

bool visualizeSSCOccupancyVoxels(const SSCOccupancyVoxel& voxel, const Point& /*coord*/, Color* color) {
    CHECK_NOTNULL(color);
    static SSCColorMap map;
    if (utils::isOccupied(voxel, 0.f)) { 
        *color = map.colorLookup(voxel.label);
        return true;
    }
    return false;
}

void createPointcloudFromSSCLayer(const Layer<SSCOccupancyVoxel>& layer,
                                  pcl::PointCloud<pcl::PointXYZRGB>* pointcloud) {
    CHECK_NOTNULL(pointcloud);
    createColorPointcloudFromLayer<SSCOccupancyVoxel>(layer, &visualizeSSCOccupancyVoxels, pointcloud);
}



void createOccupancyBlocksFromSSCLayer(const Layer<SSCOccupancyVoxel>& layer, const std::string& frame_id,
                                       visualization_msgs::MarkerArray* marker_array) {
    CHECK_NOTNULL(marker_array);
    createOccupancyBlocksFromLayer<SSCOccupancyVoxel>(layer, &visualizeSSCOccupancyVoxels, frame_id, marker_array);
}
}  // namespace voxblox
