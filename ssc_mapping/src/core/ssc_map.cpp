#include "ssc_mapping/core/ssc_map.h"

namespace voxblox {
bool SSCMap::isObserved(const Eigen::Vector3d& position) const {
    // Get the block.
    Block<SSCOccupancyVoxel>::Ptr block_ptr = ssc_layer_->getBlockPtrByCoordinates(position.cast<FloatingPoint>());
    if (block_ptr) {
        const SSCOccupancyVoxel& voxel = block_ptr->getVoxelByCoordinates(position.cast<FloatingPoint>());
        return voxel.observed;
    }
    return false;
}
}  // namespace voxblox