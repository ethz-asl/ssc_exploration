
#ifndef SSC_MAP_H_
#define SSC_MAP_H_

#include <voxblox/core/common.h>
#include <voxblox/core/layer.h>
#include <voxblox/core/voxel.h>

#include "ssc_mapping/core/voxel.h"

namespace voxblox {

class SSCMap {
   public:
    struct Config {
        FloatingPoint ssc_voxel_size = 0.2;
        size_t ssc_voxels_per_side = 16u;

        std::string print() const;
    };

    explicit SSCMap(const Config& config)
        : ssc_layer_(new Layer<SSCOccupancyVoxel>(config.ssc_voxel_size, config.ssc_voxels_per_side)) {
        block_size_ = config.ssc_voxel_size * config.ssc_voxels_per_side;
    }

    Layer<SSCOccupancyVoxel>* getSSCLayerPtr() { return ssc_layer_.get(); }
    const Layer<SSCOccupancyVoxel>* getSSCLayerConstPtr() const { return ssc_layer_.get(); }
    const Layer<SSCOccupancyVoxel>& getSSCLayer() const { return *ssc_layer_; }

    FloatingPoint block_size() const { return block_size_; }
    FloatingPoint voxel_size() const { return ssc_layer_->voxel_size(); }
    
    bool isObserved(const Eigen::Vector3d& position) const;
    FloatingPoint block_size_;
    Layer<SSCOccupancyVoxel>::Ptr ssc_layer_;
};
}  // namespace voxblox
#endif //SSC_MAP_H_