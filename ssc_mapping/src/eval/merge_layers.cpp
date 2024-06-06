

/////////////////////////////////////////////////////////////////////////
// Note:
// Merges a voxblox measured layer with ssc predicted layer and return 
// ssc occupancy layer 
// Mansoor Cheema - Oct 26, 2021
/////////////////////////////////////////////////////////////////////////

#include "ssc_mapping/eval/map_eval.h"
#include "ssc_mapping/utils/evaluation_utils.h"

typedef voxblox::Layer<voxblox::SSCOccupancyVoxel> SSC_Layer;
typedef voxblox::Layer<voxblox::TsdfVoxel> TSDFLayer;

template <typename VoxelTypeA, typename VoxelTypeB, typename VoxelTypeC>
void merge_layers(const voxblox::Layer<VoxelTypeA>& layer_a, const voxblox::Layer<VoxelTypeB>& layer_b,
                  std::shared_ptr<voxblox::Layer<VoxelTypeC>> layer_output) {
    // go over all voxels in layer_a

    size_t vps = layer_a.voxels_per_side();
    size_t num_voxels_per_block = vps * vps * vps;

    // get voxels from layer a
    voxblox::BlockIndexList blocks;
    layer_a.getAllAllocatedBlocks(&blocks);
    for (const voxblox::BlockIndex& block_idx : blocks) {
        const voxblox::Block<VoxelTypeA>& block = layer_a.getBlockByIndex(block_idx);

        for (size_t linear_index = 0; linear_index < num_voxels_per_block; ++linear_index) {
            auto voxel_idx = block.computeVoxelIndexFromLinearIndex(linear_index);
            auto global_voxel_idx =
                voxblox::getGlobalVoxelIndexFromBlockAndVoxelIndex(block_idx, voxel_idx, layer_a.voxels_per_side());
            auto voxel_a = layer_a.getVoxelPtrByGlobalIndex(global_voxel_idx);
            VoxelTypeC* voxel_c = layer_output->getVoxelPtrByGlobalIndex(global_voxel_idx);
            if (voxblox::utils::isObservedVoxel(*voxel_a)) {
                if (!voxel_c) {
                    auto block = layer_output->allocateBlockPtrByIndex(block_idx);
                    auto local_voxel_idx =
                        voxblox::getLocalFromGlobalVoxelIndex(global_voxel_idx, layer_output->voxels_per_side());
                    voxel_c = &block->getVoxelByVoxelIndex(local_voxel_idx);
                }
                bool is_voxel_a_occupied = voxblox::utils::isOccupied(*voxel_a, layer_a.voxel_size());
                if (is_voxel_a_occupied) {
                    // set voxel c as observed and occupied
                    voxel_c->observed = true;
                    voxel_c->label = 11;
                    voxel_c->probability_log = voxblox::logOddsFromProbability(0.95f);
                } else {
                    // set voxel c as observed and un occupied
                    voxel_c->observed = true;
                    voxel_c->probability_log = voxblox::logOddsFromProbability(0.1f);
                }
            }
        }
    }

    // add voxels from layer_b
    // get voxels from layer a
    blocks.clear();
    layer_b.getAllAllocatedBlocks(&blocks);
    for (const voxblox::BlockIndex& block_idx : blocks) {
        const voxblox::Block<VoxelTypeB>& block = layer_b.getBlockByIndex(block_idx);

        for (size_t linear_index = 0; linear_index < num_voxels_per_block; ++linear_index) {
            auto voxel_idx = block.computeVoxelIndexFromLinearIndex(linear_index);
            auto global_voxel_idx =
                voxblox::getGlobalVoxelIndexFromBlockAndVoxelIndex(block_idx, voxel_idx, layer_b.voxels_per_side());
            auto voxel_a = layer_a.getVoxelPtrByGlobalIndex(global_voxel_idx);
            auto voxel_b = layer_b.getVoxelPtrByGlobalIndex(global_voxel_idx);
            VoxelTypeC* voxel_c = layer_output->getVoxelPtrByGlobalIndex(global_voxel_idx);

            if (!voxel_c) {  // block not allocated yet
                             // allocate block voxels
                auto block = layer_output->allocateBlockPtrByIndex(block_idx);
                auto local_voxel_idx =
                    voxblox::getLocalFromGlobalVoxelIndex(global_voxel_idx, layer_output->voxels_per_side());
                voxel_c = &block->getVoxelByVoxelIndex(local_voxel_idx);
            }

            // voxel is observed in layer_a so leave it
            if (voxel_a && voxblox::utils::isObservedVoxel(*voxel_a)) {
                continue;
            }
            
            // not observed in layer a, check if its observed in layer b
            if (voxblox::utils::isObservedVoxel(*voxel_b)) {
                // voxel not observed in layer a but observed in layer b
                bool is_voxel_b_occupied = voxblox::utils::isOccupied(*voxel_b, layer_b.voxel_size());
                if (is_voxel_b_occupied) {
                    // set voxel c as observed and occupied
                    voxel_c->observed = true;
                    voxel_c->label = 11;
                    voxel_c->probability_log = voxblox::logOddsFromProbability(0.95f);
                } else {
                    // set voxel c as observed and un occupied
                    voxel_c->observed = true;
                    voxel_c->probability_log = voxblox::logOddsFromProbability(0.1f);
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "ssc_mapping_eval");
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    google::ParseCommandLineFlags(&argc, &argv, false);

    if (argc <4) {
        std::cout <<"usage: rosrun ssc_mapping merge_measured_predicted_layers_node measured.tsdf predicted.ssc output.mssc"<<std::endl;
        exit(-1);
    }

    // keep orignal occupied voxels. If false a layer of occupied voxels are
    // overlayed over free space.
    bool keep_occupancy = true;

    // parameters
    std::string tsdf_path;
    std::string ssc_path;
    std::string output_path;

    // 
    tsdf_path = argv[1];
    ssc_path = argv[2];
    output_path = argv[3];

    TSDFLayer::Ptr measured_layer;
    SSC_Layer::Ptr predicted_layer;
    SSC_Layer::Ptr output_layer;

    voxblox::io::LoadLayer<voxblox::TsdfVoxel>(tsdf_path, &measured_layer);
    voxblox::io::LoadLayer<voxblox::SSCOccupancyVoxel>(ssc_path, &predicted_layer);

    CHECK(measured_layer->voxel_size() == predicted_layer->voxel_size()) << "Layers should have same voxel size.";
    CHECK(measured_layer->voxels_per_side() == predicted_layer->voxels_per_side()) << "Layers should have same block size.";
    

    output_layer = std::make_shared<SSC_Layer>(measured_layer->voxel_size(),measured_layer->voxels_per_side());
    
    // add all voxels from measured layer to output layer. 
    // also add all blocks in predicted_layer that are not
    // observed in measured layer are also added to 
    // output layer
    merge_layers(*measured_layer, *predicted_layer, output_layer);
  
    output_layer->saveToFile(output_path, true);

    return 0;
}
