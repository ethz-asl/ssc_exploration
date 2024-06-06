#ifndef SSC_3D_PLANNING_EVAL_H_
#define SSC_3D_PLANNING_EVAL_H_

#include <voxblox/core/tsdf_map.h>

namespace active_3d_planning {
namespace map {

/**
 * SSCMap as a map representation for planning.
 */ 
class BaseEval {
 public:
  explicit BaseEval();
  
 protected:
  std::shared_ptr<TsdfMap> ground_truth_;
  std::shared_ptr<TsdfMap> observed_;

};

}  // namespace map
}  // namespace active_3d_planning

#endif  // SSC_3D_PLANNING_EVAL_H_
