
#ifndef SSC_COLOR_MAP_H_
#define SSC_COLOR_MAP_H_

#include <voxblox/utils/color_maps.h>

namespace voxblox {

class SSCColorMap : public IdColorMap {
   public:
    SSCColorMap() : IdColorMap() {
        colors_.push_back(Color(22, 191, 206));   // 0 empty, free space
        colors_.push_back(Color(214, 38, 40));    // 1 ceiling
        colors_.push_back(Color(43, 160, 4));     // 2 floor
        colors_.push_back(Color(158, 216, 229));  // 3 wall
        colors_.push_back(Color(114, 158, 206));  // 4 window
        colors_.push_back(Color(204, 204, 91));   // 5 chair  new: 180, 220, 90
        colors_.push_back(Color(255, 186, 119));  // 6 bed
        colors_.push_back(Color(147, 102, 188));  // 7 sofa
        colors_.push_back(Color(30, 119, 181));   // 8 table
        colors_.push_back(Color(188, 188, 33));   // 9 tvs
        colors_.push_back(Color(255, 127, 12));   // 10 furn
        colors_.push_back(Color(196, 175, 214));  // 11 objects
        colors_.push_back(Color(153, 153, 153));  // 12 Accessible area, or label==255, ignore
    }

    virtual Color colorLookup(const size_t value) const { return colors_[value]; }

   protected:
    std::vector<Color> colors_;
};

}  // namespace voxblox
#endif //SSC_COLOR_MAP_H_