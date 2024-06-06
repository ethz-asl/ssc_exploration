#ifndef SSC_SERVER_VOXBLOX_H_
#define SSC_SERVER_VOXBLOX_H_

#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>

#include <ssc_msgs/SSCGrid.h>
#include <voxblox/core/layer.h>
#include <voxblox/integrator/integrator_utils.h>
#include <voxblox/io/layer_io.h>
#include <voxblox_msgs/FilePath.h>
#include <voxblox_ros/transformer.h>
#include <voxblox_ros/conversions.h>
#include "ssc_mapping/core/ssc_map.h"
#include "ssc_mapping/fusion/base_fusion.h"

namespace voxblox {

class SSCServer {
   public:
    SSCServer(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
        : SSCServer(nh, nh_private, getFusionConfigROSParam(nh, nh_private), getSSCMapConfigFromRosParam(nh_private)) {}

    SSCServer(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private, const ssc_fusion::BaseFusion::Config&, const SSCMap::Config&);

    static ssc_fusion::BaseFusion::Config getFusionConfigROSParam(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);

    static SSCMap::Config getSSCMapConfigFromRosParam(const ros::NodeHandle& nh_private);

    void sscCallback(const ssc_msgs::SSCGrid::ConstPtr& msg);

    void insertPointcloud(const sensor_msgs::PointCloud2::Ptr& pointcloud);

    void setWorldFrame(const std::string& world_frame) { world_frame_ = world_frame; }

    std::string getWorldFrame() const { return world_frame_; }

    virtual void clear() { ssc_map_->getSSCLayerPtr()->removeAllBlocks(); }

    inline std::shared_ptr<SSCMap> getSSCMapPtr() { return ssc_map_; }
    inline std::shared_ptr<const SSCMap> getSSCMapPtr() const { return ssc_map_; }

    void publishSSCOccupancyPoints();

    void publishSSCOccupiedNodes();

    virtual bool saveMap(const std::string& file_path);

    bool saveMapCallback(voxblox_msgs::FilePath::Request& request,     // NOLINT
                       voxblox_msgs::FilePath::Response& response); 

   private:
    bool publish_pointclouds_on_update_;
    bool integrate_point_cloud_;
    float decay_weight_std_;
    std::string world_frame_;
    std::shared_ptr<SSCMap> ssc_map_;
    std::shared_ptr<ssc_fusion::BaseFusion> base_fusion_;

    // TEST: Evaluation utils.
    bool evaluate_ = false;
    std::string evaluation_path_;
    double next_save_time_ = 0.0;  // s
    double time_offset_ = -1.0; // s
    double save_interval_ = 30.0;  // s 
    double last_message_received_;
    int save_map_id_ = 0;
    ros::Timer shutdown_timer_;
    void timerCallback(const ros::TimerEvent& /* event */);

    //services/publishers/subscribers
    ros::ServiceServer save_map_srv_;
    ros::Subscriber ssc_map_sub_;
    ros::Subscriber pointcloud_sub_;
    ros::Publisher ssc_pointcloud_pub_;
    ros::Publisher occupancy_marker_pub_;
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;
    //tf::TransformListener tf_listener_;
    Transformer transformer_;
};

}  // namespace voxblox

#endif  // SSC_SERVER_VOXBLOX_H_
