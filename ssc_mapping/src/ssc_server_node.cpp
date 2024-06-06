
#include <ros/ros.h>
#include <glog/logging.h>
#include "ssc_mapping/ros/ssc_server.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "ssc_mapping");
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    google::ParseCommandLineFlags(&argc, &argv, false);

    ros::NodeHandle nh("");
    ros::NodeHandle nh_private("~");
    
    std::unique_ptr<voxblox::SSCServer> ssc_server;
    ssc_server.reset(new voxblox::SSCServer(nh, nh_private));
    ros::spin();
    return 0;
}
