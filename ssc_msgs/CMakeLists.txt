cmake_minimum_required(VERSION 3.0.0)
project(ssc_msgs)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

##########
# EXPORT #
##########

cs_install()
cs_export()
