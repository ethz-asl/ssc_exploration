cmake_minimum_required(VERSION 3.0.0)
project(ssc_network)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

catkin_package()

cs_install()
cs_export()
