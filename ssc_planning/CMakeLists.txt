cmake_minimum_required(VERSION 3.0.0)
project(ssc_planning)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


catkin_package()

cs_add_library(${PROJECT_NAME}
        src/initialization/ssc_package.cpp
        src/map/ssc_occ_map.cpp
        src/map/ssc_voxblox_map.cpp
        src/map/ssc_voxblox_criteria_map.cpp
        src/map/ssc_voxblox_oracle_map.cpp
        src/trajectory_evaluator/ssc_voxel_evaluator.cpp
        src/trajectory_evaluator/ssc_exploration_evaluator.cpp
        src/planner/exploration_planner_node.cpp
)
      
cs_add_executable(exploration_planner_node
        src/planner/exploration_planner_node.cpp
)
target_link_libraries(exploration_planner_node  ${PROJECT_NAME} ${catkin_LIBRARIES})


cs_install()
cs_export()
