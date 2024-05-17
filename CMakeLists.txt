cmake_minimum_required(VERSION 3.5)
project(SLAM_basics)

add_compile_options(-std=c++17)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -g -O3 -pg")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -O3 -pg")

#Eigen package
include_directories("/usr/include/eigen3")
find_package(Eigen3 REQUIRED)

#opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


#GTSAM
set(GTSAM_DIR "/home/shankara/ThirdParty/gtsam/build" )
set(GTSAM_UNSTABLE_DIR "/home/shankara/ThirdParty/gtsam/build")
find_package(GTSAM REQUIRED)
find_package(GTSAM_UNSTABLE REQUIRED)

set(GTSAM_LIBRARIES gtsam)

#include directories
include_directories(${PROJECT_SOURCE_DIR}/utils/include)
include_directories(${GTSAM_INCLUDE_DIR})

add_subdirectory(liegroups)
add_subdirectory(curve_fitting)
add_subdirectory(Visual_Odometry)