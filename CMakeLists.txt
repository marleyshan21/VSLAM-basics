cmake_minimum_required(VERSION 3.2)
project(SLAM_basics)

add_compile_options(-std=c++17)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -g -O3 -pg")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -O3 -pg")

#Eigen package
include_directories("/usr/include/eigen3")
find_package(Eigen3 REQUIRED)
message("EIGEN_INCLUDE_DIR :" ${EIGEN_INCLUDE_DIR})

add_subdirectory(liegroups)