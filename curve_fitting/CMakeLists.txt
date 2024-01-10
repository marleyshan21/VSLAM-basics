cmake_minimum_required(VERSION 3.5)

project(curve_fitting)
add_compile_options(-std=c++17)

#find_package(GTSAM REQUIRED)
#find_package(GTSAM_UNSTABLE REQUIRED)


add_executable(curve_fit1 curve_fit1.cpp)
target_link_libraries(curve_fit1 gtsam)