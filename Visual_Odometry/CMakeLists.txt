cmake_minimum_required(VERSION 3.5)

project(VO)
add_compile_options(-std=c++17)

add_executable(feature_extract 2d_2d/orb_test.cpp)
target_link_libraries(feature_extract ${OpenCV_LIBS})

add_executable(epipolar_test 2d_2d/epipolar_test.cpp)
target_link_libraries(epipolar_test ${OpenCV_LIBS})