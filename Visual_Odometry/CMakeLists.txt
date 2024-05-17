cmake_minimum_required(VERSION 3.5)

project(VO)
add_compile_options(-std=c++17)

add_executable(feature_extract feature_extraction/orb_test.cpp)
target_link_libraries(feature_extract ${OpenCV_LIBS})