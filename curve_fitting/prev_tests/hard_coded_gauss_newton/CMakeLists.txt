cmake_minimum_required(VERSION 3.15)
project(gaussnewton)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(gauss_newton gauss_newton.cpp)
target_link_libraries(gauss_newton ${OpenCV_LIBS})