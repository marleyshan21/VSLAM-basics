cmake_minimum_required(VERSION 3.5)

project(Sophus_test)
add_compile_options(-std=c++17)



SET(Sophus_DIR "/home/shankara/ThirdParty/Sophus/build")

find_package(Sophus REQUIRED)
if(Sophus_FOUND)
    message("Sophus library found")

else()
    message("Sophus not found")

endif()


add_executable(Sophus_test Sophus_test.cpp)
target_link_libraries(Sophus_test Sophus::Sophus)