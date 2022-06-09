project(testKmeans)
cmake_minimum_required(VERSION 2.8.3)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-Wall -O3 -g -std=c++11 -w")

find_package(Eigen3)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

add_executable(testKmeans testKmeans.cpp)

target_link_libraries(testKmeans ${OpenCV_LIBS} ${PCL_LIBRARIES})