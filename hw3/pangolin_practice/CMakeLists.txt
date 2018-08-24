cmake_minimum_required( VERSION 2.8 )
project( pangolin_practice )

set(CMAKE_CXX_FLAGS "-std=c++11")

# 添加Eigen头文件
include_directories( "/usr/include/eigen3" )

# 添加Pangolin依赖
find_package( Pangolin )
include_directories( ${Pangolin_INCLUDE_DIRS} )

# 添加Sophus依赖
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )

add_executable( pangolin_practice pangolin_practice.cpp )
target_link_libraries( pangolin_practice ${Pangolin_LIBRARIES} )
target_link_libraries( pangolin_practice ${Sophus_LIBRARIES} )


