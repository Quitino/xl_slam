cmake_minimum_required( VERSION 2.8 )
project( icp_align )

set(CMAKE_CXX_FLAGS "-std=c++11")

# 添加Eigen头文件
include_directories( "/usr/include/eigen3" )

# 添加Pangolin依赖
find_package( Pangolin )
include_directories( ${Pangolin_INCLUDE_DIRS} )

# 添加Sophus依赖
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )

add_executable( icp_align icp_align.cpp )
target_link_libraries( icp_align ${Pangolin_LIBRARIES} )
target_link_libraries( icp_align ${Sophus_LIBRARIES} )


