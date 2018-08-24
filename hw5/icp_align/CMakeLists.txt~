cmake_minimum_required( VERSION 2.8 )
project( draw_trajectory )

set(CMAKE_CXX_FLAGS "-std=c++11")

# 添加Eigen头文件
include_directories( "/usr/include/eigen3" )

# 添加Pangolin依赖
find_package( Pangolin )
include_directories( ${Pangolin_INCLUDE_DIRS} )

# 添加Sophus依赖
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )

add_executable( draw_trajectory draw_trajectory.cpp )
target_link_libraries( draw_trajectory ${Pangolin_LIBRARIES} )
target_link_libraries( draw_trajectory ${Sophus_LIBRARIES} )


