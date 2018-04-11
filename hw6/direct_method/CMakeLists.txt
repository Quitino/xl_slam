cmake_minimum_required( VERSION 2.8 )
project( direct_method )

# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11" )

# 寻找OpenCV库
find_package( OpenCV REQUIRED )
# 添加头文件
include_directories( ${OpenCV_INCLUDE_DIRS} )

# eigen 
include_directories( "/usr/include/eigen3/" )

# 添加Pangolin依赖
find_package( Pangolin )
include_directories( ${Pangolin_INCLUDE_DIRS} )

# 添加Sophus依赖
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )

add_executable( direct_method direct_method.cpp )
# 链接OpenCV库
target_link_libraries( direct_method ${OpenCV_LIBS} )
# 链接Pangolin库
target_link_libraries( direct_method ${Pangolin_LIBRARIES} )
# 链接Sophus库
target_link_libraries( direct_method ${Sophus_LIBRARIES} )
