cmake_minimum_required( VERSION 2.8 )
project( optical_flow )

# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11" )

# 寻找OpenCV库
find_package( OpenCV REQUIRED )
# 添加头文件
include_directories( ${OpenCV_INCLUDE_DIRS} )

# eigen 
include_directories( "/usr/include/eigen3/" )


add_executable( optical_flow optical_flow.cpp )
# 链接OpenCV库
target_link_libraries( optical_flow ${OpenCV_LIBS} )
