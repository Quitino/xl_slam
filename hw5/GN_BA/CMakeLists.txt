cmake_minimum_required( VERSION 2.8 )
project( GN_BA )

set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# eigen 
include_directories( "/usr/include/eigen3/" )

# 添加Sophus依赖
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )


add_executable( GN_BA GN_BA.cpp )
target_link_libraries( GN_BA ${Sophus_LIBRARIES} )

