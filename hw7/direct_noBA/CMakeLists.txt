cmake_minimum_required( VERSION 2.8 )
project( direct_noBA )

# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11" )

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

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

# 寻找G2O
find_package( G2O REQUIRED )
include_directories( ${G2O_INCLUDE_DIRS} )
# 寻找CSparse
find_package( CSparse REQUIRED )
include_directories( ${CSPARSE_INCLUDE_DIR} )


add_executable( direct_noBA direct_noBA.cpp )
# 链接OpenCV库
target_link_libraries( direct_noBA ${OpenCV_LIBS} )
# 链接Pangolin库
target_link_libraries( direct_noBA ${Pangolin_LIBRARIES} )
# 链接Sophus库
target_link_libraries( direct_noBA ${Sophus_LIBRARIES} )
# 链接CSparse库
target_link_libraries( direct_noBA ${CSPARSE_INCLUDE_DIR} )
# 链接G2O库
target_link_libraries( direct_noBA g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension )
