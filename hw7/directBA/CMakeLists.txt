cmake_minimum_required( VERSION 2.8 )
project( directBA )

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

SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear 
    g2o_types_icp g2o_types_slam2d g2o_types_sba g2o_types_slam3d g2o_core g2o_interface 
    g2o_solver_csparse g2o_solver_structure_only g2o_csparse_extension g2o_opengl_helper g2o_solver_dense 
    g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse )

# 寻找CSparse
find_package( CSparse REQUIRED )
include_directories( ${CSPARSE_INCLUDE_DIR} )


add_executable( directBA directBA.cpp )
# 链接OpenCV库
target_link_libraries( directBA ${OpenCV_LIBS} )
# 链接Pangolin库
target_link_libraries( directBA ${Pangolin_LIBRARIES} )
# 链接Sophus库
target_link_libraries( directBA ${Sophus_LIBRARIES} )
# 链接CSparse库
target_link_libraries( directBA ${CSPARSE_INCLUDE_DIR} )
# 链接G2O库
#target_link_libraries( directBA g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension )

target_link_libraries( directBA ${G2O_LIBS} )
