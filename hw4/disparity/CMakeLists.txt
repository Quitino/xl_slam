cmake_minimum_required( VERSION 2.8 )
project( disparity )

set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

<<<<<<< HEAD
set(OpenCV_LIBS_DIR "/data/lib/opencv-3.0.0/release/lib")
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_calib3d opencv_video opencv_features2d opencv_ml opencv_highgui opencv_objdetect opencv_contrib opencv_legacy opencv_imgcodecs)

# opencv 
#find_package( OpenCV REQUIRED )
include_directories( "/data/lib/opencv-3.0.0/release/include" )
=======
# opencv 
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# 添加Pangolin依赖
find_package( Pangolin )
include_directories( ${Pangolin_INCLUDE_DIRS} )
>>>>>>> bd44e8eb5325dd543994f357470a7511a2c09a07

# eigen 
include_directories( "/usr/include/eigen3/" )

<<<<<<< HEAD
link_directories(${OpenCV_LIBS_DIR})

=======
>>>>>>> bd44e8eb5325dd543994f357470a7511a2c09a07
# pcl 
find_package( PCL REQUIRED COMPONENT common io )
include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )

add_executable( disparity disparity.cpp )
target_link_libraries( disparity ${OpenCV_LIBS} ${PCL_LIBRARIES} )
<<<<<<< HEAD
=======
target_link_libraries( disparity ${Pangolin_LIBRARIES} )
>>>>>>> bd44e8eb5325dd543994f357470a7511a2c09a07
