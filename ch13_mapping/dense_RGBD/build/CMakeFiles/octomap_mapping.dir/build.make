# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /data/code/xl_slam/ch13_mapping/dense_RGBD

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /data/code/xl_slam/ch13_mapping/dense_RGBD/build

# Include any dependencies generated for this target.
include CMakeFiles/octomap_mapping.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/octomap_mapping.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/octomap_mapping.dir/flags.make

CMakeFiles/octomap_mapping.dir/octomap_mapping.cpp.o: CMakeFiles/octomap_mapping.dir/flags.make
CMakeFiles/octomap_mapping.dir/octomap_mapping.cpp.o: ../octomap_mapping.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/code/xl_slam/ch13_mapping/dense_RGBD/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/octomap_mapping.dir/octomap_mapping.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/octomap_mapping.dir/octomap_mapping.cpp.o -c /data/code/xl_slam/ch13_mapping/dense_RGBD/octomap_mapping.cpp

CMakeFiles/octomap_mapping.dir/octomap_mapping.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/octomap_mapping.dir/octomap_mapping.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/code/xl_slam/ch13_mapping/dense_RGBD/octomap_mapping.cpp > CMakeFiles/octomap_mapping.dir/octomap_mapping.cpp.i

CMakeFiles/octomap_mapping.dir/octomap_mapping.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/octomap_mapping.dir/octomap_mapping.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/code/xl_slam/ch13_mapping/dense_RGBD/octomap_mapping.cpp -o CMakeFiles/octomap_mapping.dir/octomap_mapping.cpp.s

CMakeFiles/octomap_mapping.dir/octomap_mapping.cpp.o.requires:

.PHONY : CMakeFiles/octomap_mapping.dir/octomap_mapping.cpp.o.requires

CMakeFiles/octomap_mapping.dir/octomap_mapping.cpp.o.provides: CMakeFiles/octomap_mapping.dir/octomap_mapping.cpp.o.requires
	$(MAKE) -f CMakeFiles/octomap_mapping.dir/build.make CMakeFiles/octomap_mapping.dir/octomap_mapping.cpp.o.provides.build
.PHONY : CMakeFiles/octomap_mapping.dir/octomap_mapping.cpp.o.provides

CMakeFiles/octomap_mapping.dir/octomap_mapping.cpp.o.provides.build: CMakeFiles/octomap_mapping.dir/octomap_mapping.cpp.o


# Object files for target octomap_mapping
octomap_mapping_OBJECTS = \
"CMakeFiles/octomap_mapping.dir/octomap_mapping.cpp.o"

# External object files for target octomap_mapping
octomap_mapping_EXTERNAL_OBJECTS =

octomap_mapping: CMakeFiles/octomap_mapping.dir/octomap_mapping.cpp.o
octomap_mapping: CMakeFiles/octomap_mapping.dir/build.make
octomap_mapping: /usr/local/lib/libopencv_cudabgsegm.so.3.3.0
octomap_mapping: /usr/local/lib/libopencv_cudaobjdetect.so.3.3.0
octomap_mapping: /usr/local/lib/libopencv_cudastereo.so.3.3.0
octomap_mapping: /usr/local/lib/libopencv_dnn.so.3.3.0
octomap_mapping: /usr/local/lib/libopencv_ml.so.3.3.0
octomap_mapping: /usr/local/lib/libopencv_shape.so.3.3.0
octomap_mapping: /usr/local/lib/libopencv_stitching.so.3.3.0
octomap_mapping: /usr/local/lib/libopencv_superres.so.3.3.0
octomap_mapping: /usr/local/lib/libopencv_videostab.so.3.3.0
octomap_mapping: /usr/lib/x86_64-linux-gnu/libboost_system.so
octomap_mapping: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
octomap_mapping: /usr/lib/x86_64-linux-gnu/libboost_thread.so
octomap_mapping: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
octomap_mapping: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
octomap_mapping: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
octomap_mapping: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
octomap_mapping: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
octomap_mapping: /usr/lib/x86_64-linux-gnu/libboost_regex.so
octomap_mapping: /usr/lib/x86_64-linux-gnu/libpthread.so
octomap_mapping: /usr/local/lib/libpcl_common.so
octomap_mapping: /usr/local/lib/libpcl_octree.so
octomap_mapping: /usr/lib/libOpenNI.so
octomap_mapping: /usr/local/lib/libpcl_io.so
octomap_mapping: /usr/lib/x86_64-linux-gnu/libflann_cpp.so
octomap_mapping: /usr/local/lib/libpcl_kdtree.so
octomap_mapping: /usr/local/lib/libpcl_search.so
octomap_mapping: /usr/local/lib/libpcl_sample_consensus.so
octomap_mapping: /usr/local/lib/libpcl_filters.so
octomap_mapping: /usr/lib/x86_64-linux-gnu/libboost_system.so
octomap_mapping: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
octomap_mapping: /usr/lib/x86_64-linux-gnu/libboost_thread.so
octomap_mapping: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
octomap_mapping: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
octomap_mapping: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
octomap_mapping: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
octomap_mapping: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
octomap_mapping: /usr/lib/x86_64-linux-gnu/libboost_regex.so
octomap_mapping: /usr/lib/x86_64-linux-gnu/libpthread.so
octomap_mapping: /usr/lib/libOpenNI.so
octomap_mapping: /usr/lib/x86_64-linux-gnu/libflann_cpp.so
octomap_mapping: /usr/lib/libvtkGenericFiltering.so.5.10.1
octomap_mapping: /usr/lib/libvtkGeovis.so.5.10.1
octomap_mapping: /usr/lib/libvtkCharts.so.5.10.1
octomap_mapping: /usr/local/lib/liboctomap.so
octomap_mapping: /usr/local/lib/liboctomath.so
octomap_mapping: /usr/local/lib/libopencv_cudafeatures2d.so.3.3.0
octomap_mapping: /usr/local/lib/libopencv_cudacodec.so.3.3.0
octomap_mapping: /usr/local/lib/libopencv_cudaoptflow.so.3.3.0
octomap_mapping: /usr/local/lib/libopencv_cudalegacy.so.3.3.0
octomap_mapping: /usr/local/lib/libopencv_calib3d.so.3.3.0
octomap_mapping: /usr/local/lib/libopencv_cudawarping.so.3.3.0
octomap_mapping: /usr/local/lib/libopencv_features2d.so.3.3.0
octomap_mapping: /usr/local/lib/libopencv_flann.so.3.3.0
octomap_mapping: /usr/local/lib/libopencv_highgui.so.3.3.0
octomap_mapping: /usr/local/lib/libopencv_objdetect.so.3.3.0
octomap_mapping: /usr/local/lib/libopencv_photo.so.3.3.0
octomap_mapping: /usr/local/lib/libopencv_cudaimgproc.so.3.3.0
octomap_mapping: /usr/local/lib/libopencv_cudafilters.so.3.3.0
octomap_mapping: /usr/local/lib/libopencv_cudaarithm.so.3.3.0
octomap_mapping: /usr/local/lib/libopencv_video.so.3.3.0
octomap_mapping: /usr/local/lib/libopencv_videoio.so.3.3.0
octomap_mapping: /usr/local/lib/libopencv_imgcodecs.so.3.3.0
octomap_mapping: /usr/local/lib/libopencv_imgproc.so.3.3.0
octomap_mapping: /usr/local/lib/libopencv_core.so.3.3.0
octomap_mapping: /usr/local/lib/libopencv_cudev.so.3.3.0
octomap_mapping: /usr/local/lib/libpcl_common.so
octomap_mapping: /usr/local/lib/libpcl_octree.so
octomap_mapping: /usr/local/lib/libpcl_io.so
octomap_mapping: /usr/local/lib/libpcl_kdtree.so
octomap_mapping: /usr/local/lib/libpcl_search.so
octomap_mapping: /usr/local/lib/libpcl_sample_consensus.so
octomap_mapping: /usr/local/lib/libpcl_filters.so
octomap_mapping: /usr/local/lib/liboctomap.so
octomap_mapping: /usr/local/lib/liboctomath.so
octomap_mapping: /usr/lib/libvtkViews.so.5.10.1
octomap_mapping: /usr/lib/libvtkInfovis.so.5.10.1
octomap_mapping: /usr/lib/libvtkWidgets.so.5.10.1
octomap_mapping: /usr/lib/libvtkVolumeRendering.so.5.10.1
octomap_mapping: /usr/lib/libvtkHybrid.so.5.10.1
octomap_mapping: /usr/lib/libvtkParallel.so.5.10.1
octomap_mapping: /usr/lib/libvtkRendering.so.5.10.1
octomap_mapping: /usr/lib/libvtkImaging.so.5.10.1
octomap_mapping: /usr/lib/libvtkGraphics.so.5.10.1
octomap_mapping: /usr/lib/libvtkIO.so.5.10.1
octomap_mapping: /usr/lib/libvtkFiltering.so.5.10.1
octomap_mapping: /usr/lib/libvtkCommon.so.5.10.1
octomap_mapping: /usr/lib/libvtksys.so.5.10.1
octomap_mapping: CMakeFiles/octomap_mapping.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/data/code/xl_slam/ch13_mapping/dense_RGBD/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable octomap_mapping"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/octomap_mapping.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/octomap_mapping.dir/build: octomap_mapping

.PHONY : CMakeFiles/octomap_mapping.dir/build

CMakeFiles/octomap_mapping.dir/requires: CMakeFiles/octomap_mapping.dir/octomap_mapping.cpp.o.requires

.PHONY : CMakeFiles/octomap_mapping.dir/requires

CMakeFiles/octomap_mapping.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/octomap_mapping.dir/cmake_clean.cmake
.PHONY : CMakeFiles/octomap_mapping.dir/clean

CMakeFiles/octomap_mapping.dir/depend:
	cd /data/code/xl_slam/ch13_mapping/dense_RGBD/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data/code/xl_slam/ch13_mapping/dense_RGBD /data/code/xl_slam/ch13_mapping/dense_RGBD /data/code/xl_slam/ch13_mapping/dense_RGBD/build /data/code/xl_slam/ch13_mapping/dense_RGBD/build /data/code/xl_slam/ch13_mapping/dense_RGBD/build/CMakeFiles/octomap_mapping.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/octomap_mapping.dir/depend

