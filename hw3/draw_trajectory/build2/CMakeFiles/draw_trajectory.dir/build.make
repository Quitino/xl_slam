# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /data2/code2/xl_slam/hw3/draw_trajectory

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /data2/code2/xl_slam/hw3/draw_trajectory/build2

# Include any dependencies generated for this target.
include CMakeFiles/draw_trajectory.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/draw_trajectory.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/draw_trajectory.dir/flags.make

CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o: CMakeFiles/draw_trajectory.dir/flags.make
CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o: ../draw_trajectory.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data2/code2/xl_slam/hw3/draw_trajectory/build2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o -c /data2/code2/xl_slam/hw3/draw_trajectory/draw_trajectory.cpp

CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data2/code2/xl_slam/hw3/draw_trajectory/draw_trajectory.cpp > CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.i

CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data2/code2/xl_slam/hw3/draw_trajectory/draw_trajectory.cpp -o CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.s

CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o.requires:

.PHONY : CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o.requires

CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o.provides: CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o.requires
	$(MAKE) -f CMakeFiles/draw_trajectory.dir/build.make CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o.provides.build
.PHONY : CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o.provides

CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o.provides.build: CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o


# Object files for target draw_trajectory
draw_trajectory_OBJECTS = \
"CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o"

# External object files for target draw_trajectory
draw_trajectory_EXTERNAL_OBJECTS =

draw_trajectory: CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o
draw_trajectory: CMakeFiles/draw_trajectory.dir/build.make
draw_trajectory: /usr/local/lib/libpangolin.so
draw_trajectory: /data2/code/svo_workspace/Sophus/build/libSophus.so
draw_trajectory: /usr/lib/x86_64-linux-gnu/libGLU.so
draw_trajectory: /usr/lib/x86_64-linux-gnu/libGL.so
draw_trajectory: /usr/lib/x86_64-linux-gnu/libSM.so
draw_trajectory: /usr/lib/x86_64-linux-gnu/libICE.so
draw_trajectory: /usr/lib/x86_64-linux-gnu/libX11.so
draw_trajectory: /usr/lib/x86_64-linux-gnu/libXext.so
draw_trajectory: /usr/lib/x86_64-linux-gnu/libGLEW.so
draw_trajectory: /usr/lib/x86_64-linux-gnu/libSM.so
draw_trajectory: /usr/lib/x86_64-linux-gnu/libICE.so
draw_trajectory: /usr/lib/x86_64-linux-gnu/libX11.so
draw_trajectory: /usr/lib/x86_64-linux-gnu/libXext.so
draw_trajectory: /usr/lib/x86_64-linux-gnu/libGLEW.so
draw_trajectory: /usr/lib/x86_64-linux-gnu/libpython2.7.so
draw_trajectory: /usr/lib/x86_64-linux-gnu/libdc1394.so
draw_trajectory: /usr/lib/x86_64-linux-gnu/libavcodec.so
draw_trajectory: /usr/lib/x86_64-linux-gnu/libavformat.so
draw_trajectory: /usr/lib/x86_64-linux-gnu/libavutil.so
draw_trajectory: /usr/lib/x86_64-linux-gnu/libswscale.so
draw_trajectory: /opt/ros/indigo/lib/librealsense.so
draw_trajectory: /usr/lib/libOpenNI.so
draw_trajectory: /usr/lib/libOpenNI2.so
draw_trajectory: /usr/lib/x86_64-linux-gnu/libpng.so
draw_trajectory: /usr/lib/x86_64-linux-gnu/libz.so
draw_trajectory: /usr/lib/x86_64-linux-gnu/libjpeg.so
draw_trajectory: /usr/lib/x86_64-linux-gnu/libtiff.so
draw_trajectory: /usr/lib/x86_64-linux-gnu/libIlmImf.so
draw_trajectory: CMakeFiles/draw_trajectory.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/data2/code2/xl_slam/hw3/draw_trajectory/build2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable draw_trajectory"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/draw_trajectory.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/draw_trajectory.dir/build: draw_trajectory

.PHONY : CMakeFiles/draw_trajectory.dir/build

CMakeFiles/draw_trajectory.dir/requires: CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o.requires

.PHONY : CMakeFiles/draw_trajectory.dir/requires

CMakeFiles/draw_trajectory.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/draw_trajectory.dir/cmake_clean.cmake
.PHONY : CMakeFiles/draw_trajectory.dir/clean

CMakeFiles/draw_trajectory.dir/depend:
	cd /data2/code2/xl_slam/hw3/draw_trajectory/build2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data2/code2/xl_slam/hw3/draw_trajectory /data2/code2/xl_slam/hw3/draw_trajectory /data2/code2/xl_slam/hw3/draw_trajectory/build2 /data2/code2/xl_slam/hw3/draw_trajectory/build2 /data2/code2/xl_slam/hw3/draw_trajectory/build2/CMakeFiles/draw_trajectory.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/draw_trajectory.dir/depend

