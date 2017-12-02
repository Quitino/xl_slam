# 声明要求的 cmake 最低版本
cmake_minimum_required( VERSION 2.8 )

# 声明一个 cmake 工程
project( hello )

# 设置编译模式
set( CMAKE_BUILD_TYPE "Release" )

set( CMAKE_INSTALL_PREFIX "/usr/local")

include_directories( ${PROJECT_SOURCE_DIR}/include )

# 添加一个共享库
add_library( hello SHARED src/hello.cpp )

# 添加一个可执行程序
add_executable( sayhello src/useHello.cpp )
# 将库文件链接到可执行程序上
target_link_libraries( sayhello hello )


INSTALL(FILES include/hello.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include )
INSTALL(TARGETS hello DESTINATION ${CMAKE_INSTALL_PREFIX}/lib )
