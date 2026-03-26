cmake_minimum_required(VERSION 3.10)

project(rknn_yolov5seg_demo)

# set(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# 从系统中查找ROS及必要的组件
find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    sensor_msgs
    cv_bridge
    image_transport
    message_filters
)

# 从系统中查找opencv
find_package(OpenCV REQUIRED)

# 从系统中查找PCL
find_package(PCL REQUIRED)

# 从3rdparty中查找opencv
# set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv/share/OpenCV)
# find_package(OpenCV REQUIRED)
# message(STATUS OpenCV_DIR=${OpenCV_DIR})
# message(STATUS OpenCV_LIBS=${OpenCV_LIBS})

# rga
set(RGA_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/librga)
set(LIBRGA ${RGA_PATH}/lib/librga.a)
set(RGA_INCLUDES ${RGA_PATH}/include)
install(PROGRAMS ${RGA_PATH}/lib/librga.so DESTINATION lib)

# ffmpeg
set(FFMPEG_LIBS_DIR /usr/local/ffmpeg/lib)
set(FFMPEG_HEADERS_DIR /usr/local/ffmpeg/include)

include_directories(${FFMPEG_HEADERS_DIR})
link_directories(${FFMPEG_LIBS_DIR})
set(FFMPEG_LIBS libavcodec.so libavformat.so libswscale.so libavdevice.so libavutil.so libswresample.so libavfilter.so)


# only RGA on rv1106 and rk3588 support handle
if (TARGET_SOC STREQUAL "rk3588")
    add_definitions(-DLIBRGA_IM2D_HANDLE)
endif()

# RGA2 only support under 4G memory
if (ENABLE_DMA32 STREQUAL "TRUE")
    add_definitions(-DDMA_ALLOC_DMA32)
endif()

# support rknn zero copy api
if (ENABLE_ZERO_COPY STREQUAL "TRUE")
    add_definitions(-DENABLE_ZERO_COPY)
endif()

# rknn runtime
set(RKNN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rknpu2)
set(LIBRKNNRT ${RKNN_PATH}/lib/librknnrt.so)
set(LIBRKNNRT_INCLUDES ${RKNN_PATH}/include)
install(PROGRAMS ${LIBRKNNRT} DESTINATION lib)
set(LIBRKNNRT ${LIBRKNNRT})

# dma
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/allocator/dma)

# drm
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/allocator/drm)

include_directories(
    ${catkin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${RGA_INCLUDES}
    ${LIBRKNNRT_INCLUDES}
    ${FFMPEG_HEADERS_DIR}
)

# rknn_yolov5_demo
include_directories( ${CMAKE_SOURCE_DIR}/include)

link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
link_directories(${FFMPEG_LIBS_DIR})

# yolov5seg_video_demo
add_executable(yolov5seg_video
    src/main.cc
    src/postprocess.cc
    src/yolov5_seg.cc
    src/image_utils.c
)

target_link_libraries(yolov5seg_video
    ${catkin_LIBRARIES}
    ${LIBRGA}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${FFMPEG_LIBS}
    ${LIBRKNNRT}
    -ldl
)

# install target and libraries
# set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov5seg_demo)
# install(TARGETS rknn_yolov5seg_demo DESTINATION .)
# install(TARGETS yolov5seg_video DESTINATION .)
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/bus.jpg DESTINATION model)
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/coco_80_labels_list.txt DESTINATION model)
# file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/model/*.rknn")
# install(FILES ${RKNN_FILES} DESTINATION model)