cmake_minimum_required(VERSION 3.1)

project(stereo_sgm_cuda)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CUDA_ARCH "-arch=sm_50" CACHE STRING "Value of the NVCC -arch option.")
if(${CMAKE_VERSION} VERSION_EQUAL 3.7 OR ${CMAKE_VERSION} VERSION_GREATER 3.7)
  option(AUTO_DETECT_ARCH "Detect local GPU compute arch automatically" ON)
endif()

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

add_subdirectory(src)

file(GLOB SOURCES "*.cpp")

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

add_executable(stereo_sgm_cuda ${SOURCES})

target_link_libraries(stereo_sgm_cuda libsgm_cuda ${OpenCL_LIBRARIES} ${OpenCV_LIBS} pthread udev v4l2)
