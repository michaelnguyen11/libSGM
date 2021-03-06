cmake_minimum_required(VERSION 3.1)

set(LIB libsgm_cuda)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH}")

file(GLOB STEREOSRCS "*.cu" "*.cpp")

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

CUDA_ADD_LIBRARY(${LIB} ${STEREOSRCS} STATIC)

target_include_directories(${LIB} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_link_libraries(${LIB} ${OpenCV_LIBS})
