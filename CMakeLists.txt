cmake_minimum_required(VERSION 3.9)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

project(stereo_fusion)

# Options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs CACHE PATH "Path to libs folder")
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src CACHE PATH "Path to src folder")
set(OpenCV_DIR /opt/homebrew/Cellar/opencv/4.8.1_5/lib/cmake/opencv4 CACHE PATH "Path to OpenCV folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Open3D REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

include(FetchContent)

FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG master # Can be a tag (yaml-cpp-x.x.x), a commit hash, or a branch name (master)
)
FetchContent_GetProperties(yaml-cpp)

if(NOT yaml-cpp_POPULATED)
    message(STATUS "Fetching yaml-cpp...")
    FetchContent_Populate(yaml-cpp)
    add_subdirectory(${yaml-cpp_SOURCE_DIR} ${yaml-cpp_BINARY_DIR})
endif()

FetchContent_Declare(matplotplusplus
        GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
        GIT_TAG origin/master) # or whatever tag you want

FetchContent_GetProperties(matplotplusplus)
if(NOT matplotplusplus_POPULATED)
    message(STATUS "Fetching matplot++...")
    FetchContent_Populate(matplotplusplus)
    add_subdirectory(${matplotplusplus_SOURCE_DIR} ${matplotplusplus_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()


set(HEADER_FILES 
    ${SOURCE_DIR}/StereoImage/StereoImage.h
    ${SOURCE_DIR}/SparseMatching/SparseMatching.h
    ${SOURCE_DIR}/StereoRectification/StereoRectification.h

    ${SOURCE_DIR}/OpenCVTools/OpenCVHelpers.h
    ${SOURCE_DIR}/SFTools/SFSparseMatching.h
    ${SOURCE_DIR}/SFTools/SFRectification.h

    ${SOURCE_DIR}/ReconstructionUtils/ReconstructionUtils.h
    ${SOURCE_DIR}/ReconstructionUtils/Eigen.h
    ${SOURCE_DIR}/SFTools/ProgressBar.h
)

set(SOURCE_FILES 
    ${SOURCE_DIR}/StereoImage/StereoImage.cpp
    ${SOURCE_DIR}/SparseMatching/SparseMatching.cpp
    ${SOURCE_DIR}/StereoRectification/StereoRectification.cpp

    ${SOURCE_DIR}/OpenCVTools/OpenCVHelpers.cpp
    ${SOURCE_DIR}/SFTools/SFSparseMatching.cpp
    ${SOURCE_DIR}/SFTools/SFRectification.cpp

    ${SOURCE_DIR}/ReconstructionUtils/ReconstructionUtils.cpp
)
add_executable(${PROJECT_NAME} ${SOURCE_DIR}/stereo_pipeline.cpp ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen ${OpenCV_LIBS} ${Open3D_LIBRARIES} yaml-cpp::yaml-cpp matplot)
 