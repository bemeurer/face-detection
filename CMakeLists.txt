cmake_minimum_required(VERSION 3.5)
project(face_detection)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

set(SOURCE_FILES main.cpp)
add_executable(face_detection ${SOURCE_FILES})
target_link_libraries(face_detection ${OpenCV_LIBS})