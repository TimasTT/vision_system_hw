cmake_minimum_required(VERSION 3.20)
project(vision_system_hw_task_2)

set(CMAKE_CXX_STANDARD 14)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(${PROJECT_NAME} least_squares.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)