cmake_minimum_required(VERSION 3.20)
project(vision_system_hw_task_4)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(lib)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC framer)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_HOME_DIRECTORY}/lib)