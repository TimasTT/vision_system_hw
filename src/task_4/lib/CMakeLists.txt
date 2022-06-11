find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(framer SHARED frame_matching.cpp)

target_link_libraries(framer PRIVATE ${OpenCV_LIBS})