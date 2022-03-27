add_executable(matrix_test matrix_test.cpp)

target_link_libraries(matrix_test gtest matrix)
target_include_directories(matrix_test PUBLIC ${CMAKE_HOME_DIRECTORY}/lib)