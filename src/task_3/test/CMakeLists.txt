find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(solver_test solver_test.cpp)

target_link_libraries(solver_test gtest solver Eigen3::Eigen)
target_include_directories(solver_test PUBLIC ${CMAKE_HOME_DIRECTORY}/lib)