find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_library(solver Solver.cpp)

target_link_libraries(solver PRIVATE Eigen3::Eigen)