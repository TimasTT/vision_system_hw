#include "Solver.hpp"

#include <gtest/gtest.h>

const double a = -0.7;
const double b = 0.7;
const double d = 1;

Eigen::VectorXd example1(double time, const Eigen::VectorXd &val) {
    Eigen::VectorXd ret(1);
    ret(0) = a * time - b * val(0, 0);
    return ret;
}

Eigen::VectorXd result1(double time) {
    Eigen::VectorXd result(1);
    result(0) = a / b * (time - 1 / b) + 0.57 * exp(-b * time);
    return result;
}

TEST(SolverTests, equation1) {
    Eigen::VectorXd init_value(1, 1);
    init_value(0, 0) = d;
    auto answer = eq::Calculate(example1, result1, init_value);

    std::cout << "min_step (DP) : " << answer.answer_min_step_dp << std::endl;
    std::cout << "total_steps (DP) : " << answer.answer_total_steps_dp << std::endl;
    std::cout << "max_diff (Runge) : " << answer.answer_max_diff_runge << std::endl;
    std::cout << "step (Runge) : " << answer.answer_step_runge << std::endl << std::endl;

    EXPECT_TRUE(answer.answer_min_step_dp != -1);
    EXPECT_TRUE(answer.answer_total_steps_dp != -1);
    EXPECT_TRUE(answer.answer_max_diff_runge!= -1);
    EXPECT_TRUE(answer.answer_step_runge != -1);
}

Eigen::VectorXd example2(double time, const Eigen::VectorXd &val) {
    Eigen::VectorXd ret(2, 1);
    ret(0, 0) = 9 * val(0, 0) + 24 * val(1, 0) + 5 * cos(time) - 1.0 / 3 * sin(time);
    ret(1, 0) = -24 * val(0, 0) - 51 * val(1, 0) + 9 * cos(time) + 1.0 / 3 * sin(time);
    return ret;
}

Eigen::VectorXd result2(double time) {
    Eigen::VectorXd result(2);
    result(0) = 2.0 * exp(-3 * time) - exp(-39 * time) + 1.0 / 3 * cos(time);
    result(1) = - exp(-3 * time) + 2.0 * exp(-39 * time) - 1.0 / 3 * cos(time);
    return result;
}

TEST(SolverTests, equation2) {
    Eigen::VectorXd init_value(2, 1);
    init_value(0, 0) = 4.0 / 3;
    init_value(1, 0) = 2.0 / 3;
    auto answer = eq::Calculate(example2, result2, init_value);

    std::cout << "min_step (DP) : " << answer.answer_min_step_dp << std::endl;
    std::cout << "total_steps (DP) : " << answer.answer_total_steps_dp << std::endl;
    std::cout << "max_diff (Runge) : " << answer.answer_max_diff_runge << std::endl;
    std::cout << "step (Runge) : " << answer.answer_step_runge << std::endl;

    EXPECT_TRUE(answer.answer_min_step_dp != -1);
    EXPECT_TRUE(answer.answer_total_steps_dp != -1);
    EXPECT_TRUE(answer.answer_max_diff_runge!= -1);
    EXPECT_TRUE(answer.answer_step_runge != -1);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}