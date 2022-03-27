//
// Created by timas on 27.03.2022.
//

#include "Matrix.hpp"

#include <gtest/gtest.h>

TEST(MatrixTests, testSolve) { // A * x = B
    Matrix A({{10, 6, 2, 0}, {5, 1, -2, 4}, {3, 5, 1, -1}, {0, 6, -2, 2}});
    Matrix B({{25}, {14}, {10}, {8}});
    Matrix expectedX({{2}, {1}, {-0.5}, {0.5}});
    Matrix x = Solve(A, B);

    ASSERT_EQ(x, expectedX);
}

TEST(matrix, inverse) { // Inverse matrix
    Matrix A({{10, 6, 2, 0}, {5, 1, -2, 4}, {3, 5, 1, -1}, {0, 6, -2, 2}});
    Matrix expectedInv({{-8. / 11, 8. / 11, 17. / 11, -15. / 22},
                             {3. / 11, -3. / 11, -5. / 11, 7. / 22},
                             {73. / 22, -31. / 11, -70. / 11, 27. / 11},
                             {5. / 2, -2., -5., 2.}});

    A.Inverse();
    ASSERT_EQ(Matrix(A.GetMatrix()), expectedInv);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}