//
// Created by timas on 27.03.2022.
//

#pragma once

#include <iostream>
#include <vector>

class Matrix {
public:
    Matrix() = delete;

    ~Matrix() = default;

    Matrix(size_t rows, size_t cols);

    explicit Matrix(const std::vector <std::vector<double>> &mat);

    Matrix(const Matrix &mat) = default;

    Matrix &operator=(const Matrix &mat) = default;

    double operator()(size_t i, size_t j) const;

    double &operator()(size_t i, size_t j);

    size_t GetCols() const;

    size_t GetRaws() const;

    std::vector <std::vector<double>> GetMatrix() const;

    bool operator==(const Matrix& mat) const;

    bool operator!=(const Matrix& mat) const;

    std::pair<Matrix, Matrix> LU() const;

    void Inverse();

private:
    friend std::ostream &operator<<(std::ostream &os, const Matrix &matrix);

    std::vector <std::vector<double>> data;
};

/**
 * @brief solve A * x = B
 * @param A Matrix A
 * @param B Matrix B
 * @return x
 */
Matrix Solve(const Matrix &A, const Matrix &B);
