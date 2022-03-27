//
// Created by timas on 27.03.2022.
//

#include "Matrix.hpp"

static const char* BAD_INPUT_PARAMETERS = "Bad Input Parameters";
static const char* BAD_SIZE_PARAMETERS = "Bad Size Parameters";

static const double EPS = 1e-7;

Matrix::Matrix(size_t raws, size_t cols) {
    if (raws == 0 || cols == 0) throw std::runtime_error(BAD_INPUT_PARAMETERS);
    data.resize(raws);
    for (auto& raw : data) {
        raw.resize(cols);
    }
}

Matrix::Matrix(const std::vector<std::vector<double>> &mat) {
    if (mat.empty() || mat.data()->empty()) throw std::runtime_error(BAD_INPUT_PARAMETERS);
    data = mat;
}

double Matrix::operator()(size_t i, size_t j) const {
    if (i >= GetRaws() || j >= GetCols()) throw std::runtime_error(BAD_SIZE_PARAMETERS);
    return data[i][j];
}

double &Matrix::operator()(size_t i, size_t j) {
    if (i >= GetRaws() || j >= GetCols()) throw std::runtime_error(BAD_SIZE_PARAMETERS);
    return data[i][j];

}

size_t Matrix::GetCols() const {
    return data.data()->size();
}

size_t Matrix::GetRaws() const {
    return data.size();
}

std::vector<std::vector<double>> Matrix::GetMatrix() const {
    return data;
}

bool Matrix::operator==(const Matrix& mat) const {
    auto raws = GetRaws(); auto cols = GetCols();
    if (raws != mat.GetRaws() || cols != mat.GetCols()) throw std::runtime_error("not correct sizes of matrix");
    for (size_t i = 0; i < raws; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (std::abs(mat(i, j) - (*this)(i, j)) > EPS) {
                return false;
            }
        }
    }
    return true;
}

bool Matrix::operator!=(const Matrix& mat) const { return !(*this == mat); }

std::pair<Matrix, Matrix> Matrix::LU() const {
    auto raws = GetRaws(); auto cols = GetCols();
    Matrix L(raws, cols); Matrix U(raws, cols);
    for (size_t i = 0; i < raws; ++i) {
        for (size_t j = 0; j < cols; j++) {
            if (j < i) {
                U(i, j) = 0;
            } else {
                U(i, j) = (*this)(i, j);
                for (size_t k = 0; k < i; k++) {
                    U(i, j) -= L(i, k) * U(k, j);
                }
            }
        }
        for (size_t j = 0; j < cols; ++j) {
            if (j < i) {
                L(j, i) = 0;
            } else if (j == i) {
                L(i, j) = 1;
            } else {
                L(j, i) = (*this)(j, i) / U(i, i);
                for (size_t k = 0; k < i; ++k) {
                    L(j, i) -= L(j, k) * U(k, i) / U(i, i);
                }
            }
        }
    }
    return {L, U};
}

void Matrix::Inverse() {
    auto raws = GetRaws(); auto cols = GetCols();
    Matrix invMat(raws, cols);
    for (size_t i = 0; i < raws; ++i) {
        invMat(i, i) = 1;
    }
    auto LU = (*this).LU();
    for (size_t j = 0; j < raws; j++) {
        for (size_t i = 0; i < cols; i++) {
            for (size_t k = 0; k < i; k++) {
                invMat(i, j) -= LU.first(i, k) * invMat(k, j);
            }
            invMat(i, j) /= LU.first(i, i);
        }
        for (int i = raws - 1; i >= 0; i--) {
            for (size_t k = i + 1; k < raws; k++) {
                invMat(i, j) -= LU.second(i, k) * invMat(k, j);
            }

            invMat(i, j) /= LU.second(i, i);
        }
    }
    *this = invMat;
}

std::ostream &operator<<(std::ostream &os, const Matrix &matrix) {
    size_t cols = matrix.GetCols(); size_t raws = matrix.GetRaws();
    for (size_t i = 0; i < raws; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            os << matrix(i, j);
            if (j == cols - 1) {
                os << "\n";
            } else {
                os << " ";
            }
        }
    }
    return os;
}

static Matrix forward(const Matrix& L, const Matrix& b) {
    Matrix y(L.GetRaws(), 1);
    for (size_t i = 0; i < L.GetRaws(); ++i) {
        y(i, 0) = b(i, 0);
        for (size_t j = 0; j < i; ++j) {
            y(i, 0) -= L(i, j) * y(j, 0);
        }
        y(i, 0) /= L(i, i);
    }
    return y;
}

static Matrix back(const Matrix& U, const Matrix& y) {
    Matrix x(U.GetRaws(), 1);
    for (int i = U.GetRaws() - 1; i >= 0; --i) {
        x(i, 0) = y(i, 0);
        for (size_t j = i + 1; j < U.GetRaws(); ++j) {
            x(i, 0) -= U(i, j) * x(j, 0);
        }
        x(i, 0) /= U(i, i);
    }
    return x;
}

Matrix Solve(const Matrix &A, const Matrix &B) {
    auto LU = A.LU();
    Matrix y = forward(LU.first, B);
    Matrix x = back(LU.second, y);
    return x;
}