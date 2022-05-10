#include <fstream>
#include <vector>
#include <iostream>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>

template<typename M>
M load_csv(const std::string &path) {
    std::ifstream file(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ' ')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Eigen::Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, Eigen::RowMajor>>(
            values.data(), rows, values.size() / rows);
}

template<typename M>
static void show_lu_solution(M& data) {
    Eigen::MatrixXd Z = data.col(0);
    for (long i = 0; i < Z.rows(); ++i) {
        Z(i, 0) = std::log(data(i, 0));
    }
    Eigen::MatrixXd T(data.rows(), 2);

    for (long i = 0; i < T.rows(); ++i) {
        T(i, 0) = data(i, 1);
        T(i, 1) = 1;
    }

    auto T_2 = T.transpose() * T;
    auto T_Z = T.transpose() * Z;

    std::cout << T_2.lu().solve(T_Z) << std::endl;
}

int main() {
    const std::string FILEPATH("../temp.csv");
    auto data = load_csv<Eigen::MatrixXd>(FILEPATH);

    show_lu_solution(data);

    return 0;
}
