#include "Solver.hpp"

#include <utility>

void eq::RungeSolver::calc_step() {
    Eigen::MatrixXd tmp = Generate_mat();

    Eigen::VectorXd x = (_solver._b[0] * tmp).transpose();

    _solver._values += x * _step;
    _solver._time += _step;
}

void eq::RungeSolver::Solver_set_matrix() {
    _solver._b.clear();
    Eigen::MatrixXd matrix(5, 5);
    matrix << 0, 0, 0, 0, 0, 1.0 / 2, 1.0 / 2, 0, 0, 0, 1.0 / 2, 0, 1.0 / 2, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 1.0 / 6, 1.0 / 3, 1.0 / 3, 1.0 / 6;
    _solver._b.emplace_back(matrix.block(matrix.rows() - 1, 1, 1, matrix.cols() - 1));
    _solver._a = matrix.block(0, 1, matrix.cols() - 1, matrix.cols() - 1);
    _solver._steps = matrix.block(0, 0, matrix.cols() - 1, 1);
}

void eq::RungeSolver::Calculate() {
    double max_diff = 0;
    while (_solver.Get_time() < MAX_TIME_RUNGE) {
        calc_step();
        double current_diff = (_solver.Get_values() - _solver._result_function(_solver.Get_time())).cwiseAbs().maxCoeff();

        if (current_diff > max_diff) max_diff = current_diff;

        if (current_diff > TASK_DIFF_STEP) {
            Set_step(Get_step() / 2);
            max_diff = 0;
            _solver.Get_values() = _solver._result_function(_solver.Get_time());
        }
    }
    _solver._answer_max_diff_runge = max_diff;
    _solver._answer_step_runge = Get_step();

}

void eq::DPSolver::calc_step() {
    Eigen::VectorXd x1;
    Eigen::VectorXd x2;
    Eigen::MatrixXd tmp;
    double diff;
    do {
        tmp = Generate_mat();

        x1 = (_solver._b[0] * tmp).transpose();
        x2 = (_solver._b[1] * tmp).transpose();

        diff = (x1 - x2).cwiseAbs().maxCoeff();
        if (diff > _max_diff) {
            _step /= 2;
        } else if (diff < _min_diff) {
            _step *= 2;
        }
    } while (diff > _max_diff);

    _solver._values += x1 * _step;
    _solver._time += _step;
}

void eq::DPSolver::Solver_set_matrix() {
    _solver._b.clear();
    Eigen::MatrixXd matrix(9, 8);
    matrix << 0, 0, 0, 0, 0, 0, 0, 0, 1.0 / 5, 1.0 / 5, 0, 0, 0, 0, 0, 0, 3.0 / 10, 3.0 / 40, 9.0 / 40, 0, 0, 0, 0, 0, 4.0 / 5, 44.0 / 45,
            -56.0 / 15, 32.0 / 9, 0, 0, 0, 0, 8.0 / 9, 19372.0 / 6561, -25360.0 / 2187, 64448.0 / 6561, -212.0 / 729, 0, 0, 0, 1, 9017.0 /
                                                                                                                                  3168,
            -355.0 / 33, 46732.0 / 5247, 49.0 / 176, -5103.0 / 18656, 0, 0, 1, 35.0 / 384, 0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784,
            11.0 / 84, 0, 0, 35.0 / 384, 0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84, 0, 0, 5179.0 / 57600, 0, 7571.0 / 16695,
            393.0 / 640, -92097.0 / 339200, 187.0 / 2100, 1.0 / 40;
    _solver._b.emplace_back(matrix.block(matrix.rows() - 2, 1, 1, matrix.cols() - 1));
    _solver._b.emplace_back(matrix.block(matrix.rows() - 1, 1, 1, matrix.cols() - 1));
    _solver._a = matrix.block(0, 1, matrix.cols() - 1, matrix.cols() - 1);
    _solver._steps = matrix.block(0, 0, matrix.cols() - 1, 1);
}

void eq::DPSolver::Calculate() {
    double min_step = DEFAULT_STEP;
    int total_steps = 0;
    while (_solver.Get_time() < MAX_TIME_DP) {
        ++total_steps;
        calc_step();

        if (Get_step() < min_step) min_step = Get_step();
    }

    _solver._answer_min_step_dp = min_step;
    _solver._answer_total_steps_dp = total_steps;
}

Eigen::MatrixXd eq::Solver::Generate_mat() {
    Eigen::MatrixXd tmp(_solver._a.cols(), _solver._values.size());
    auto val = _solver._values;

    tmp.row(0) = _solver._recalc_function(_solver._time, val);

    for (int i = 1; i < tmp.rows(); ++i) {
        val = _solver._values;
        for (int j = 0; j < i; ++j) {
            val += tmp.row(j) * _solver._a(i, j) * _step;
        }
        tmp.row(i) = _solver._recalc_function(_solver._time + _solver._steps(i) * _step, val);
    }

    return tmp;
}

void eq::EquationSolver::Set_solver(std::unique_ptr<Solver> &&solver) {
    _solver = std::move(solver);
    _solver->Solver_set_matrix();
}

void eq::EquationSolver::Reload_values(const Eigen::VectorXd &init_vals, double init_time) {
    _values = init_vals; _time = init_time;
}

void eq::EquationSolver::Solve() {
    _solver->Calculate();
}

eq::task_answer eq::EquationSolver::Get_answer() const {
    task_answer answer;
    answer.answer_step_runge = _answer_step_runge;
    answer.answer_max_diff_runge = _answer_max_diff_runge;
    answer.answer_total_steps_dp = _answer_total_steps_dp;
    answer.answer_min_step_dp = _answer_min_step_dp;
    return answer;
}

eq::task_answer eq::Calculate(std::function<Eigen::VectorXd(double, const Eigen::VectorXd &)> src_function,
                   std::function<Eigen::VectorXd(double)> result_function, const Eigen::VectorXd& init_val) {
    EquationSolver eq_solver(std::move(src_function), init_val, INIT_TIME, std::move(result_function));

    std::unique_ptr<Solver> runge_solver = std::make_unique<eq::RungeSolver>(eq_solver);
    eq_solver.Set_solver(std::move(runge_solver));
    eq_solver.Solve();

    std::unique_ptr<eq::Solver> dp_solver = std::make_unique<eq::DPSolver>(eq_solver);
    eq_solver.Set_solver(std::move(dp_solver));
    eq_solver.Reload_values(init_val, INIT_TIME);
    eq_solver.Solve();

    return eq_solver.Get_answer();
}
