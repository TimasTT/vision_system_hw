#pragma once

#include <Eigen/Core>

#include <memory>
#include <utility>
#include <vector>

namespace eq {
    const int INIT_TIME       = 0;
    const double MAX_MIN_DIFF = 10e-10;
    const double DEFAULT_STEP = 0.1;

    const double TASK_DIFF_STEP = 0.001;
    const double MAX_TIME_RUNGE = 0.2;
    const double MAX_TIME_DP    = 10;

    struct task_answer {
        double answer_max_diff_runge{-1};
        double answer_step_runge{-1};
        double answer_min_step_dp{-1};
        int answer_total_steps_dp{-1};
    };

    class EquationSolver;

    class Solver {
    public:
        virtual ~Solver() = default;
        explicit Solver(EquationSolver &eq_solver, double step) : _solver(eq_solver), _step(step) {}

        Solver() = delete;
        Solver(const Solver&) = delete;
        void operator=(const Solver&) = delete;

        virtual void Calculate() = 0;
        virtual void Solver_set_matrix() = 0;

        Eigen::MatrixXd Generate_mat();

        double Get_step() const { return _step; }
        void Set_step(double step) {_step = step; }

    protected:
        EquationSolver &_solver;
        double _step;
    };

    class RungeSolver : public Solver {
    public:
        explicit RungeSolver(EquationSolver &eq_solver, double step = DEFAULT_STEP) : Solver(eq_solver, step) {}

        RungeSolver() = delete;
        RungeSolver(const RungeSolver&) = delete;
        void operator=(const RungeSolver&) = delete;

        void Calculate() override;
        void Solver_set_matrix() override;

    private:
        void calc_step();
    };

    class DPSolver : public Solver {
    public:
        explicit DPSolver(EquationSolver &eq_solver, double max_diff = MAX_MIN_DIFF, double min_diff = MAX_MIN_DIFF,
                          double step = DEFAULT_STEP) :
                          Solver(eq_solver, step), _max_diff(max_diff), _min_diff(min_diff) {}

        DPSolver() = delete;
        DPSolver(const DPSolver&) = delete;
        void operator=(const DPSolver&) = delete;

        void Calculate() override;
        void Solver_set_matrix() override;

    private:
        void calc_step();

    private:
        double _max_diff;
        double _min_diff;
    };

    class EquationSolver {
    public:
        explicit EquationSolver(std::function<Eigen::VectorXd(double, const Eigen::VectorXd &)> recalc_function,
                        Eigen::VectorXd start_vals, double start_time, std::function<Eigen::VectorXd(double)> result_function) :
                        _recalc_function(std::move(recalc_function)), _values(std::move(start_vals)), _time(start_time),
                        _result_function(std::move(result_function)) {}

        EquationSolver() = delete;
        EquationSolver(const EquationSolver&) = delete;
        void operator=(const EquationSolver&) = delete;

        void Set_solver(std::unique_ptr<Solver>&& solver);
        void Reload_values(const Eigen::VectorXd &init_vals, double init_time);

        void Solve();
        task_answer Get_answer() const;

        double Get_time() const { return _time; }
        Eigen::VectorXd& Get_values() { return _values; }

    public:
        friend Solver;
        friend RungeSolver;
        friend DPSolver;

    private:
        std::unique_ptr<Solver> _solver;

        std::function<Eigen::VectorXd(double, const Eigen::VectorXd &)> _recalc_function;
        Eigen::VectorXd _values;
        Eigen::VectorXd _steps;
        std::function<Eigen::VectorXd(double)> _result_function;

        Eigen::MatrixXd _a;
        std::vector<Eigen::MatrixXd> _b;

        double _time{0};

    private:
        double _answer_max_diff_runge{-1};
        double _answer_step_runge{-1};
        double _answer_min_step_dp{-1};
        int _answer_total_steps_dp{-1};
    };

    task_answer Calculate(std::function<Eigen::VectorXd(double, const Eigen::VectorXd &)> src_function,
                   std::function<Eigen::VectorXd(double)> result_function, const Eigen::VectorXd& init_val);
}
