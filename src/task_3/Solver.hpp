#pragma once

#include <Eigen/Core>

#include <memory>
#include <utility>
#include <vector>

namespace eq {
    const int INIT_TIME = 0;

    class EquationSolver;

    class Solver {
    public:
        virtual ~Solver() = default;
        explicit Solver(EquationSolver &eq_solver, double step) : _solver(eq_solver), _step(step) {}

        Solver() = delete;
        Solver(const Solver&) = delete;
        void operator=(const Solver&) = delete;

        virtual void Calc_step() = 0;
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
        explicit RungeSolver(EquationSolver &eq_solver, double step = 0.1) : Solver(eq_solver, step) {}

        RungeSolver() = delete;
        RungeSolver(const RungeSolver&) = delete;
        void operator=(const RungeSolver&) = delete;

        void Calc_step() override;
        void Solver_set_matrix() override;
    };

    class DPSolver : public Solver {
    public:
        explicit DPSolver(EquationSolver &eq_solver, double max_diff = 10e-10, double min_diff = 10e-10, double step = 0.1) :
                          Solver(eq_solver, step), _max_diff(max_diff), _min_diff(min_diff) {}

        DPSolver() = delete;
        DPSolver(const DPSolver&) = delete;
        void operator=(const DPSolver&) = delete;

        void Calc_step() override;
        void Solver_set_matrix() override;

    private:
        double _max_diff;
        double _min_diff;
    };

    class EquationSolver {
    public:
        explicit EquationSolver(std::function<Eigen::VectorXd(double, const Eigen::VectorXd &)> recalc_function,
                        Eigen::VectorXd start_vals, double start_time): _recalc_function(std::move(recalc_function)),
                                                                               _values(std::move(start_vals)), _time(start_time) {}

        EquationSolver() = delete;
        EquationSolver(const EquationSolver&) = delete;
        void operator=(const EquationSolver&) = delete;

        void Set_solver(std::unique_ptr<Solver>&& solver);
        void Reload_values(const Eigen::VectorXd &init_vals, double init_time);

        void Calc_step();
        double Get_step() const;
        void Set_step(double step);

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

        Eigen::MatrixXd _a;
        std::vector<Eigen::MatrixXd> _b;

        double _time{0};
    };

    void Calculate(std::function<Eigen::VectorXd(double, const Eigen::VectorXd &)> src_function,
                   std::function<Eigen::VectorXd(double)> result_function, const Eigen::VectorXd& init_val);
}
