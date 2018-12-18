/**
    @file
    @author  Alexander Sherikov
    @copyright

    @brief
*/

#pragma once

#include <iostream>

#include "solver_parameters.h"
#include "objective.h"
#include "objective_ad.h"
#include <qpmad/solver.h>


namespace sqp
{
    class Solver
    {
        protected:
            Parameters parameters_;

            Eigen::VectorXd minimizer_increment_;

            double value_;
            Eigen::VectorXd jacobian_;
            Eigen::MatrixXd hessian_;


            qpmad::Solver qp_solver_;
            Eigen::VectorXd lb_;
            Eigen::VectorXd ub_;
            double trust_region_size_;


        public:
            Solver()
            {
            }


        public:
            void setParameters(const Parameters &param)
            {
                param.validate();
                parameters_ = param;
            }


        public:
            template<class t_Scalar, class t_ScalarAD>
            void solve(EigenVectorXX<t_Scalar> & minimizer, const ObjectiveBase<t_ScalarAD> & objective, const bool trace = false)
            {
                const std::size_t minimizer_size = objective.getDimension();
                if (minimizer.size() != minimizer_size)
                {
                    minimizer.setZero(minimizer_size);
                }
                minimizer_increment_.resize(minimizer_size);


                objective.evaluateFunction(minimizer, value_);

                trust_region_size_ = parameters_.initial_trust_region_size_;
                std::size_t iter = 0;
                for(; iter < parameters_.max_interations_; ++iter)
                {
                    if (true == trace)
                    {
                        std::cout << "-----------------------" << std::endl;
                        std::cout << "iteration: " << iter << std::endl;
                    }

                    objective.evaluate(minimizer, jacobian_, hessian_);


                    lb_.setConstant(minimizer_size, -trust_region_size_);
                    ub_.setConstant(minimizer_size, trust_region_size_);


                    if (parameters_.regularization_factor_ > 0)
                    {
                        for (std::ptrdiff_t i = 0; i < hessian_.rows(); ++i)
                        {
                            hessian_(i,i) += parameters_.regularization_factor_;
                        }
                    }

                    if (true == trace)
                    {
                        std::cout << "hessian: \n" << hessian_ << std::endl;
                        std::cout << "jacobian: " << jacobian_ << std::endl;
                        std::cout << "lb: " << lb_ << std::endl;
                        std::cout << "ub: " << ub_ << std::endl;
                        std::cout << "minimizer: " << minimizer.transpose() << std::endl;
                    }
                    qpmad::Solver::ReturnStatus qp_result = qp_solver_.solve(minimizer_increment_, hessian_, jacobian_, lb_, ub_);
                    if (true == trace)
                    {
                        std::cout << "minimizer_increment: " << minimizer_increment_.transpose() << std::endl;
                    }

                    SQP_ASSERT(qpmad::Solver::OK == qp_result, "QP solver failed.");
                    for (std::size_t i = 0; i < minimizer_size; ++i)
                    {
                        SQP_ASSERT(false == std::isnan(minimizer_increment_(i)), "NaN in the solution, negative definite Hessian?");
                    }


                    double new_value;
                    objective.evaluateFunction(minimizer + minimizer_increment_, new_value);
                    if (true == trace)
                    {
                        std::cout << "value: " << value_ << std::endl;
                        std::cout << "new_value: " << new_value << std::endl;
                    }

                    if (std::abs(value_ - new_value) < parameters_.value_tolerance_)
                    {
                        break;
                    }

                    if (new_value < value_)
                    {
                        // expand if good
                        trust_region_size_ *= parameters_.trust_region_expand_factor_;
                        minimizer += minimizer_increment_;

                        value_ = new_value;
                    }
                    else
                    {
                        // shrink if bad
                        trust_region_size_ *= parameters_.trust_region_shrink_factor_;
                        SQP_ASSERT(trust_region_size_*trust_region_size_ > parameters_.value_tolerance_,
                                "Trust region is too small. Terminating.");
                    }

                    if (true == trace)
                    {
                        std::cout << "-----------------------" << std::endl;
                    }
                }

                SQP_ASSERT(iter < parameters_.max_interations_, "Maximal number of iterations exceeded.");
            }
    };
}
