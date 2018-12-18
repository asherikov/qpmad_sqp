/**
    @file
    @author  Alexander Sherikov

    @brief
*/

#include <sqp/solver.h>
#include <gtest/gtest.h>
#include "test_base.h"



namespace
{
    class GenericObjectiveAD : public sqp::ObjectiveAutoDiffBase<double>
    {
        public:
            std::size_t getDimension() const final
            {
                return 2;
            }

            void evaluateFunction(const sqp::EigenVectorXX<t_ScalarAD> & at, t_ScalarAD &value) const final
            {
                value = at(0)*at(0)*at(1)
                        + 0.3*at(1)*at(0)
                        + 0.1*at(0)*at(0)*at(0);
                value = value * value;
            }
    };


/*
import sympy
x0 = sympy.Symbol('x0');
x1 = sympy.Symbol('x1');
f = sympy.Matrix([pow(x0*x0*x1 + 0.3*x1*x0 + 0.1*x0*x0*x0, 2)])
x = sympy.Matrix([x0, x1])
J = f.jacobian(x)

sympy.hessian(f, x)
H =  sympy.simplify(sympy.hessian(f, x))
*/
    class GenericObjective : public sqp::ObjectiveAnalyticBase<double>
    {
        protected:
            void evaluateJacobian(const Eigen::VectorXd & at, Eigen::VectorXd &value) const
            {
                value.resize(getDimension());
                value <<
                    (0.6*at(0)*at(0) + 4*at(0)*at(1) + 0.6*at(1))*(0.1*at(0)*at(0)*at(0) + at(0)*at(0)*at(1) + 0.3*at(0)*at(1)), (2*at(0)*at(0) + 0.6*at(0))*(0.1*at(0)*at(0)*at(0) + at(0)*at(0)*at(1) + 0.3*at(0)*at(1));
            }

            void evaluateHessian(const Eigen::VectorXd & at, Eigen::MatrixXd &value) const
            {
                value.resize(getDimension(), getDimension());
                value <<
                    0.3*at(0)*at(0)*at(0)*at(0) + 4.0*at(0)*at(0)*at(0)*at(1) + 12.0*at(0)*at(0)*at(1)*at(1) + 0.72*at(0)*at(0)*at(1) + 3.6*at(0)*at(1)*at(1) + 0.18*at(1)*at(1),
                    at(0)*(1.0*at(0)*at(0)*at(0) + 8.0*at(0)*at(0)*at(1) + 0.24*at(0)*at(0) + 3.6*at(0)*at(1) + 0.36*at(1)),
                    at(0)*(1.0*at(0)*at(0)*at(0) + 8.0*at(0)*at(0)*at(1) + 0.24*at(0)*at(0) + 3.6*at(0)*at(1) + 0.36*at(1)),
                    at(0)*at(0)*(at(0) + 0.3)*(2*at(0) + 0.6);
            }


        public:
            std::size_t getDimension() const
            {
                return 2;
            }

            void evaluateFunction(const Eigen::VectorXd & at, double &value) const
            {
                value = at(0)*at(0)*at(1)
                        + 0.3*at(1)*at(0)
                        + 0.1*at(0)*at(0)*at(0);
                value = value * value;
            }
    };


    template<class t_Objective>
    class GenericTest : public ::testing::Test, public GenericTestBase<t_Objective>
    {
    };
}

using MyTypes = ::testing::Types<GenericObjectiveAD, GenericObjective>;
TYPED_TEST_CASE(GenericTest, MyTypes);


TYPED_TEST(GenericTest, test_ok)
{
    this->minimizer_ << 0.2, 1.0;
    this->parameters_.regularization_factor_ = 0.5;
    EXPECT_NO_THROW(this->solve());
}


TEST(GenericTest, Analytic_vs_AutoDiff)
{
    GenericTestBase<GenericObjectiveAD> autodiff;
    GenericTestBase<GenericObjective> analytic;

    autodiff.parameters_.regularization_factor_ = 0.5;
    analytic.parameters_.regularization_factor_ = 0.5;

    autodiff.minimizer_ << -0.4, 1.5;
    analytic.minimizer_ << -0.4, 1.5;

    analytic.solve();
    autodiff.solve();

    EXPECT_NEAR(autodiff.minimizer_[0], analytic.minimizer_[0], 1e-9);
    EXPECT_NEAR(autodiff.minimizer_[1], analytic.minimizer_[1], 1e-9);
}


int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
