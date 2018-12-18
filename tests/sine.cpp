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
    class SineObjectiveAD : public sqp::ObjectiveAutoDiffBase<double>
    {
        public:
            std::size_t getDimension() const final
            {
                return 1;
            }

            void evaluateFunction(const sqp::EigenVectorXX<t_ScalarAD> & at, t_ScalarAD &value) const final
            {
                value = CppAD::sin(at(0));
            }
    };


    class SineObjective : public sqp::ObjectiveAnalyticBase<double>
    {
        protected:
            void evaluateJacobian(const Eigen::VectorXd & at, Eigen::VectorXd &value) const
            {
                value.resize(1);
                value(0) = std::cos(at(0));
            }

            void evaluateHessian(const Eigen::VectorXd & at, Eigen::MatrixXd &value) const
            {
                value.resize(1, 1);
                value(0,0) = -std::sin(at(0));
            }


        public:
            std::size_t getDimension() const
            {
                return 1;
            }

            void evaluateFunction(const Eigen::VectorXd & at, double &value) const
            {
                value = std::sin(at(0));
            }
    };


    template<class t_Objective>
    class SineTest : public ::testing::Test, public GenericTestBase<t_Objective>
    {
    };
}

using MyTypes = ::testing::Types<SineObjectiveAD, SineObjective>;
TYPED_TEST_CASE(SineTest, MyTypes);


TYPED_TEST(SineTest, test_fail)
{
    this->minimizer_ << 0.5;
    EXPECT_THROW(this->solve(), std::runtime_error);
}


TYPED_TEST(SineTest, test_ok_regularization)
{
    this->minimizer_ << 0.5;
    this->parameters_.regularization_factor_ = 3.0;
    EXPECT_NO_THROW(this->solve());
}


TYPED_TEST(SineTest, test_ok)
{
    this->minimizer_ << 3.2;
    EXPECT_NO_THROW(this->solve());
}


TEST(SineTest, Analytic_vs_AutoDiff)
{
    GenericTestBase<SineObjectiveAD> autodiff;
    GenericTestBase<SineObjective> analytic;

    autodiff.minimizer_ << 3.3;
    analytic.minimizer_ << 3.3;

    analytic.solve();
    autodiff.solve();

    EXPECT_NEAR(autodiff.minimizer_[0], analytic.minimizer_[0], 1e-9);
}


int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
