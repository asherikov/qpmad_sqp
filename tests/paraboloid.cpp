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
    class ParaboloidObjectiveAD : public sqp::ObjectiveAutoDiffBase<double>
    {
        public:
            std::size_t getDimension() const final
            {
                return 2;
            }

            void evaluateFunction(const sqp::EigenVectorXX<t_ScalarAD> & at, t_ScalarAD &value) const final
            {
                value = at(0)*at(0) + at(1)*at(1);
            }
    };


    class ParaboloidObjective : public sqp::ObjectiveAnalyticBase<double>
    {
        protected:
            void evaluateJacobian(const Eigen::VectorXd & at, Eigen::VectorXd &value) const
            {
                value.resize(getDimension());
                value << 2*at(0), 2*at(1);
            }

            void evaluateHessian(const Eigen::VectorXd & at, Eigen::MatrixXd &value) const
            {
                value.resize(getDimension(), getDimension());
                value << 2.0, 0.0,
                         0.0, 2.0;
            }


        public:
            std::size_t getDimension() const
            {
                return 2;
            }

            void evaluateFunction(const Eigen::VectorXd & at, double &value) const
            {
                value = at(0)*at(0) + at(1)*at(1);
            }
    };


    template<class t_Objective>
    class ParaboloidTest : public ::testing::Test, public GenericTestBase<t_Objective>
    {
    };
}

using MyTypes = ::testing::Types<ParaboloidObjectiveAD, ParaboloidObjective>;
TYPED_TEST_CASE(ParaboloidTest, MyTypes);


TYPED_TEST(ParaboloidTest, test_ok)
{
    this->minimizer_ << 0.5, 21.0;
    EXPECT_NO_THROW(this->solve());
}


TEST(ParaboloidTest, Analytic_vs_AutoDiff)
{
    GenericTestBase<ParaboloidObjectiveAD> autodiff;
    GenericTestBase<ParaboloidObjective> analytic;

    autodiff.minimizer_ << 1003.3, 200.3;
    analytic.minimizer_ << 1003.3, 200.3;

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
