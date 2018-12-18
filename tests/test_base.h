/**
    @file
    @author  Alexander Sherikov
    @copyright

    @brief
*/

#pragma once

namespace
{
    template<class t_Objective>
    class GenericTestBase
    {
        public:
            sqp::Parameters parameters_;
            sqp::Solver solver_;
            t_Objective objective_;
            Eigen::VectorXd minimizer_;


        public:
            GenericTestBase()
            {
                SetUp();
            }

            void SetUp()
            {
                minimizer_.resize(objective_.getDimension());
            }

            void solve()
            {
                const bool trace = true;
                solver_.setParameters(parameters_);
                solver_.solve(minimizer_, objective_, trace);
            }
    };
}
