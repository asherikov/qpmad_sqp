/**
    @file
    @author  Alexander Sherikov
    @copyright

    @brief
*/

#pragma once

#include "utils.h"
#include <Eigen/Core>

namespace sqp
{
    template<class t_Scalar>
    using EigenVectorXX = Eigen::Matrix<t_Scalar, Eigen::Dynamic, 1>;

    template<class t_Scalar>
    using EigenMatrixXX = Eigen::Matrix<t_Scalar, Eigen::Dynamic, Eigen::Dynamic>;


    template <class t_Scalar>
    class ObjectiveBase
    {
        public:
            virtual std::size_t getDimension() const = 0;

            virtual void evaluateFunction(const EigenVectorXX<t_Scalar> & at, t_Scalar &value) const = 0;

            virtual void evaluate(
                    const EigenVectorXX<t_Scalar> & at,
                    EigenVectorXX<t_Scalar> &jacobian,
                    EigenMatrixXX<t_Scalar> &hessian) const = 0;
    };


    template <class t_Scalar>
    class ObjectiveAnalyticBase : public ObjectiveBase<t_Scalar>
    {
        protected:
            virtual void evaluateJacobian(const EigenVectorXX<t_Scalar> & at, EigenVectorXX<t_Scalar> &value) const = 0;
            virtual void evaluateHessian(const EigenVectorXX<t_Scalar> & at, EigenMatrixXX<t_Scalar> &value) const = 0;

        public:
            using ObjectiveBase<t_Scalar>::getDimension;
            using ObjectiveBase<t_Scalar>::evaluateFunction;

            virtual void evaluate(
                    const EigenVectorXX<t_Scalar> & at,
                    EigenVectorXX<t_Scalar> &jacobian,
                    EigenMatrixXX<t_Scalar> &hessian) const
            {
                SQP_ASSERT(at.size() == getDimension(), "Unexpected size of the minimizer vector.");

                evaluateJacobian(at, jacobian);
                evaluateHessian(at, hessian);

                SQP_ASSERT(at.size() == jacobian.rows(), "Wrong number of rows in the Jacobian.");
                SQP_ASSERT(hessian.rows() == at.size(), "Wrong number of rows in the Hessian.");
                SQP_ASSERT(hessian.cols() == at.size(), "Wrong number of columns in the Hessian.");
            }
    };
}
