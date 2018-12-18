/**
    @file
    @author  Alexander Sherikov
    @copyright

    @brief
*/

#pragma once

#include "objective.h"
#include <cppad/cppad.hpp>
#include <cppad/example/cppad_eigen.hpp>


namespace sqp
{
    template<class t_Scalar>
    class ObjectiveAutoDiffBase : public ObjectiveBase< t_Scalar >
    {
        protected:
            typedef CppAD::AD<t_Scalar>                t_ScalarAD;

        protected:
            virtual void evaluateFunction(const EigenVectorXX<t_ScalarAD> & at, t_ScalarAD &value) const = 0;


        public:
            using ObjectiveBase<t_Scalar>::getDimension;

            virtual void evaluateFunction(const EigenVectorXX<t_Scalar> & at, t_Scalar &value) const final
            {
                EigenVectorXX<t_ScalarAD> at_ad;
                at_ad.resize(at.size());
                for (std::size_t i = 0; i < at.size(); ++i)
                {
                    at_ad(i)= at(i);
                }

                EigenVectorXX<t_ScalarAD> value_ad;
                value_ad.resize(1);

                evaluateFunction(at_ad, value_ad(0));

                value = CppAD::Value(value_ad(0));
            }


            virtual void evaluate(
                    const EigenVectorXX<t_Scalar> & at,
                    EigenVectorXX<t_Scalar> &jacobian,
                    EigenMatrixXX<t_Scalar> &hessian) const final
            {
                SQP_ASSERT(at.size() == getDimension(), "Unexpected size of the minimizer vector.");

                EigenVectorXX<t_ScalarAD> at_ad;
                at_ad.resize(at.size());
                for (std::size_t i = 0; i < at.size(); ++i)
                {
                    at_ad(i)= at(i);
                }

                EigenVectorXX<t_ScalarAD> value_ad;
                value_ad.resize(1);

                CppAD::Independent(at_ad);
                evaluateFunction(at_ad, value_ad(0));

                CppAD::ADFun<t_Scalar> f(at_ad, value_ad);

                jacobian = f.Jacobian(at);
                SQP_ASSERT(at.size() == jacobian.rows(), "Wrong number of rows in the Jacobian.");


                Eigen::VectorXd hessian_vector;
                hessian_vector = f.Hessian(at, 0);
                SQP_ASSERT(hessian_vector.size() == at.size()*at.size(), "Wrong size of Hessian.");

                Eigen::Map<
                    Eigen::Matrix<  t_Scalar,
                                    Eigen::Dynamic,
                                    Eigen::Dynamic,
                                    Eigen::RowMajor> >  map(hessian_vector.data(),
                                                        at.size(),
                                                        at.size());
                hessian = map;
            }
    };
}
