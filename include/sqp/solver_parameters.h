/**
    @file
    @author  Alexander Sherikov
    @copyright

    @brief
*/

#pragma once

#include "utils.h"

namespace sqp
{
    class Parameters
    {
        public:
            double regularization_factor_;
            double initial_trust_region_size_;
            double trust_region_expand_factor_;
            double trust_region_shrink_factor_;
            double value_tolerance_;

            std::size_t max_interations_;


        public:
            Parameters()
            {
                setDefaults();
            }


            void setDefaults()
            {
                value_tolerance_ = 1e-8;

                regularization_factor_ = 1e-10;
                initial_trust_region_size_ = 1.0;

                trust_region_expand_factor_ = 2.0;
                trust_region_shrink_factor_ = 0.5;

                max_interations_ = 50000;
            }


            void validate() const
            {
                SQP_ASSERT(value_tolerance_ > 0.0, "Value tolerance must be positive.");

                SQP_ASSERT(regularization_factor_ >= 0.0, "Regularization factor must be nonnegative.");
                SQP_ASSERT(initial_trust_region_size_ > 0.0, "Initial trust region size must be positive.");

                SQP_ASSERT(trust_region_expand_factor_ > 1.0, "Expand factor must be > 1.0.");
                SQP_ASSERT(trust_region_shrink_factor_ < 1.0, "Shrink factor must be < 1.0.");
            }
    };
}
