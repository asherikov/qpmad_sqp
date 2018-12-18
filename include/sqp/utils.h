/**
    @file
    @author  Alexander Sherikov
    @copyright

    @brief
*/

#pragma once

#include <stdexcept>
#include <string>

#define SQP_THROW_MSG(s) throw std::runtime_error(std::string("In ") + __func__ + "() // " + (s))
#define SQP_ASSERT(condition, message) if (!(condition)) {SQP_THROW_MSG(message);};
