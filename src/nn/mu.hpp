/*
 * Copyright 2015-2016 Nicholas Andrews
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifndef __NN_MATH_UTIL_HPP__
#define __NN_MATH_UTIL_HPP__

#include <limits>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <cfloat>
#include <stdexcept>

namespace nn {

    const int MATH_MATRIX_ROW = 0;
    const int MATH_MATRIX_COL = 1;
    const double NEG_INF = -std::numeric_limits<double>::infinity();
    const double POS_INF = std::numeric_limits<double>::infinity();
    const double a0 = std::log(2);

    // Compute log2(x) -- the log2 provided by GCC is somewhat weird.
    inline double log2(double x) {
        static const double LOG2 = std::log(2.);
        return std::log(x)/LOG2;
    }

    // Numerically stable way of computing log(e^{a_1}+...e^{a_n})
    inline double logsumexp(double nums[], size_t ct) {
        double max_exp = nums[0], sum = 0.0;
        size_t i;

        for (i = 1 ; i < ct ; i++)
            if (nums[i] > max_exp)
                max_exp = nums[i];

        for (i = 0; i < ct ; i++)
            sum += exp(nums[i] - max_exp);

        return log(sum) + max_exp;
    }

    // Numerically stable way of computing f(a) = log(1 - exp(-a))
    inline double log1mexp(double a) {
        assert(a > 0);
        if(a <= a0) {
            return std::log(-std::expm1(-a));
        } else {
            return std::log1p(-std::exp(-a));
        }
    }

    // Numerically stable way of computing f(a) = log(1 + exp(a))
    inline double log1pexp(double a) {
        if(a <= -37.0) {
            return std::exp(a);
        } else if(a > -37.0 && a <= 18.0) {
            return std::log1p(std::exp(a));
        } else if(a > 18.0 && a <= 33.3) {
            return a + std::exp(-a);
        } else if(a > 33.3) {
            return a;
        }
        return a; // shouldn't get here
    }

    inline double log_substract(double l1, double l2) {
        if(l1 <= l2) throw new std::logic_error("computing log of negative number");
        if(l2 == -std::numeric_limits<double>::infinity()) return l1;
        return l1 + log1mexp(l1-l2);
    }

    inline double log_add(double a, double b) {
        if (a == -std::numeric_limits<double>::infinity()) return b;
        if (b == -std::numeric_limits<double>::infinity()) return a;
        return a>b ? a+log1pexp(b-a) : b+log1pexp(a-b);
    }

    inline void log_plus_equals(double &l1, double l2) {
        l1 = log_add(l1,l2);
    }

    template<typename V>
    inline double log_add(V vs) {
        double ret = -std::numeric_limits<double>::infinity();
        //for (auto v : vs) { log_plus_equals(ret, v); }
        for(auto v : vs) {
            ret = log_add(ret, v);
        }
        return ret;
    }

    inline bool logically_equal(double a, double b, double error_factor=1.0) {
        return a==b ||
            std::abs(a-b)<std::abs(std::min(a,b))*std::numeric_limits<double>::epsilon()*
                          error_factor;
    }
}

#endif
