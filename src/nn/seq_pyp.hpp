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

#ifndef __NN_SEQ_PYP_HPP__
#define __NN_SEQ_PYP_HPP__

#include <vector>

#include <nn/log.hpp>
#include <nn/restaurants.hpp>

namespace nn {

template <typename S,
          typename T = std::vector<S>,
          typename Restaurant = SimpleFullRestaurant<T>
          >
class seq_pyp : public restaurant_interface<T> {

    Restaurant restaurant;

    void* crp  {nullptr};   // CRP sufficient statistics
    void* data {nullptr};   // additional data required by the CRP

public:
    seq_pyp(S _BOS, S _EOS, S _SPACE) {
        crp = restaurant.getFactory().make();
    }
    ~seq_pyp() {
        restaurant.getFactory().recycle(crp);
    }

    // Forbid copy constructor and assignment
    seq_pyp(seq_pyp const&) = delete;
    seq_pyp& operator=(seq_pyp const&) = delete;

    size_t get_c(const T& obs) const override {
        return restaurant.getC( crp, obs );
    }

    size_t get_c() const override {
        return restaurant.getC( crp );
    }

    size_t get_t(const T& obs) const override {
        return restaurant.getT( crp, obs );
    }

    size_t get_t() const override {
        return restaurant.getT( crp );
    }

    double prob(const T& obs, double p0, double d, double a) const override {
        return restaurant.computeProbability( crp, obs, p0, d, a );
    }

    double log_prob(const T& obs, double ln_p0, double d, double a) const override {
        return restaurant.computeLogProbability( crp, obs, ln_p0, d, a );
    }

    double log_cache_prob(const T& obs, double d, double a) const override {
        return restaurant.computeLogCacheProb( crp, obs, d, a );
    }

    double log_new_prob(double ln_p0, double d, double a) const override {
        return restaurant.computeLogNewProb( crp, ln_p0, d, a );
    }

    bool add(const T& obs, double ln_p0, double d, double a) override {
        return restaurant.logAddCustomer( crp, obs, ln_p0, d, a, nullptr );
    }

    bool remove(const T& obs, double d, double a) override {
    //std::cout << "seq_pyp remove..." << std::endl;
    //LOG(INFO) << " got here ... ";
        data = restaurant.createAdditionalData(crp, d, a);
        auto removed_table = restaurant.removeCustomer(crp, obs, d, data);
        restaurant.freeAdditionalData(data);
        data = nullptr;
        return removed_table;
    }
};

}

#endif
