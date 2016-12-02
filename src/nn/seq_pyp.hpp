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
#include <memory>

#include <nn/log.hpp>
#include <nn/restaurants.hpp>

#include <cereal/types/memory.hpp>

namespace nn {

template <typename S,
          typename T = std::vector<S>,
          typename Restaurant = SimpleFullRestaurant<T>
          >
class seq_pyp : public restaurant_interface<T> {
    Restaurant restaurant;
    std::unique_ptr<typename Restaurant::Payload> crp;

public:
    seq_pyp(S _BOS, S _EOS, S _SPACE)
        : crp(std::make_unique<typename Restaurant::Payload>()) {
    }

    seq_pyp(seq_pyp const&)            = delete;
    seq_pyp& operator=(seq_pyp const&) = delete;

    size_t get_c(const T& obs) const override {
        return restaurant.getC( crp.get(), obs );
    }

    size_t get_c() const override {
        return restaurant.getC( crp.get() );
    }

    size_t get_t(const T& obs) const override {
        return restaurant.getT( crp.get(), obs );
    }

    size_t get_t() const override {
        return restaurant.getT( crp.get() );
    }

    double prob(const T& obs, double p0, double d, double a) const override {
        return restaurant.computeProbability( crp.get(), obs, p0, d, a );
    }

    double log_prob(const T& obs, double ln_p0, double d, double a) const override {
        return restaurant.computeLogProbability( crp.get(), obs, ln_p0, d, a );
    }

    double log_cache_prob(const T& obs, double d, double a) const override {
        return restaurant.computeLogCacheProb( crp.get(), obs, d, a );
    }

    double log_new_prob(double ln_p0, double d, double a) const override {
        return restaurant.computeLogNewProb( crp.get(), ln_p0, d, a );
    }

    bool add(const T& obs, double ln_p0, double d, double a) override {
        return restaurant.logAddCustomer( crp.get(), obs, ln_p0, d, a );
    }

    bool remove(const T& obs, double d, double a) override {
        auto removed_table = restaurant.removeCustomer(crp.get(), obs, d);
        return removed_table;
    }

    template<class Archive>
    void serialize(Archive & archive) {
        archive( crp );
    }
};

}

#endif
