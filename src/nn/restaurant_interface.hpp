/*
 * Copyright 2008-2016 Jan Gasthaus
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

#ifndef __NN_RESTAURANT_INTERFACE_HPP__
#define __NN_RESTAURANT_INTERFACE_HPP__

#include <vector>

#include <nn/utils.hpp>

namespace nn {

    template <typename T>
    class restaurant_interface {
    public:
        virtual ~restaurant_interface() {}
        virtual size_t get_c(const T& type) const = 0;
        virtual size_t get_c() const = 0;
        virtual size_t get_t(const T& type) const = 0;
        virtual size_t get_t() const = 0;
        virtual double prob(const T& type, double p0,
                            double d, double a) const = 0;
        virtual double log_prob(const T& type, double log_p0,
                                double d, double a) const = 0;
        virtual double log_new_prob(double log_p0,
                                    double d, double a) const = 0;
        virtual double log_cache_prob(const T& type,
                                      double d, double a) const = 0;
        virtual bool add(const T& type, double log_p0, double d, double a) = 0;
        virtual bool remove(const T& type, double d, double a) = 0;
    };

    template <typename Dish>
    class IHPYPBaseRestaurant {
    public:
        typedef typename std::vector<Dish> TypeVector;
        typedef typename TypeVector::iterator TypeVectorIterator;
        virtual ~IHPYPBaseRestaurant() {}
        virtual l_type getC(void* payloadPtr, Dish type) const = 0;
        virtual l_type getC(void* payloadPtr)            const = 0;
        virtual l_type getT(void* payloadPtr, Dish type) const = 0;
        virtual l_type getT(void* payloadPtr)            const = 0;
        virtual double computeProbability(void* payloadPtr,
                                          Dish type,
                                          double parentProbability,
                                          double discount,
                                          double concentration) const = 0;
        virtual TypeVector getTypeVector(void* payloadPtr)      const = 0;
        virtual bool checkConsistency(void* payloadPtr)         const = 0;
    };

    template <typename Dish>
    class IAddRestaurant : public IHPYPBaseRestaurant<Dish> {
    public:
        virtual ~IAddRestaurant() {}
        virtual bool addCustomer(void* payloadPtr,
                                 Dish type,
                                 double parentProbability,
                                 double discount,
                                 double concentration) const = 0;
        virtual bool logAddCustomer(void* payloadPtr,
                                    Dish type,
                                    double parentLogProbability,
                                    double discount,
                                    double concentration) const = 0;
    };

    template <typename Dish>
    class IAddRemoveRestaurant : public IAddRestaurant<Dish> {
    public:
        virtual ~IAddRemoveRestaurant() {}
        virtual bool removeCustomer(void* payloadPtr,
                                    Dish type,
                                    double discount) const = 0;
    };
};

#endif
