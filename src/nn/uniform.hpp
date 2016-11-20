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

#ifndef __NN_UNIFORM_HPP__
#define __NN_UNIFORM_HPP__

#include <cmath>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <unordered_map>

namespace nn {

    template<typename T>
    struct Uniform {
        size_t size;
        Uniform(size_t _size) : size(_size) {};
        double prob(T t) const {
            return 1.0/(double)size;
        }
        double log_prob(T t) const {
            return log(prob(t));
        }
        void observe(T t) {}
        size_t cardinality() { return size; }
    };

    template<typename T>
    struct HashIntegralMeasure {
        std::unordered_map<T,double> weight;
        double Z {0};

    public:
        double prob(T t)     const { return weight.at(t)/Z; }
        double log_prob(T t) const { return log(prob(t));   }

        void add(T t, double w) {
            weight[t] = w;
            Z += w;
        }

        void observe(T t) {}
        size_t cardinality() const { return weight.size(); }
        double partition()   const { return Z;             }
    };

    template<typename T>
    class SimpleDiscreteMeasure {
        std::vector<double> weights;
        std::vector<double> probs;

        void normalize() {
            double Z = std::accumulate(weights.begin(), weights.end(), 0.0);
            for(size_t i = 0; i < weights.size(); ++i) {
                probs[i] = weights[i]/Z;
            }
        }

    public:
        SimpleDiscreteMeasure(size_t nitems) {
            for(size_t i = 0; i < nitems; ++i) {
                weights.push_back(1.0);
            }
            probs.reserve(nitems);
            normalize();
        }

        double prob(T t) const {
            return probs[t];
        }

        double log_prob(T t) const {
            return log(probs[t]);
        }

        void set_weight(T t, double w) {
            weights[t] = w;
            normalize();
        }

        size_t cardinality() const {
            return weights.size();
        }
    };

    // NOTE: not in log-space
    class SimpleBaseDistribution {
        std::vector<double> weights;
        std::vector<double> probs;

        void normalize() {
            double Z = std::accumulate(weights.begin(), weights.end(), 0.0);
            for(size_t i = 0; i < weights.size(); ++i) {
                probs[i] = weights[i]/Z;
            }
        }

    public:
        SimpleBaseDistribution(size_t nitems) {
            for(size_t i = 0; i < nitems; ++i) {
                weights.push_back(1.0);
            }
            probs.reserve(nitems);
            normalize();
        }
        double prob(size_t state) const {
            return probs[state];
        }
        double log_prob(size_t state) const {
            return log(probs[state]);
        }
        void set_weight(size_t state, double w) {
            weights[state] = w;
            normalize();
        }
        size_t cardinality() const {
            return weights.size();
        }
    };
}

#endif
