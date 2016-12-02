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

#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>

namespace nn {

    template<typename T>
    struct Uniform {
        size_t size {0};
        Uniform() {};
        Uniform(size_t _size) : size(_size) {};
        double prob(T t) const {
            return 1.0/(double)size;
        }
        double log_prob(T t) const {
            return log(prob(t));
        }
        void observe(T t) {}
        size_t cardinality() { return size; }

        template<class Archive>
        void serialize(Archive & archive) {
            archive( size );
        }
    };

    template<typename T>
    struct HashIntegralMeasure {
        std::unordered_map<T,double> weight;
        double Z {0};

    public:
        HashIntegralMeasure() {}

        HashIntegralMeasure(size_t nsyms, double w) {
            for(size_t i=0; i<nsyms; ++i) {
                add(i, w);
            }
        }

        HashIntegralMeasure(size_t nsyms) : HashIntegralMeasure(nsyms, 1.0) {}

        double prob(T t)     const { return weight.at(t)/Z; }
        double log_prob(T t) const { return log(prob(t));   }
        double w(T t)        const { return weight.at(t);   }

        void add(T t, double w) {
            weight[t] = w;
            Z += w;
        }

        void observe(T t) {}
        size_t cardinality() const { return weight.size(); }
        double partition()   const { return Z;             }

        template<class Archive>
        void serialize(Archive & archive) {
            archive( weight, Z );
        }
    };
}

#endif
