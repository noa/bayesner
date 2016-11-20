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

#ifndef __NN_EMPIRICAL_DIST_HPP__
#define __NN_EMPIRICAL_DIST_HPP__

#include <unordered_map>

namespace nn {

  template<typename T, typename U = unsigned long long>
  class EmpiricalDist {
    std::unordered_map<T,U> counts;
    U N {0};
  public:
    EmpiricalDist() {};

      void observe(T t) {
          counts[t] += 1;
          N += 1;
      }

      bool has_key(T type) const {
          return !(counts.find(type) == counts.end());
      }

      double prob(T type) const {
          if(!has_key(type)) {
              return 0.0;
          }
          auto c = counts.at(type);
          return (double)c/(double)N;
      }

      std::size_t size() {
          return counts.size();
      }

      std::size_t total() {
          return N;
      }

      double operator [] (T type) const {
          return prob(type);
      }

      auto begin() const -> decltype (counts.begin()) {
          return counts.begin();
      }

      auto end() const -> decltype (counts.end()) {
          return counts.end();
      }

  };

    template<typename T>
    class StaticEmpiricalDist {
        std::unordered_map<T,double> counts;
        unsigned long long N {0};
        bool normalized {false};
    public:
        StaticEmpiricalDist() {};

        void observe(T t) {
            counts[t] += 1.0;
            ++N;
        }

        void normalize() {
            assert(!normalized);
            for(auto keyval : counts) {
                counts[keyval.first] /= (double)N;
            }
            normalized = true;
        }

        auto size() -> decltype(counts.size()) {
            return counts.size();
        }

        double prob(T t) const {
            assert(normalized);
            return counts[t];
        }

        double operator [] (T type) const {
            return prob(type);
        }

        auto begin() const -> decltype (counts.begin()) {
            return counts.begin();
        }

        auto end() const -> decltype (counts.end()) {
            return counts.end();
        }
    };

}

#endif
