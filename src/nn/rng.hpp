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

#ifndef __NN_RNG_HPP__
#define __NN_RNG_HPP__

#include <random>
#include <array>
#include <functional>
#include <unordered_map>
#include <omp.h>

#include <nn/mu.hpp>

typedef std::mt19937 RandomEngine;

namespace nn {

  struct rng {
    static std::vector<RandomEngine> engines;

    static RandomEngine get_prng() {
      std::random_device r;
      std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
      return std::mt19937(seed);
    }

    static RandomEngine get_prng(int seed) {
      return std::mt19937(seed);
    }

    static void init() {
      for(auto i = 0; i < omp_get_max_threads(); ++i) {
        engines.push_back( get_prng() );
      }
      LOG(INFO) << engines.size() << " threads.";
    }

    static RandomEngine & get() {
      return engines.at( omp_get_thread_num() );
    }

    static size_t get_num_engines() {
      return engines.size();
    }
  };

  // define the static member
  //std::unordered_map<std::thread::id, RandomEngine> rng::engines;
  std::vector<RandomEngine> rng::engines;

  template<typename RNG>
  double uni(RNG& rng) {
    static std::uniform_real_distribution<> d{};
    return d(rng);
  }

  template<typename RNG>
  double uni_pos(RNG& rng) {
    double d;
    do { d = uni(rng); } while(d == 0.0 || d == 1.0);
    return d;
  }

  template<typename F, typename RNG>
  inline unsigned sample_bernoulli(const F a, const F b, RNG& eng) {
    const F z = a + b;
    return static_cast<unsigned>(uni(eng) > (a/z));
  }

  template<typename F, typename RNG>
  inline unsigned log_sample_bernoulli(const F la, const F lb, RNG& eng) {
    const F lz = log_add(la, lb);
    const F lu = log(uni(eng));
    return static_cast<unsigned>(lu > (la - lz));
  }

  template<typename RNG>
  size_t randint(RNG& rng, size_t low, size_t high) {
    std::uniform_int_distribution<size_t> d(low, high);
    return d(rng);
  }

  template<typename RNG, typename Container>
  auto select(RNG& rng, const Container& c) -> typename Container::value_type {
    auto n = randint(rng, 0, c.size()-1);
    auto i = 0;
    for (auto it = c.begin(); it != c.end(); ++it) {
      if (n == i) return *it;
      i++;
    }
    CHECK(false) << "should never get here";
  }

    template<typename RNG, typename Container>
    auto pop(RNG& rng, Container& c) -> typename Container::value_type {
        auto ret = select(rng, c);
        c.erase(ret);
        return ret;
    }

    // TODO: test me
    template<typename RNG>
    double normal_one_d(RNG& rng, double mean, double stddev) {
        std::normal_distribution<> d(mean, stddev);
        return d(rng);
    }

    // TODO: test me
    template<typename RNG>
    double uni_range(RNG& gen, double start, double stop) {
        std::uniform_real_distribution<double> d(start, stop);
        return d(gen);
    }

    // TODO: test me
    template<typename RNG>
    void multinomial(RNG& gen, const std::vector<double>& weights, std::vector<size_t>& result, size_t count) {
        std::discrete_distribution<size_t> d(weights.begin(), weights.end());
        size_t i;
        for(i=0; i<count; ++i) {
            ++result[ d(gen) ];
        }
    }

    template<typename T, typename RNG>
    std::vector<T> rand_seq(size_t nsyms, size_t len, RNG& rng) {
        std::vector<T> ret;
        size_t i;
        for(i=0; i<len; ++i) {
            auto s = randint(rng, 0, nsyms-1);
            ret.push_back(s);
        }
        return ret;
    }


    /**
     * Returns true with probability true_prob.
     */
    template<typename RNG>
    bool coin(double true_prob, RNG& rng) {
        return (true_prob>uni(rng));
    }

    // TODO: test me
    template<typename Vector, typename RNG>
    size_t sample_normalized_pdf(Vector pdf, RNG& rng) {
        for (size_t i = 0; i < pdf.size()-1; ++i) { pdf[i+1] += pdf[i]; }
        double z = uni_pos(rng); // sample z ~ Uniform(0, 1.0)
        return std::lower_bound(pdf.begin(), pdf.end(), z) - pdf.begin();
    }

    template<typename Vector, typename RNG>
    size_t sample_unnormalized_lnpdf(Vector pdf, RNG& rng) {
        for (size_t i = 0; i < pdf.size()-1; ++i) {
            log_plus_equals(pdf[i+1], pdf[i]);
        }
        double z = log(uni_pos(rng)) + pdf.back();
        return std::lower_bound(pdf.begin(), pdf.end(), z) - pdf.begin();
    }

    template<typename Vector, typename RNG>
    size_t sample_unnormalized_pdf(Vector pdf, RNG& rng, int end_pos = 0) {
        // CHECK(pdf.size() > 0);
        // CHECK((size_t)end_pos < pdf.size());
        // CHECK(end_pos >= 0);

        // LOG(INFO) << "got here...";

        // if end_pos == 0, use entire vector
        if (end_pos == 0) {
            end_pos = pdf.size()-1;
        }

        // compute CDF (inplace)
        //LOG(INFO) << "computing CDF inplace...";
        for (int i = 0; i < end_pos; ++i) {
            //CHECK(pdf[i] >= 0);
            pdf[i+1] += pdf[i];
        }

        //CHECK(pdf[end_pos] > 0);

        // sample pos ~ Uniform(0,Z)
        double z = uni_pos(rng)*pdf[end_pos];

        //CHECK((z >= 0) && (z <= pdf[end_pos]));

        // Perform binary search for z using std::lower_bound.
        // lower_bound(begin, end, x) returns the first element within [begin,end)
        // that is equal or larger than x.
        //LOG(INFO) << "about to lower bound...";
        size_t x = std::lower_bound(pdf.begin(), pdf.begin() + end_pos + 1, z) - pdf.begin();

        //CHECK(x == 0 || pdf[x-1] != pdf[x]);

        return x;
    }

    int random_partition(int* arr, int start, int end) {
        //    srand(time(NULL));
        int pivotIdx = start + (int)uni(rng::get()) % (end-start+1); //check
        int pivot = arr[pivotIdx];
        std::swap(arr[pivotIdx], arr[end]); // move pivot element to the end
        pivotIdx = end;
        int i = start -1;

        for(int j=start; j<=end-1; j++) {
            if(arr[j] <= pivot) {
                i = i+1;
                std::swap(arr[i], arr[j]);
            }
        }

        std::swap(arr[i+1], arr[pivotIdx]);
        return i+1;
    }

    // int A[] = {9,5,7,1,10,2,3};
    // int loc = random_selection(A, 0,6,5);
    int random_selection(int* arr, int start, int end, int k) {
        if(start == end) {
            return arr[start];
        }

        if(k ==0) return -1;

        if(start < end) {
            int mid = random_partition(arr, start, end);
            int i = mid - start + 1;
            if(i == k) {
                return arr[mid];
            }
            else if(k < i) {
                return random_selection(arr, start, mid-1, k);
            }
            else {
                return random_selection(arr, mid+1, end, k-i);
            }
        }
        return 0; //warning: shouldn't get here?
    }

}

#endif
