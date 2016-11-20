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

#ifndef __NN_STAT_HPP__
#define __NN_STAT_HPP__

#include <cstdlib>
#include <cassert>
#include <cmath>
#include <cfloat>
#include <limits>

#include <boost/math/distributions/students_t.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <nn/rng.hpp>
#include <nn/empirical_dist.hpp>

namespace nn {

    template<typename T>
    double kl(std::unordered_map<T, double> P, std::unordered_map<T, double> Q) {
        double ret = 0.0;
        for(auto keyval : P) {
            ret += keyval.second * log(keyval.second/Q[keyval.first]);
        }
        return ret;
    }

    template<typename T>
    double kl(EmpiricalDist<T> P, EmpiricalDist<T> Q) {
        double ret = 0.0;
        for(auto keyval : P) {
            double p = P.prob(keyval.first);
            double q = Q.prob(keyval.first);
            ret += p * log(p/q);
        }
        return ret;
    }

    template<typename T>
    double kl(EmpiricalDist<T> P, std::unordered_map<T, double> Q) {
        double ret = 0.0;
        for(auto keyval : P) {
            double p = P.prob(keyval.first);
            double q = Q[keyval.first];
            ret += p * log(p/q);
        }
        return ret;
    }

    template<typename T>
    double ln_kl(EmpiricalDist<T> P, std::unordered_map<T, double> lnQ) {
        double ret = 0.0;
        for(auto keyval : P) {
            double p = P.prob(keyval.first);
            double ln_q = lnQ[keyval.first];
            //			ret += p * log(p/q);
            ret += p * (log(p) - ln_q);
        }
        return ret;
    }

    template<typename T>
    void plot_sample_kl(std::string file_path,
                        std::string legend,
                        std::vector<T> sample,
                        std::unordered_map<T, double> exact_ln_prob,
                        unsigned incr = 1) {

        // Step 1. Open the file
        std::ofstream of( file_path );

        // Step 2. Write the legend
        of << legend << std::endl;

        // Step 3. Plot KL divergence by iteration
        EmpiricalDist<T> running_prob;
        auto x = 0;
        for(auto e : sample) {
            running_prob.observe(e);
            if(x++ % incr == 0) {
                auto y = ln_kl(running_prob, exact_ln_prob);
                of << x << " " << y << std::endl;
            }
        }
    }

    template<typename T>
    void write_mean_kl(std::string out_path,
                       std::vector<std::vector<T>> samples,
                       std::unordered_map<T, double> exact_ln_prob) {

        using namespace boost::math;

        auto num_replications = samples.size();
        auto num_samples = samples[0].size();

        std::vector<std::vector<double>> ys; // KLs for each replication
        ys.resize(num_samples);

        for(auto r=0; r<num_replications; ++r) {
            auto sample = samples[r];
            EmpiricalDist<std::string> running_prob;
            auto x = 0;
            for(auto e : sample) {
                running_prob.observe(e);
                auto y = nn::ln_kl(running_prob, exact_ln_prob);
                ys[x].push_back(y);
                x++;
            }
        }

        // Step 1. Open the file
        std::ofstream of( out_path );

        // Step 2. Write the legend
        of << out_path << std::endl;

        students_t dist(num_replications - 1);
        double t = quantile(complement(dist, 0.05/2.0));

        for(auto x=0; x<num_samples; ++x) {
            auto mean = 0.0;
            for(auto kl : ys[x]) {
                mean += kl;
            }
            mean /= ys[x].size();
            auto Sd = 0.0;
            for(auto kl : ys[x]) {
                Sd += pow(kl-mean,2);
            }
            Sd /= ys[x].size()-1;
            Sd = sqrt(Sd);
            double w = t * Sd / sqrt(double(num_replications));
            of << x << " " << mean << " " << w << std::endl;
        }
    }

    template<typename RNG>
    int log_sample_index(const std::vector<double>& weights,
                         double total,
                         RNG& rng) {
        assert(weights.size() > 0);

        double threshold = log(uni(rng)) + total;
        double partialsum = -DBL_MAX;

        int i = 0;
        do {
            partialsum = log_add(partialsum, weights[i++]);
        } while(partialsum <= threshold && i < weights.size());

        return i-1;
    }

    template<typename RNG>
    int log_sample_index(const std::vector<double>& weights, RNG& rng) {
        assert(weights.size() > 0);

        double total = -DBL_MAX;
        for(auto lw : weights) {
            total = log_add(total, lw);
        }
        return log_sample_index(weights, total, rng);
    }

    template<typename RNG>
    int sample_index(const std::vector<double>& weights,
                     double total,
                     RNG& rng) {
        assert(weights.size() > 0);

        double threshold = uni(rng)*total;
        double partialsum = 0;

        int i = 0;

        do {
            partialsum += weights[i++];
        } while(partialsum <= threshold && i < weights.size());

        return i-1;
    }

    template<typename RNG>
    int sample_index(const std::vector<double>& weights, RNG& rng) {
        assert(weights.size() > 0);

        double total = 0.0;
        for(auto w : weights) {
            total += w;
        }
        return sample_index(weights, total);
    }

    template<typename T, typename RNG>
    std::pair<T,double> sample_object(const std::vector<std::pair<T,double>>&
                                      weighted_objects,
                                      double total,
                                      RNG& rng) {
        assert(weighted_objects.size() > 0);
        double threshold = uni(rng)*total;
        double partialsum = 0;
        int i = 0;

        do {
            partialsum += weighted_objects[i++].second;
        } while(partialsum <= threshold && i < weighted_objects.size());

        return weighted_objects[i-1];
    }

    template<typename T, typename RNG>
    std::pair<T,double> sample_object(const std::vector<std::pair<T,double>>&
                                      weighted_objects,
                                      RNG& rng) {
        assert(weighted_objects.size() > 0);
        double total = 0.0;
        for(int i=0; i<weighted_objects.size(); i++) {
            total += weighted_objects[i].second;
        }
        return sample_object(weighted_objects, total);
    }
}

#endif
