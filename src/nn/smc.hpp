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

#ifndef __NN_SMC_HPP__
#define __NN_SMC_HPP__

#include <vector>
#include <random>
#include <utility>
#include <map>
#include <omp.h>

#include <nn/log.hpp>
#include <nn/rng.hpp>

enum class rmethod {
    SMC_RESAMPLE_MULTINOMIAL,
    SMC_RESAMPLE_RESIDUAL,
    SMC_RESAMPLE_STRATIFIED,
    SMC_RESAMPLE_SYSTEMATIC,
    SMC_RESAMPLE_NONE
};

template<typename P, typename O>
struct Filter {

    typedef std::vector<P> pvec;
    typedef std::vector<double> dvec;

    struct settings {
        size_t num_particles { 100 };
        double resample_threshold { 0.5 };
        rmethod resample { rmethod::SMC_RESAMPLE_NONE };
    };

    struct particle_system {
        pvec particle;
        dvec log_weight; // log weight of each particle
        dvec log_prob;   // distribution over particles (log space)
        double ess;      // effective sample size
        double log_Z;    // log sum of the weights
    };

    particle_system sys;
    settings config;

    // Temporary work buffers for resampling
    std::vector<size_t> resample_counts;
    std::vector<size_t> resample_indices;
    std::vector<double> resample_weights;

    Filter(settings _config) : config(_config) {
        DLOG(INFO) << config.num_particles << " particles";

        // Initialize storage
        sys.particle.resize(config.num_particles);
        sys.log_weight.resize(config.num_particles);
        sys.log_prob.resize(config.num_particles);

        resample_counts.resize(config.num_particles);
        resample_indices.resize(config.num_particles);
        resample_weights.resize(config.num_particles);
    }

    // initialize particle, returning incremental importance weight
    virtual double init_p(P& particle) = 0;

    // advance particle with the given observation, returning inremental importance weight
    virtual double move_p(P& particle, const O& obs) = 0;

    // setup work (e.g. precompute probabilities)
    // (Optional override)
    virtual void set_up(const std::vector<O>& obs) {}

    double get_log_partition() const {
        return sys.log_Z - log( (double)config.num_particles );
    }

    double get_ess() const {
        return sys.ess;
    }

    // compute ESS
    double calc_ess() const {
        auto sum = 0.0;
        for(auto lp : sys.log_prob) { // CHECK: log_prob or log_weight?
            auto p = std::exp(lp);
            sum += (p*p); //pow(p, 2);
        }
        return 1.0/sum;
    }

    // compute expected sample size and normalizing constant
    void update() {
        sys.log_Z = nn::log_add(sys.log_weight);
        for(size_t m = 0; m < config.num_particles; ++m) {
            sys.log_prob[m] = sys.log_weight[m] - sys.log_Z;
        }
        sys.ess = calc_ess();
    }

    // initialize particle system
    void init(size_t start, size_t stop) {

        #pragma omp parallel for
        for(size_t m=start; m<stop; ++m) {
            sys.log_weight[m] = init_p(sys.particle[m]); // NOTE: = not +=
        }

        update();
    }

    // advance particle system over given observation
    void advance(const O& obs, size_t start, size_t stop) {
        #pragma omp parallel for
        for (size_t m=start; m<stop; ++m) {
             sys.log_weight[m] += move_p(sys.particle[m], obs);
        }

        update(); // update weights and calculate ESS

        if(config.resample == rmethod::SMC_RESAMPLE_NONE) {
            return;
        }

        // if ESS drops below a threshold, resample
        if(config.resample_threshold < 1) {
            auto f = sys.ess / (double)config.num_particles;
            if( f < config.resample_threshold ) {
                resample();
            }
        } else {
            if( sys.ess < config.resample_threshold )  {
                resample();
            }
        }
    }

    // run SMC over a vector of fixed length
    void smc(const std::vector<O>& obs) {
        size_t start = 0;
        size_t stop = config.num_particles;

        // setup work (e.g. precompute distributions)
        set_up(obs);

        // initialize particles
        init(start, stop);

        // run particle filter
        for(auto o : obs) {
            advance(o, start, stop);
        }
    }

    // resample particle system
    void resample() {

        //Resampling is done in place.
        double dWeightSum = 0;
        size_t uMultinomialCount;

        std::fill(resample_counts.begin(), resample_counts.end(), 0);
        std::fill(resample_indices.begin(), resample_indices.end(), 0);

        //First obtain a count of the number of children each particle has.
        switch( config.resample ) {
        case rmethod::SMC_RESAMPLE_MULTINOMIAL:
        {
            //Sample from a suitable multinomial vector
            for(size_t m = 0; m < config.num_particles; ++m) {
                resample_weights[m] = exp( sys.log_prob[m] );
            }
            nn::multinomial( nn::rng::get(), resample_weights, resample_counts, config.num_particles );
            break;
        }
        case rmethod::SMC_RESAMPLE_RESIDUAL:
        {
            //Sample from a suitable multinomial vector and add the integer replicate
            //counts afterwards.

            for(size_t m = 0; m < config.num_particles; ++m) {
                resample_weights[m] = exp( sys.log_prob[m] );
                dWeightSum += resample_weights[m];
            }

            uMultinomialCount = config.num_particles;
            for(size_t m = 0; m < config.num_particles; ++m) {
                resample_weights[m] = config.num_particles * resample_weights[m] / dWeightSum;
                resample_indices[m] = size_t(floor(resample_weights[m]));
                resample_weights[m] = (resample_weights[m] - resample_indices[m]);
                uMultinomialCount -= resample_indices[m];
            }

            nn::multinomial( nn::rng::get(), resample_weights, resample_counts, uMultinomialCount );

            break;
        }
        case rmethod::SMC_RESAMPLE_STRATIFIED:
        default:
        {
            dWeightSum = 0;
            double dWeightCumulative = 0;
            for(size_t m = 0; m < config.num_particles; ++m) {
                dWeightSum += exp( sys.log_prob[m] );
            }

            //Generate a random number between 0 and 1/N times the sum of the weights
            double dRand = nn::uni_range(nn::rng::get(), 0.0, 1.0/(double)config.num_particles);

            size_t j = 0, k = 0;
            dWeightCumulative = exp( sys.log_prob[0] ) / dWeightSum;
            while(j < config.num_particles) {
                while((dWeightCumulative - dRand) > ((double)j)/((double)config.num_particles) && j < config.num_particles) {
                    resample_counts[k]++;
                    j++;
                    dRand = nn::uni_range(nn::rng::get(), 0.0, 1.0 / ((double)config.num_particles));
                }
                k++;
                dWeightCumulative += exp( sys.log_prob[k] ) / dWeightSum;
            }

            break;
        }
        case rmethod::SMC_RESAMPLE_SYSTEMATIC:
        {
            // Procedure for stratified sampling but with a common RV for each stratum
            dWeightSum = 1.0;
            double dWeightCumulative = 0;

            //Generate a random number between 0 and 1/N times the sum of the weights
            double dRand = nn::uni_range(nn::rng::get(), 0.0, 1.0 / ((double)config.num_particles));
            dWeightCumulative = exp( sys.log_prob[0] );

            size_t j = 0, k = 0;
            while(j < config.num_particles) {
                while((dWeightCumulative - dRand) > ((double)j)/((double)config.num_particles) && j < config.num_particles) {
                    resample_counts[k]++;
                    j++;
                }
                k++;
                dWeightCumulative += exp( sys.log_prob[k] );
            }

            break;
        }
        }

        for(size_t i=0, j=0; i<config.num_particles; ++i) {
            if(resample_counts[i] > 0) {
                resample_indices[i] = i;
                while( resample_counts[i] > 1 ) {
                    while( resample_counts[j] > 0 ) ++j; // find next free spot
                    resample_indices[j++] = i;
                    --resample_counts[i];
                }
            }
        }

        for(size_t m = 0; m < config.num_particles; ++m) {
            if(resample_indices[m] != m) {
                sys.particle[m] = sys.particle[resample_indices[m]];
            }
            sys.log_weight[m] = 0;
            sys.log_prob[m] = log(1.0 / (double)config.num_particles);
        }
    }

    std::vector<std::pair<P,double>> get_particle_log_probs() const {
        std::vector<std::pair<P,double>> ret;
        for(size_t m = 0; m < sys.particle.size(); ++m) {
            ret.push_back( std::make_pair(sys.particle[m], sys.log_prob[m]) );
        }
        return ret;
    }
};

#endif
