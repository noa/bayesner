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

#ifndef __NN_CSMC_HPP__
#define __NN_CSMC_HPP__

#include <nn/smc.hpp>

template<typename P, typename O>
struct ConditionalFilter : public Filter<P,O> {

    ConditionalFilter(typename Filter<P,O>::settings _config)
        : Filter<P,O>(_config) {
    }

    // take final particle src and use init
    virtual void swap_complete_particle(P& dst, const P& src) = 0;

    // compute probability of state[t] and obs[t]
    virtual double incr_p(P& particle, const O& obs, size_t t) = 0;

    // run conditional SMC over a vector of fixed length
    void csmc(const P& fixed, const std::vector<O>& obs) {
        size_t start = 1;
        size_t stop  = this->config.num_particles;

        // set up
        this->set_up(obs);

        // initialize particles
        this->init(start, stop);

        // fix particle 0 and initialize its weight
        this->swap_complete_particle(this->sys.particle[0], fixed);
        this->sys.log_weight[0] = 0;

        // run particle filter
        size_t t = 0;
        for(auto o : obs) {
            // handle the fixed particle
            this->sys.log_weight[0] += incr_p(this->sys.particle[0], o, t);

            // handle the rest
            this->advance(o, start, stop);

            t++;
        }
    }
};

#endif
