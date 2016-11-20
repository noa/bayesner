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

#ifndef __GENERIC_FILTER_HPP__
#define __GENERIC_FILTER_HPP__

#include <nn/rng.hpp>
#include <nn/csmc.hpp>

namespace nn {
    typedef std::vector<size_t>      Observation;
    typedef std::vector<Observation> Observations;

    template<typename Model, typename Particle>
    class generic_filter
        : public ConditionalFilter<Particle, Observation> {

        const Model& model;
        double zero_frac;

    public:
        generic_filter(
            typename ConditionalFilter<Particle, Observation>::settings _config,
            const Model& _model)
            : ConditionalFilter<Particle, Observation>(_config), model(_model) {
        }

        double get_zero_frac() { return zero_frac; }

        static typename ConditionalFilter<Particle, Observation>::settings
        get_default_config() {
            typename ConditionalFilter<Particle, Observation>::settings config;
            return config;
        }

        // initialize particle, returning the log incremental importance weight
        double init_p(Particle& particle) override {
            model.init(particle);
            return 0.0;
        }

        // Advance particle with the given observation, returning the
        // log incremental importance weight.
        double move_p(Particle& p, const Observation& obs) override {
            return model.extend(p, obs);
        }

        void swap_complete_particle(Particle& dst,
                                    const Particle& src) override {
            // model.init(dst);
            // dst.tags = src.tags;
            model.swap(dst, src);
        }

        // Compute the incremental weight for the given observation
        // (used for the fixed particle). Note that the time step t is
        // passed in to fetch the relevant part of the existing
        // particle.
        double incr_p(Particle& p,
                      const Observation& obs,
                      size_t t) override {
            return model.score(p, obs, t);
        }

        // e.g., precompute some probabilities
        void set_up(const std::vector<Observation>& obs) override {}

        Particle sample() {
            auto m = nn::sample_unnormalized_lnpdf(this->sys.log_weight,
                                                   nn::rng::get());
            return this->sys.particle[m];
        }

        Particle sample(const std::vector<size_t>& tags,
                        const std::vector<size_t>& lens,
                        const Observations& obs,
                        Annotation a) {
            switch(a) {
            case Annotation::FULL: {
                return model.make_particle(tags, lens);
            }
            case Annotation::SEMI: {
                return sample(tags, obs);
            }
            case Annotation::NONE: {
                return sample(obs);
            }
            case Annotation::UNDEF: {
                CHECK(false) << "undef annotation value";
            }
            }
        }

        // some of the tags may be unk
        Particle sample(const std::vector<size_t>& tags,
                        const Observations& obs) {

            Particle p;
            // TODO
            CHECK(false) << "implement this (sample with some unks)";
            return p;
        }

        Particle sample(const Observations& obs) {
            this->smc(obs);
            auto nzero = 0;
            for(auto m = 0; m < this->sys.log_weight.size(); ++m) {
                if(this->sys.log_weight[m] == NEG_INF) {
                    nzero++;
                }
            }
            CHECK(nzero < 1) << "bad system";
            zero_frac = (double)nzero/(double)this->sys.log_weight.size();
            auto m = nn::sample_unnormalized_lnpdf(this->sys.log_weight,
                                                   nn::rng::get());
            CHECK(this->sys.log_weight[m] != NEG_INF) << "sampled -inf particle";
            return this->sys.particle[m];
        }

        double estimate_log_partition(const Observations& obs) {
            this->smc(obs);
            return this->get_log_partition();
        }

        Particle conditional_sample(const Particle& p,
                                    const Observations& obs) {
            this->csmc(p, obs);
            auto m = nn::sample_unnormalized_lnpdf(this->sys.log_weight,
                                                   nn::rng::get());
            return this->sys.particle[m];
        }

        Particle conditional_sample(const Particle &p,
                                    const std::vector<size_t>& tags,
                                    const Observations& obs) {
            CHECK(false) << "unimplemented";
            auto m = nn::sample_unnormalized_lnpdf(this->sys.log_weight,
                                                   nn::rng::get());
            return this->sys.particle[m];
        }

        Particle conditional_sample(const Particle &p,
                                    const std::vector<size_t>& tags,
                                    const Observations& obs,
                                    Annotation a) {
                switch(a) {
                case Annotation::FULL: { // TODO: unnecessary memory allocations
                    return p;
                }
                case Annotation::SEMI: {
                    return conditional_sample(p, tags, obs);
                }
                case Annotation::NONE: {
                    return conditional_sample(p, obs);
                }
                case Annotation::UNDEF: {
                    CHECK(false) << "undef annotation value";
                }
                }
        }
    };
}

#endif
