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

#ifndef __NN_SEQ_MODEL_HPP__
#define __NN_SEQ_MODEL_HPP__

#include <nn/mu.hpp>
#include <nn/rng.hpp>
#include <nn/discrete_distribution.hpp>
#include <nn/uniform.hpp>
#include <nn/fixed_depth_hpyp.hpp>

namespace nn {

    template<typename T = size_t,
             typename C = size_t,
             typename H = SimpleDiscreteMeasure<T>,
             typename M = FixedDepthHPYP<T, C, H, 5>
             >
    class seq_model {
        static constexpr double STOP_WEIGHT { 5.0 };

        typedef std::vector<T> seq_t;     // observation vector
        typedef std::vector<C> context_t; // context vector

        const T BOS;
        const T EOS;
        H base;
        M model;

    public:
        struct param {
            size_t nsyms;
            T BOS;
            T EOS;
        };

        seq_model(size_t nsyms, T _BOS, T _EOS) : BOS(_BOS),
                                                  EOS(_EOS),
                                                  base(nsyms), // includes BOS and EOS
                                                  model(base) {
            init(); debug_log_info();
        }

        seq_model(param p) : seq_model(p.nsyms, p.BOS, p.EOS) {
            init(); debug_log_info();
        }

        void init() {
            base.set_weight(EOS, static_cast<double>(base.cardinality())/STOP_WEIGHT);
        }

        void debug_log_info() {
            DLOG(INFO) << "H cardinality: " << base.cardinality() << " BOS: " << BOS << " EOS: " << EOS << " pr(EOS) = " << base.prob(EOS);
        }

        T get_initial_symbol()     { return BOS;    }
        T get_initial_state()      { return BOS;    }
        T get_final_symbol()       { return EOS;    }
        T get_final_state()        { return EOS;    }
        H* get_base()              { return &base;  }
        M* get_model()             { return &model; }

        double prob(const context_t& seq, T obs) const {
            return model.prob(seq, obs);
        }

        double log_prob(const context_t& seq, T obs) const {
            return model.log_prob(seq, obs);
        }

        void observe(const context_t& seq, T obs) {
            model.observe(seq.begin(), seq.end(), obs);
        }

        void remove(const context_t& seq, T obs) {
            model.remove(seq.begin(), seq.end(), obs);
        }

        discrete_distribution<T> dist(const context_t& context, bool include_final = false) const {
            T s;
            discrete_distribution<T> ret;
            for(s=0; s<base.cardinality(); ++s) {
                CHECK(s != BOS) << "unexpected symbol: " << BOS << " BOS = " << BOS << " EOS = " << EOS;
                if(s != EOS) {
                    VLOG(1000) << "adding transition to " << s;
                    ret.push_back_prob(s, model.prob(context, s));
                }
            }
            if(context.size() > 0 && include_final) {
                VLOG(1000) << "adding (final) transition to " << EOS;
                ret.push_back_prob(EOS, model.prob(context, EOS)); // stop probability
            }
            return ret;
        }

        void set_prior(const std::map<size_t, double>& prior) {
            LOG(INFO) << "setting symbol priors:";
            for(auto keyval : prior) {
                LOG(INFO) << keyval.first << " weight = " << keyval.second;
                base.set_weight(keyval.first, keyval.second);
            }
        }
    };
};

#endif
