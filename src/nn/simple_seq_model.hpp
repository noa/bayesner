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

#ifndef __NN_SIMPLE_SEQ_MODEL_HPP__
#define __NN_SIMPLE_SEQ_MODEL_HPP__

#include <nn/mu.hpp>
#include <nn/rng.hpp>
#include <nn/discrete_distribution.hpp>
#include <nn/uniform.hpp>
#include <nn/fixed_depth_hpyp.hpp>
#include <nn/mutable_symtab.hpp>

namespace nn {

    template<typename T = size_t,
             typename H = SimpleDiscreteMeasure<T>,
             typename M = FixedDepthHPYP<T, T, H, 7>
             >
    class simple_seq_model {
        static constexpr double STOP_WEIGHT { 5.0 };

        typedef std::vector<T> seq_t; // context vector

        const T BOS;
        const T EOS;

        std::unique_ptr<H> base;
        std::unique_ptr<M> model;

    public:
        struct param {
            size_t nsyms;
            T BOS;
            T EOS;
        };

        simple_seq_model(size_t nsyms, T _BOS, T _EOS) : BOS(_BOS), EOS(_EOS) {
            base  = std::make_unique<H>(nsyms);
            model = std::make_unique<M>(base.get());
            init();
            debug_log_info();
        }

        simple_seq_model(param p) : simple_seq_model(p.nsyms, p.BOS, p.EOS) {}

        void init() {
            base->set_weight(EOS,
                             static_cast<double>(base->cardinality())/STOP_WEIGHT);
        }

        void debug_log_info() {
            DLOG(INFO) << "H cardinality: " << base->cardinality() << " BOS: " << BOS << " EOS: " << EOS << " pr(EOS) = " << base->prob(EOS);
        }

        T get_initial_symbol()     { return BOS;    }
        T get_initial_state()      { return BOS;    }
        T get_final_symbol()       { return EOS;    }
        T get_final_state()        { return EOS;    }
        H* get_base()              { return base.get();  }
        M* get_model()             { return model.get(); }

        double log_prob(const seq_t& seq) const {
            CHECK(seq.front() == BOS) << "seq doesn't start with BOS";
            double ret = 0.0;
            auto start = seq.begin();
            for(auto iter = std::next(start); iter != seq.end(); iter++) {
                auto lp = model->log_prob(start, iter, *iter);
                ret += lp;
            }
            return ret;
        }

        double prob(const seq_t& seq, T obs) const {
            return model->prob(seq, obs);
        }

        double log_prob(const seq_t& seq, T obs) const {
            return model->log_prob(seq, obs);
        }

        double log_prob_stop(const seq_t& seq) const {
            return model->log_prob(seq, EOS);
        }

        double log_prob_cont(const seq_t& seq) const {
            return log(1.0-model->prob(seq, EOS));
        }

        void observe(const seq_t& seq) {
            CHECK(seq.front() == BOS) << "seq doesn't start with BOS";
            CHECK(seq.back() == EOS) << "seq doesn't stop with EOS";
            try {
                auto start = seq.begin();
                for(auto iter = std::next(start); iter != seq.end(); iter++) {
                    model->observe(start, iter, *iter);
                }
            }
            catch (const std::out_of_range& oor) {
                std::cerr << "Out of Range error: " << oor.what() << '\n';
                exit(1);
            }
        }

        void remove(const seq_t& seq) {
            CHECK(seq.front() == BOS) << "seq doesn't start with BOS";
            CHECK(seq.back() == EOS) << "seq doesn't stop with EOS";

            auto start = seq.begin();
            for(auto iter = std::next(start); iter != seq.end(); iter++) {
                model->remove(start, iter, *iter);
            }
        }

        // WARNING: This assumes symbols are contiguous
        // e.g.  0, 1, 2
        // NOT:  0, 2, 3
        nn::discrete_distribution<T> dist(const seq_t& context, bool include_final = false) const {
            T s;
            VLOG(1000) << "base cardinality: " << base.cardinality();
            VLOG(1000) << "include final? " << include_final;
            nn::discrete_distribution<T> ret;
            for(s=0; s<base.cardinality(); ++s) {
                //if(s != EOS && s != BOS) {
                CHECK(s != BOS) << "unexpected symbol: " << BOS << " BOS = " << BOS << " EOS = " << EOS;
                if(s != EOS) {
                    VLOG(1000) << "adding transition to " << s;
                    ret.push_back_prob(s, model->prob(context, s));
                }
            }
            if(context.size() > 0 && include_final) {
                VLOG(1000) << "adding (final) transition to " << EOS;
                ret.push_back_prob(EOS, model->prob(context, EOS)); // stop probability
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
