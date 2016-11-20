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

#ifndef __NN_ADAPTED_SEQ_PREFIX_HPP__
#define __NN_ADAPTED_SEQ_PREFIX_HPP__

#include <cmath>
#include <iterator>
#include <type_traits>
#include <vector>
#include <utility>

#include <nn/prefix_matcher.hpp>
#include <nn/adapted_seq_model.hpp>
#include <nn/data.hpp>

namespace nn {

    template<typename T = size_t,
             typename H = simple_seq_model<T>,
             typename A = seq_pyp<T>>
    class adapted_seq_model_prefix {
        static_assert(std::is_integral<T>::value, "non-integral symbol type");

        typedef std::pair<double, double> prm;
        typedef std::vector<T> seq_t;

        PrefixMap<T,int> matcher;

        H base; // base distribution
        prm p;  // parameters
        A crp;  // adaptor

        const T BOS;
        const T EOS;
        const T SPACE;

    public:
        struct param {
            size_t nsyms;
            T BOS;
            T EOS;
            T SPACE;
            double discount {0.5};
            double alpha    {0.1};
        };

        adapted_seq_model_prefix(param p) : base(p.nsyms, p.BOS, p.EOS),
                                            p(p.discount, p.alpha),
                                            crp(p.BOS, p.EOS, p.SPACE),
                                            BOS(p.BOS),
                                            EOS(p.EOS),
                                            SPACE(p.SPACE) {}

        double log_stop_prob(const seq_t& prefix) const {
            return base.log_prob(prefix, EOS);
        }

        double log_cont_prob(const seq_t& prefix) const {
            //return log(1.0-base.prob(prefix, EOS));
            return base.log_prob(prefix, SPACE);
        }

        double log_prob(const seq_t& seq) const {
            CHECK(seq.front() == BOS) << "unexpected 1st symbol: "
                                      << seq.front();
            CHECK(seq.back()  == EOS) << "unexpected last symbol: "
                                      << seq.back();
            auto log_p0 = base.log_prob(seq);
            return crp.log_prob(seq, log_p0, p.first, p.second);
        }

        double log_prob(const std::vector<seq_t>& prefix,
                        const seq_t& last) const {
            auto key = nn::from_vec(prefix, last, BOS, EOS, SPACE);
            return log_prob(key);
        }

        // Calculates the total probability of *future* expansions of
        // the given key
        double log_prefix_prob(const seq_t& key) const {
            auto cache_log_prob = NEG_INF;

            const auto key_len = key.size();
            const auto result  = matcher.match_prefix(key);

            // Tally the probability of existing strings
            for(auto it = result.first; it != result.second; ++it) {
                if (it->first.at(key_len) == EOS) continue;
                auto lcp = log_cached_prob(it->first);
                nn::log_plus_equals(cache_log_prob, lcp);
            }

            // Tally the probability of generating the given key from
            // the base, and then NOT generating the STOP symbol.
            auto new_log_prob = log_new_prob(key) + log_cont_prob(key);
            return nn::log_add(cache_log_prob, new_log_prob);
        }

        H* get_base() { return &base; }

        double log_prefix_prob(const std::vector<seq_t>& prefix,
                               const seq_t& last)
            const {
            auto key = from_vec(prefix, last, BOS, EOS, SPACE);
            key.erase(key.end()-1); // remove EOS
            return log_prefix_prob( key );
        }

        double log_stop_prob(const std::vector<seq_t>& prefix,
                             const seq_t& last)
            const {
            auto key = from_vec(prefix, last, BOS, EOS, SPACE);
            return log_prob( key );
        }

        double log_cached_prob(const seq_t& seq) const {
            return crp.log_cache_prob(seq, p.first, p.second);
        }

        double log_new_prob(const seq_t& seq) const {
            CHECK(seq[0] == BOS) << "unexpected first symbol: " << seq[0];
            auto log_p0 = base.log_prob(seq);
            return crp.log_new_prob(log_p0, p.first, p.second);
        }

        size_t get_num_tables() const {
            return crp.get_t();
        }

        size_t get_num_customers() const {
            return crp.get_c();
        }

        void observe(const seq_t& seq) {
            CHECK(seq.front() == BOS) << "unexpected first symbol: " << seq.front();
            CHECK(seq.back()  == EOS) << "unexpected last symbol: "  << seq.back();
            matcher.add(seq, 0);
            auto log_p0 = base.log_prob(seq);
            auto new_table = crp.add(seq, log_p0, p.first, p.second);
            if(new_table) {
                base.observe(seq);
            }
        }

        double log_prob(typename std::vector<seq_t>::const_iterator start,
                        typename std::vector<seq_t>::const_iterator stop) {
            return log_prob(nn::join(start, stop, BOS, SPACE, EOS));
        }

        void observe(typename std::vector<seq_t>::const_iterator start,
                     typename std::vector<seq_t>::const_iterator stop) {
            observe(nn::join(start, stop, BOS, SPACE, EOS));
        }

        void observe(const std::vector<seq_t>& seqs) {
            observe(seqs.begin(), seqs.end());
        }

        void remove(const seq_t& seq) {
            CHECK(seq.front() == BOS) << "unexpected first symbol: "
                                      << seq.front();
            CHECK(seq.back() == EOS) << "unexpected last symbol: "
                                     << seq.back();
            auto ret = crp.remove(seq, p.first, p.second);
            if(ret) {
                base.remove(seq);
            }
        }

        void remove(typename std::vector<seq_t>::const_iterator start,
                    typename std::vector<seq_t>::const_iterator stop) {
            auto joined = nn::join(start, stop, BOS, SPACE, EOS);
            remove(joined);
        }

        void set_emission_param(double d, double a) {
            p.first = d;
            p.second = a;
        }

        template<typename RNG>
        void resample_hyperparameters(RNG& rng) {}
    };
}

#endif
