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

#ifndef __NN_ADAPTED_SEQ_MODEL_HPP__
#define __NN_ADAPTED_SEQ_MODEL_HPP__

#include <iterator>
#include <type_traits>
#include <vector>
#include <utility>

#include <nn/discrete_distribution.hpp>
#include <nn/trie_histogram_restaurant.hpp>
#include <nn/simple_seq_model.hpp>
#include <nn/seq_pyp.hpp>

namespace nn {

    template<typename T = size_t,
             typename H = simple_seq_model<T>,
             typename A = seq_pyp<T>>
    class adapted_seq_model {
        static_assert(std::is_integral<T>::value, "non-integral symbol type");

        typedef std::pair<double, double> prm;
        typedef std::vector<T> seq_t;

        std::unique_ptr<H> base;
        std::unique_ptr<A> crp;
        prm p;

        const T BOS;
        const T EOS;
        const T SPACE;

    public:
        struct param {
            size_t nsyms;
            T BOS;
            T EOS;
            T SPACE;
            double discount {0.1};
            double alpha    {1.0};
        };

        adapted_seq_model(param p) : p(p.discount, p.alpha),
                                     BOS(p.BOS),
                                     EOS(p.EOS),
                                     SPACE(p.SPACE) {
            base = std::make_unique<H>(p.nsyms, p.BOS, p.EOS);
            crp = std::make_unique<A>(p.BOS, p.EOS, p.SPACE);
        }

        H* get_base() const { return base.get(); }

        double log_prob(const seq_t& seq) const {
            auto log_p0 = base->log_prob(seq);
            return crp->log_prob(seq, log_p0, p.first, p.second);
        }

        double log_prob(const std::vector<seq_t>& prefix, const seq_t& last) {
            CHECK(false) << "unimplemented";
            return 0.0;
        }

        double log_prefix_prob(const seq_t& seq) const {
            CHECK(false) << "unimplemented";
            return 0.0;
        }

        double log_prefix_prob(const std::vector<seq_t>& prefix,
                               const seq_t& last) {
            CHECK(false) << "unimplemented";
            return 0.0;
        }

        double log_cached_prob(const seq_t& seq) const {
            return crp->log_cache_prob(seq, p.first, p.second);
        }

        double log_new_prob(const seq_t& seq) const {
            auto log_p0 = base->log_prob(seq);
            return crp->log_new_prob(seq, log_p0, p.first, p.second);
        }

        size_t get_num_tables() const {
            return crp->get_t();
        }

        size_t get_num_customers() const {
            return crp->get_c();
        }

        void observe(const seq_t& seq) {
            CHECK(seq.front() == BOS) << "bad first symbol: " << seq.front();
            CHECK(seq.back() == EOS)  << "bad last symbol: "  << seq.back();

            auto log_p0 = base->log_prob(seq);
            auto new_table = crp->add(seq, log_p0, p.first, p.second);
            if(new_table) {
                base->observe(seq);
            }
        }

        void observe(typename seq_t::const_iterator, size_t len) {
            CHECK(false) << "unsupported";
        }

        void observe(typename std::vector<seq_t>::const_iterator start,
                     typename std::vector<seq_t>::const_iterator stop) {
            observe(join(start, stop, BOS, SPACE, EOS));
        }

        void remove(const seq_t& seq) {
            CHECK(seq.front() == BOS) << "unexpected first symbol: " << seq.front();
            CHECK(seq.back() == EOS) << "unexpected last symbol: " << seq.back();
            auto ret = crp->remove(seq, p.first, p.second);
            if(ret) {
                base->remove(seq);
            }
        }

        void set_emission_param(double d, double a) {
            p.first = d;
            p.second = a;
        }

        discrete_distribution<seq_t> match(const seq_t& seq) {
            CHECK(false) << "unimplemented";
        }
    };
};

#endif
