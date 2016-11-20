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

#ifndef __NN_TRIE_HISTOGRAM_RESTAURANT_HPP__
#define __NN_TRIE_HISTOGRAM_RESTAURANT_HPP__

#include <utility>

#include <nn/rng.hpp>
#include <nn/trie.hpp>
#include <nn/restaurants.hpp>
#include <nn/restaurant_interface.hpp>

namespace nn {

    typedef size_t sym_t;
    typedef std::vector<sym_t> seq_t;
    typedef std::map<size_t, size_t> histogram_t;

    struct arrangement {
        size_t cw {0};
        size_t tw {0};
        histogram_t histogram;
    };

    class trie_histogram_restaurant : restaurant_interface<seq_t> {
        typedef ptr_trie<sym_t, arrangement> trie_t;

        trie_t trie;
        size_t c {0};
        size_t t {0};
        const sym_t SPACE;

    public:
        trie_histogram_restaurant(size_t _BOS, size_t _EOS, size_t _SPACE)
            : trie(_EOS), SPACE(_SPACE) {}

        size_t get_c(const seq_t& seq) const override {
            return trie.get_val(seq).cw;
        }

        size_t get_c() const override {
            return c;
        }

        size_t get_t(const seq_t& seq) const override {
            return trie.get_val(seq).tw;
        }

        size_t get_t() const override {
            return t;
        }

        size_t get_num_types() const {
            return trie.num_keys();
        }

        sym_t get_space() const {
            return SPACE;
        }

        bool new_type(const seq_t& type) const {
            return trie.has_key(type);
        }

        std::vector<seq_t> starts_with(const seq_t& prefix) const {
            auto rs = trie.starts_with(prefix);
            std::vector<seq_t> ret;
            for(auto r : rs) {
                ret.push_back(r.first);
            }
            return ret;
        }

        double prob(const seq_t& type, double p0, double discount, double alpha) const override {
            if(!trie.has_key(type)) {
                return computeHPYPPredictive(0, 0, c, t, p0, discount, alpha);
            }
            auto& a = trie.get_val(type);
            return computeHPYPPredictive(a.cw, a.tw, c, t, p0, discount, alpha);
        }

        double log_prob(const seq_t& type, double log_p0, double discount, double alpha) const override {
            if(!trie.has_key(type)) {
                return computeLogHPYPPredictive(0, 0, c, t, log_p0, discount, alpha);
            }
            auto& a = trie.get_val(type);
            return computeLogHPYPPredictive(a.cw, a.tw, c, t, log_p0, discount, alpha);
        }

        double log_new_prob(double log_p0, double discount, double alpha) const override {
            return computeHPYPLogNewProb(c, t, log_p0, discount, alpha);
        }

        double log_cache_prob(const seq_t& type, double discount, double alpha) const override {
            if(!trie.has_key(type)) {
                return computeHPYPLogCachedProb(0, 0, c, discount, alpha);
            }
            auto& a = trie.get_val(type);
            return computeHPYPLogCachedProb(a.cw, a.tw, c, discount, alpha);
        }

        std::vector<std::pair<seq_t, double>> log_prob_cache_matching(const seq_t& seq, double discount, double alpha) const {
            std::vector<std::pair<seq_t, double>> ret;
            for(auto r : trie.starts_with(seq)) {
                LOG(INFO) << r.second.cw << ", " << r.second.tw << ", " << c << ", d=" << discount << " , a=" << alpha;
                ret.emplace_back(r.first, computeHPYPLogCachedProb(r.second.cw, r.second.tw, c, discount, alpha) );
            }
            return ret;
        }

        bool add(const seq_t& type, double log_p0, double discount, double alpha) override {
            auto& a = trie.get_or_insert_val(type);
            a.cw += 1;
            c += 1;
            if (a.cw == 1) {
                // first customer sits at the first table
                // this special case is needed as otherwise things will break when alpha=0
                // and the restaurant has 0 customers in the singleton bucket
                a.histogram[1] += 1;
                a.tw += 1;
                t += 1;
                return true;
            }

            int numBuckets = a.histogram.size();
            d_vec tableProbs(numBuckets + 1, 0);
            std::vector<int> assignment(numBuckets,0);
            int i = 0;
            for(auto it = a.histogram.begin(); it != a.histogram.end(); ++it) {
                // prob for joining a table of size k: \propto (k - d)t[k]
                tableProbs[i] = log(((*it).first - discount) * (*it).second);
                assignment[i] = (*it).first;
                ++i;
            }

            // prob for new table: \propto (alpha + d*t)*P0
            // this can be 0 for the first customer if concentration=0, but that is ok
            tableProbs[numBuckets] = log(alpha + discount*t) + log_p0;

            // choose table for customer to sit at
            int sample = nn::sample_unnormalized_lnpdf(tableProbs, nn::rng::get());
            assert(sample <= (int)numBuckets);

            if(sample == (int)numBuckets) {
                // sit at new table
                a.histogram[1] += 1;
                a.tw += 1;
                t += 1;
                return true;
          } else {
                // existing table
                a.histogram[assignment[sample]] -= 1;
                if (a.histogram[assignment[sample]] == 0) {
                    // delete empty bucket from histogram
                    a.histogram.erase(assignment[sample]);
                }
                a.histogram[assignment[sample]+1] += 1;
                return false;
            }
        }

        bool remove(const seq_t& type, double discount, double alpha) override {
            auto& a = trie.get_or_insert_val(type);

            a.cw -= 1;
            c -= 1;

            assert(a.cw >= 0);
            assert(c >= 0);

            int numBuckets = a.histogram.size();
            int singletonBucket = -1; // invalid bucket
            d_vec tableProbs(numBuckets, 0);
            std::vector<int> assignment(numBuckets,0);

            int i = 0;
            for(auto it = a.histogram.begin();
                it != a.histogram.end();
                ++it) {
                // prob for choosing a bucket k*t[k]
              tableProbs[i] = (*it).first * (*it).second;
              assignment[i] = (*it).first;
              if ((*it).first == 1) {
                  singletonBucket = i;
              }
              ++i;
            }

            // choose table for customer to sit at
            int sample = nn::sample_unnormalized_pdf(tableProbs, nn::rng::get());
            assert(sample <= (int)numBuckets);
            assert(tableProbs[sample] > 0);

            if (sample == singletonBucket) {
                assert(a.histogram[1] > 0);
                // singleton -> drop table
                a.histogram[1] -= 1;
                if (a.histogram[1] == 0) {
                  // delete empty bucket from histogram
                  //arrangement.histogram.erase(1);
                }
                a.tw -= 1;
                t -= 1;

                assert(a.tw >= 0);
                assert(t >= 0);

                return true;
            } else {
                // non-singleton bucket
                a.histogram[assignment[sample]] -= 1;
                assert(a.histogram[assignment[sample]] >= 0);
                if (a.histogram[assignment[sample]] == 0) {
                    // delete empty bucket from histogram
                    a.histogram.erase(assignment[sample]);
                }
                a.histogram[assignment[sample]-1] += 1;
              return false;
            }
        }
    };
};

#endif
