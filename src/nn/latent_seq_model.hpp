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

#ifndef __NN_LATENT_SEQ_MODEL_HPP__
#define __NN_LATENT_SEQ_MODEL_HPP__

#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <vector>

#include <nn/uniform.hpp>
#include <nn/prefix_matcher.hpp>

namespace nn {

    template<typename sym_t, typename base_t, typename dist_t>
    class LatentSequenceModel {
        base_t H;
        std::unordered_map<size_t, std::unique_ptr<dist_t>> E;
        std::unordered_map<size_t, size_t> counts;
        typedef std::vector<sym_t> seq_t;

        typedef PrefixMap<size_t,size_t> matcher_t;
        std::unordered_map<size_t, matcher_t> matchers;

    public:
        LatentSequenceModel(size_t nsym,
                            std::unordered_set<size_t> types) : H(nsym) {
            for (auto t : types) {
                auto e = std::make_unique<dist_t>(&H);
                E[t] = std::move(e);
                matchers.emplace(t, matcher_t());
            }
        }

      double log_new_prob(size_t t, const seq_t& context, sym_t obs) const {
        CHECK(false) << "TODO";
        return 0.0;
      }

      double log_cache_prob(size_t t, const seq_t& context, sym_t obs) const {
        CHECK(false) << "TODO";
        return 0.0;
      }

      double log_prob(size_t t, const seq_t& context, sym_t obs) const {
            return E.at(t)->log_prob(context, obs);
        }

        double log_prob(size_t t, const seq_t& seq) const {
            double ret = 0.0;
            auto start = seq.begin();
            auto end = seq.end();
            auto model = E.at(t).get();
            for(auto it = std::next(start); it != end; ++it) {
                ret += model->log_prob(start, it, *it);
            }
            return ret;
        }

        std::pair<typename std::map<seq_t, size_t>::const_iterator,
                  typename std::map<seq_t, size_t>::const_iterator>
        match(size_t t, const seq_t& seq) {
            return matchers.at(t).match_prefix(seq);
        }

        double log_prob(const obs_t& obs) const {
            double ret = 0.0;
            auto start = obs.second.begin();
            auto end = obs.second.end();
            auto model = E.at(obs.first).get();
            for(auto it = std::next(start); it != end; ++it) {
                ret += model->log_prob(start, it, *it);
            }
            return ret;
        }

        void observe(const obs_t& obs) {
            auto t = obs.first;
            auto start = obs.second.begin();
            auto end = obs.second.end();
            auto model = E.at(obs.first).get();
            for(auto it = std::next(start); it != end; ++it) {
                model->observe(start, it, *it);
            }
            matchers.at(obs.first).add(obs.second, 0);
            if(counts.count(t)) {
                ++ counts[t];
            } else {
                counts[t] = 1;
            }
        }

        void remove(const obs_t& obs) {
            auto t = obs.first;
            auto start = obs.second.begin();
            auto end = obs.second.end();
            auto model = E.at(obs.first).get();
            for(auto it = std::next(start); it != end; ++it) {
                model->remove(start, it, *it);
            }
            matchers.at(obs.first).remove(obs.second);
            -- counts[t];
        }

        void log_stats() const {
            LOG(INFO) << "Type counts:";
            for(auto kv : counts) {
                LOG(INFO) << "\t" << kv.first << " : " << kv.second;
            }
        }
    };
}

#endif
