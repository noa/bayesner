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

#ifndef __NN_INSPECTORS_HPP__
#define __NN_INSPECTORS_HPP__

#include <map>
#include <set>
#include <vector>
#include <nn/discrete_distribution.hpp>
#include <nn/sequence_sequence_model.hpp>

namespace nn {

    class state_dist_inspector {
        std::string desc;

    public:
        state_dist_inspector(std::string _desc) : desc(_desc) {
        }

        void operator()(const std::vector<gm::symbol>& type, const std::vector<gm::hseq>& state) {
            std::map<gm::symbol, swiss::histogram<gm::symbol>> dists;
            std::set<gm::symbol> ts;
            size_t i = 0;
            for(i = 0; i < type.size(); ++i) {
                auto t = type[i];
                ts.insert(t);
                for(auto s : state[i].states) {
                    dists[t].observe(s);
                }
            }
            LOG(INFO) << ts.size() << " types";
            for(auto keyval1 : dists) {
                auto t = keyval1.first;
                auto dst = keyval1.second;
                LOG(INFO) << "label " << t << ": " << dst.str();
            }
        }
    };
};

#endif
