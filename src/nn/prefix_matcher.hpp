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

#ifndef __NN_PREFIX_MAP_HPP__
#define __NN_PREFIX_MAP_HPP__

#include <algorithm>
#include <vector>
#include <set>
#include <map>
#include <utility>

namespace nn {

    template<typename T>
    bool vector_less(const std::vector<T>& lhs, const std::vector<T>& rhs) {
        return std::lexicographical_compare(lhs.begin(),
                                            lhs.end(),
                                            rhs.begin(),
                                            rhs.end());
    }

    template<typename T, typename V>
    class PrefixMap {
        typedef std::vector<T> seq_t;
        typedef std::map<seq_t,V> map_t;
        typedef std::pair<typename map_t::const_iterator,
                          typename map_t::const_iterator> ret_t;
        map_t keys;

    public:
        void add(const seq_t& key, const V& val) {
            keys[key] = val;
        }

        void remove(const seq_t& key) {
            keys.erase(key);
        }

        ret_t match_prefix(const seq_t& prefix) const {
            auto start = keys.lower_bound(prefix);
            bool match = true;
            auto it = start;
            while(it != keys.end() && match) {
                auto res = std::mismatch(prefix.begin(),
                                         prefix.end(),
                                         it->first.begin());
                if (res.first == prefix.end()) {
                    ++it;
                } else {
                    match = false;
                }
            }
            return ret_t(start, it);
        }
    };
}

#endif
