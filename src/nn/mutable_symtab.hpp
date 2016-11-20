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

#ifndef __NN_MUTABLE_SYMTAB_HPP__
#define __NN_MUTABLE_SYMTAB_HPP__

#include <string>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>

#include <nn/log.hpp>
#include <nn/rng.hpp>

namespace nn {

    template<typename K = size_t, typename V = std::string>
    class mutable_symbol_table {
        std::unordered_map<K, V> symtab;
        std::unordered_map<V, K> inv_symtab;
        std::unordered_set<K>    key_set;
        bool frozen {false};

    public:
        void freeze()              { frozen=true;                  }
        size_t size()   const      { return symtab.size();         }
        bool has(K key) const      { return symtab.count(key);     }
        bool hasValue(V val) const { return inv_symtab.count(val); }

        void put(K key, V val) {
            symtab[key] = val;
            inv_symtab[val] = key;
        }

        template<typename T>
        void put_all(typename T::const_iterator begin,
                     typename T::const_iterator end) {
            for(auto it = begin; it != end; ++it) {
                symtab[it->first] = it->second;
                inv_symtab[it->second] = it->first;
            }
        }

        const V& val(K key) const { return symtab.at(key);     }
        const K& key(V val) const { return inv_symtab.at(val); }

        K add_key(V val) {
            CHECK(!frozen) << "trying to add symbol to frozen map: " << val;
            CHECK(inv_symtab.count(val) < 1) << "trying to add existing value";
            K key = this->size();
            inv_symtab[val] = key;
            symtab[key] = val;
            key_set.insert(key);
            return key;
        }

        bool has_key(V val) { return inv_symtab.count(val) > 0; }

        K get_or_add_key(V val) {
            if ( inv_symtab.count(val) ) {
                return inv_symtab[val];
            }
            return add_key(val);
        }

        std::unordered_map<K, V> get_map()     const { return symtab;     }
        std::unordered_map<V, K> get_inv_map() const { return inv_symtab; }

        const std::unordered_set<K>& get_key_set() const {
            return key_set;
        }
    };

    typedef mutable_symbol_table<size_t, std::string> uint_str_table;

    std::vector<size_t> wrap(const std::vector<size_t>& w,
                             const uint_str_table& symtab,
                             size_t BOS, size_t EOS) {
        std::vector<size_t> ret;
        ret.push_back(BOS);
        for(const auto& c : w) {
            ret.push_back(c);
        }
        ret.push_back(EOS);
        return ret;
    }
}

#endif
