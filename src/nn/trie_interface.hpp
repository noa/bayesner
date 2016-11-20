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

#ifndef __NN_TRIE_INTERFACE__
#define __NN_TRIE_INTERFACE__

#include <vector>

namespace nn {

    template<typename K, typename V>
    class trie_interface {
        typedef std::vector<K> seq_t;
        typedef std::pair<seq_t,V> result_t;
        typedef std::vector<result_t> results_t;

    public:
        virtual ~trie_interface() {}

        virtual const V& get_val(const seq_t& key) const = 0;
        virtual V& get_or_insert_val(const seq_t& key) = 0;
        virtual bool insert(const seq_t& key, V val) = 0;
        virtual bool has_key(const seq_t& key) const = 0;
        virtual results_t starts_with(const seq_t& prefix) const = 0;
        virtual size_t num_keys() const = 0;
    };

};

#endif
