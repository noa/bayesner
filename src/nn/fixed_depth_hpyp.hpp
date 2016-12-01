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

#ifndef __NN_FIXED_DEPTH_HPYP_HPP__
#define __NN_FIXED_DEPTH_HPYP_HPP__

#include <type_traits>
#include <memory>
#include <iostream>
#include <unordered_map>
#include <stack>
#include <functional>

#include <boost/range/adaptor/reversed.hpp>
#include <boost/utility/string_ref.hpp>

#include <nn/log.hpp>
#include <nn/discrete_distribution.hpp>
#include <nn/node.hpp>
#include <nn/restaurants.hpp>

#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>

namespace nn {

template<typename T,
         typename C,
         typename BaseMeasure,
         size_t MAX_DEPTH = 10,
         typename Restaurant = SimpleFullRestaurant<T>
         >
struct FixedDepthHPYP {
    static_assert(MAX_DEPTH > 4, "must use max depth at least 4");

    // Typedefs
    typedef std::vector<C> Context;
    typedef hash_node<C, Restaurant> Node;

    // Work storage
    // +1 for base distribution; note: 0 index is never used
    std::array<Node*, MAX_DEPTH+1> node_storage;
    size_t node_storage_size; // contexts might be len < MAX_DEPTH

    // +1 for the top-level base distribution
    std::array<double, MAX_DEPTH+1> prob_storage;
    size_t prob_storage_size; // contexts might be len < MAX_DEPTH

    // Parameters
    constexpr static double default_discount {0.75};
    constexpr static double default_alpha    {1.0};

    std::array<double, MAX_DEPTH+1> discounts;
    std::array<double, MAX_DEPTH+1> alphas;

    // Restaurant type
    Restaurant restaurant;

    // Base distribution of the HPYP
    std::shared_ptr<BaseMeasure> H;

    // Unique root of context tree
    std::unique_ptr<Node> root;

    // Diagnostics
    size_t total_n_customers {0};
    size_t total_n_tables    {0};
    bool init_bit            {false};

    FixedDepthHPYP() {}
    FixedDepthHPYP(std::shared_ptr<BaseMeasure> _H, double alpha)
        : H(_H) {
        CHECK(H->cardinality() > 0) << "cardinality <= 0";
        CHECK(H->cardinality() < 100000) << "very large cardinality: "
                                         << H->cardinality();
        init_bit = true;
        CHECK(alpha > 0)              << "alpha = "    << alpha;
        CHECK(alpha < 10000)          << "alpha = "    << alpha;
        CHECK(default_discount < 1.0) << "discount = " << default_discount;
        size_t d;
        discounts[0] = -1.0;
        discounts[1] = 0.62;
        discounts[2] = 0.69;
        discounts[3] = 0.74;
        discounts[4] = 0.80;
        d = 5;
        while(d <= MAX_DEPTH) {
            discounts[d] = default_discount;
            d ++;
        }
        alphas[0] = -1.0;
        alphas[1] = alpha;
        d = 2;
        while(d <= MAX_DEPTH) {
            alphas[d] = alphas[d-1] * discounts[d-1];
            d++;
        }
        root = std::make_unique<Node>();
    }

    FixedDepthHPYP(std::shared_ptr<BaseMeasure> _H)
        : FixedDepthHPYP(_H, default_alpha) {}

    // Forbid copy constructor and assignment
    FixedDepthHPYP(FixedDepthHPYP const&)            = delete;
    FixedDepthHPYP& operator=(FixedDepthHPYP const&) = delete;

    // Retrieve the root
    Node* getRoot() const { return root.get(); }

    double pred(Node* n, T type, double p0, double d, double alpha) const {
        return restaurant.computeProbability(n->get_payload(), type, p0, d, alpha);
    }

    size_t totalCustomers() const { return total_n_customers; }
    size_t totalTables()    const { return total_n_tables;    }

    size_t rootCustomers() const {
        auto root = getRoot();
        return restaurant.getC(root->get_payload());
    }

    size_t rootTables() const {
        auto root = getRoot();
        return restaurant.getT(root->get_payload());
    }

    void fill_node_array(typename Context::const_iterator start,
                         typename Context::const_iterator stop) {
        // depth 0 is a fill-in for the base distribution; no node is
        // associated with it
        size_t depth = 1;
        Node* node = getRoot();
        node_storage[depth++] = node; // set the root node at depth=1
        if (MAX_DEPTH == 1 || start == stop) {
            node_storage_size = depth;
            return;
        }
        auto iter = stop;
        --iter;
        while (true) {
            node = node->get_or_make(*iter);
            node_storage[depth++] = node;
            if (depth == MAX_DEPTH || iter == start) {
                node_storage_size = depth;
                return;
            }
            --iter;
        };
        node_storage_size = depth;
    }

    void debug_print_restaurants(typename Context::const_iterator start,
                                 typename Context::const_iterator stop,
                                 T obs
        ) {
        fill_node_array(start, stop);
        fill_prob_array(obs);
        LOG(INFO) << "Base prob = " << prob_storage[0];
        for(size_t d = 1; d < node_storage_size; ++d) {
            auto node = node_storage[d];
            auto consistent = restaurant.checkConsistency(node->get_payload());
            CHECK(consistent) << "bad restaurant";
            auto prob = prob_storage[d];
            auto c = restaurant.getC(node->get_payload());
            auto t = restaurant.getT(node->get_payload());
            LOG(INFO) << "c=" << c << " t=" << t << " pr=" << prob;
            auto types = restaurant.getTypeVector(node->get_payload());
            for(auto t : types) {
                auto cw = restaurant.getC(node->get_payload(), t);
                auto tw = restaurant.getT(node->get_payload(), t);
                LOG(INFO) << "\t" << t << " cw=" << cw << " tw=" << tw;
            }
        }
    }

    void fill_prob_array(T obs) {
        prob_storage[0] = H->prob(obs);
        size_t depth = 1;
        while(depth < node_storage_size) {
            CHECK(node_storage.at(depth) != nullptr);
            prob_storage[depth] = pred(node_storage.at(depth), obs,
                                       prob_storage[depth-1],
                                       discounts.at(depth), alphas.at(depth));
            ++depth;
        }
    }

    void observe(typename Context::const_iterator start,
                 typename Context::const_iterator stop,
                 T obs) {
        fill_node_array(start, stop);
        fill_prob_array(obs);
        size_t depth = node_storage_size - 1;
        bool new_table;
        do {
            new_table = restaurant.addCustomer(
                                               node_storage.at(depth)->get_payload(),
                                               obs,
                                               prob_storage.at(depth-1),
                                               discounts.at(depth),
                                               alphas.at(depth)
                                               );
            if (new_table) total_n_tables ++;
            depth --;
        } while(depth > 0 && new_table);
        total_n_customers ++;
    }

    void observe(const Context& prefix, T obs) {
        observe(prefix.begin(), prefix.end(), obs);
    }

    void debug_print_restaurants(const Context& prefix, T obs) {
        debug_print_restaurants(prefix.begin(), prefix.end(), obs);
    }

    void remove(typename Context::const_iterator start,
                typename Context::const_iterator stop,
                T obs) {
        fill_node_array(start, stop);
        bool removed_table;
        size_t depth = node_storage_size - 1;
        do {
            auto node = node_storage[depth];
            auto d = discounts.at(depth);
            auto a = alphas.at(depth);
            removed_table = restaurant.removeCustomer(node->get_payload(),
                                                      obs,
                                                      d);
            depth --;
            if (removed_table) total_n_tables --;
        } while(depth > 0 && removed_table);
        total_n_customers --;
    }

    void remove(const Context& prefix, T obs) {
        remove(prefix.begin(), prefix.end(), obs);
    }

    double prob(typename Context::const_iterator start,
                typename Context::const_iterator stop,
                T obs)
        const {
        double p = H->prob(obs);
        Node* node = root.get();
        size_t depth = 1;
        p = pred(node, obs, p, discounts.at(depth), alphas.at(depth));
        if(start == stop) {
            return p;
        }
        depth++;
        auto riter = stop;
        --riter;
        while(true) {
            if (node != nullptr) node = node->get_or_null(*riter);
            if (node == nullptr) {
                p = computeHPYPPredictive(0, 0, 0, 0, p,
                                          discounts.at(depth),
                                          alphas.at(depth));
            } else {
                p = restaurant.computeProbability(node->get_payload(),
                                                  obs,
                                                  p,
                                                  discounts.at(depth),
                                                  alphas.at(depth));
            }
            ++depth;
            if (depth == MAX_DEPTH) break;
            if (riter == start) break;
            --riter;
        }
        return p;
    }

    double log_prob(typename Context::const_iterator start,
                    typename Context::const_iterator stop,
                    T obs)
        const {
        CHECK(start != stop) << "bad iterator";
        return log(prob(start, stop, obs));
    }

    double prob(const Context& prefix, T obs) const {
        CHECK(prefix.size() > 0);
        return prob(prefix.begin(), prefix.end(), obs);
    }

    double log_new_prob(const Context& prefix, T obs) const {
        CHECK(false) << "TODO";
        return 0;
    }

    double log_cache_prob(const Context& prefix, T obs) const {
        CHECK(false) << "TODO";
        return 0;
    }

    double log_prob(const Context& prefix, T obs) const {
        CHECK(prefix.size() > 0);
        return log(prob(prefix, obs));
    }

    size_t cardinality() const {
        return H->cardinality();
    }

    template<class Archive>
    void serialize(Archive & archive) {
        archive( discounts, alphas, H, root);
    }
};

}

#endif
