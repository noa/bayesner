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

#ifndef __NN_LOG_FIXED_DEPTH_HPYP_HPP__
#define __NN_LOG_FIXED_DEPTH_HPYP_HPP__

#include <type_traits>
#include <memory>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <stack>
#include <functional>

#include <boost/range/adaptor/reversed.hpp>
#include <boost/utility/string_ref.hpp>

#include <nn/log.hpp>
#include <nn/discrete_distribution.hpp>
#include <nn/node.hpp>
#include <nn/restaurants.hpp>

namespace nn {

template<typename T,
         typename C,
         typename BaseMeasure,
         size_t MAX_DEPTH = 10,
         typename Restaurant = SimpleFullRestaurant<T>
         >
struct LogFixedDepthHPYP {
    static_assert(MAX_DEPTH > 4, "must use max depth at least 4");

    // Typedefs
    typedef std::vector<C>           Context;
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
    BaseMeasure *H;

    // Unique root of context tree
    std::unique_ptr<Node> root;

    // List of all nodes
    std::vector<Node*> nodes;

    // Diagnostics
    size_t total_n_customers {0};
    size_t total_n_tables    {0};
    bool init_bit {false};

    LogFixedDepthHPYP(BaseMeasure* _H, double alpha)
        : H(_H) {

        // CHECK(H->cardinality() > 0) << "cardinality <= 0";
        // CHECK(H->cardinality() < 100000) << "very large cardinality: "
        //                                  << H->cardinality();

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

        // Make the root node
        root = std::make_unique<Node>();
    }

    LogFixedDepthHPYP(BaseMeasure* _H)
        : LogFixedDepthHPYP(_H, default_alpha) {}

    // Forbid copy constructor and assignment
    LogFixedDepthHPYP(LogFixedDepthHPYP const&) = delete;
    LogFixedDepthHPYP& operator=(LogFixedDepthHPYP const&) = delete;

    // Retrieve the root
    Node* getRoot() const { return root.get(); }

    double log_pred(Node* n, T type, double log_p0, double d, double alpha)
        const {
        CHECK(d < 1 && d >= 0);
        CHECK(init_bit) << "not initialized";
        CHECK(alpha > -d) << "alpha = " << alpha << " d = " << d;
        return restaurant.computeLogProbability(n->crp, type, log_p0, d, alpha);
    }

    size_t totalCustomers() const { return total_n_customers; }
    size_t totalTables()    const { return total_n_tables;    }

    size_t rootCustomers() const {
      auto root = getRoot();
      return restaurant.getC(root->crp);
    }

    size_t rootTables() const {
      auto root = getRoot();
      return restaurant.getT(root->crp);
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
                               T obs) {
    fill_node_array(start, stop);
    fill_prob_array(obs);
    LOG(INFO) << "Base prob = " << prob_storage[0];
    for(size_t d = 1; d < node_storage_size; ++d) {
      auto node = node_storage[d];
      auto consistent = restaurant.checkConsistency(node->crp);
      CHECK(consistent) << "bad restaurant";
      auto prob = prob_storage[d];
      auto c = restaurant.getC(node->crp);
      auto t = restaurant.getT(node->crp);
      LOG(INFO) << "c=" << c << " t=" << t << " pr=" << prob;
      auto types = restaurant.getTypeVector(node->crp);
      for(auto t : types) {
        auto cw = restaurant.getC(node->crp, t);
        auto tw = restaurant.getT(node->crp, t);
        LOG(INFO) << "\t" << t << " cw=" << cw << " tw=" << tw;
      }
    }
  }

  void fill_prob_array(T obs) {
    prob_storage[0] = H->log_prob(obs);
    size_t depth = 1;
    while (depth < node_storage_size) {
      CHECK(node_storage.at(depth) != nullptr);
      prob_storage[depth] = log_pred(node_storage.at(depth),
                                     obs,
                                     prob_storage[depth-1],
                                     discounts.at(depth),
                                     alphas.at(depth));
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
      assert( node_storage.at(depth) != nullptr );
      new_table = restaurant.logAddCustomer(
                                            node_storage.at(depth)->crp,
                                            obs,
                                            prob_storage.at(depth-1),
                                            discounts.at(depth),
                                            alphas.at(depth)
                                            );
      if (new_table) total_n_tables ++;
      //LOG(INFO) << "depth " << depth << " new table " << new_table;
      depth --;
    } while(depth > 0 && new_table);
    total_n_customers ++;
    if (new_table) H->observe(obs);
    //if (new_table) LOG(INFO) << "final add table!";

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
      removed_table = restaurant.removeCustomer(node->crp,
                                                obs,
                                                d);
      depth --;
      if (removed_table) total_n_tables --;
    } while(depth > 0 && removed_table);
    if (removed_table) H->remove(obs);
    total_n_customers --;
  }

  void remove(const Context& prefix, T obs) {
    remove(prefix.begin(), prefix.end(), obs);
  }

  double prob(typename Context::const_iterator start,
              typename Context::const_iterator stop,
              T obs)
    const {
    return exp(log_prob(start,stop,obs));
  }

  double log_new_prob(const Context& prefix, T obs) const {
    CHECK(false) << "TODO";
    return 0;
  }

  double log_cache_prob(const Context& prefix, T obs) const {
    CHECK(false) << "TODO";
    return 0;
  }

  double log_prob(const Context& context, T obs) const {
    return log_prob(context.begin(), context.end(), obs);
  }

  double log_prob(typename Context::const_iterator start,
                  typename Context::const_iterator stop,
                  T obs)
    const {
    double log_p = H->log_prob(obs);
    Node* node = root.get();
    size_t depth = 1;
    log_p = log_pred(node,
                     obs,
                     log_p,
                     discounts.at(depth),
                     alphas.at(depth));
    if (start == stop) {
      return log_p;
    }
    depth++;
    auto riter = stop;
    --riter;
    while(true) {
      if (node != nullptr) node = node->get_or_null(*riter);
      if (node == nullptr) {
        log_p = computeLogHPYPPredictive(0, 0, 0, 0, log_p,
                                         discounts.at(depth),
                                         alphas.at(depth));
      } else {
        log_p = restaurant.computeLogProbability(node->crp, obs, log_p,
                                                 discounts.at(depth),
                                                 alphas.at(depth));
      }
      ++depth;
      if (depth == MAX_DEPTH) break;
      if (riter == start) break;
      --riter;
    }
    return log_p;
  }
};

}

#endif
