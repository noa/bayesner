// Nicholas Andrews

#ifndef _TRAINABLE_FIXED_DEPTH_HPYP_HPP_
#define _TRAINABLE_FIXED_DEPTH_HPYP_HPP_

#include <type_traits>
#include <memory>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <stack>
#include <functional>
#include <mutex>
#include <future>

#include "glog/logging.h"

#include <boost/range/adaptor/reversed.hpp>
#include <boost/utility/string_ref.hpp>

#include "discrete_distribution.hpp"
#include "node.hpp"
#include "restaurants.hpp"
#include "crp.hpp"

namespace nn {

template<typename T,
         typename C,
         typename BaseMeasure,
         size_t MAX_DEPTH = 10,
         typename Restaurant = cpyp::crp<T>>
struct TrainableFixedDepthHPYP {
    static_assert(std::is_integral<T>::value, "non-integral symbol type");
    static_assert(MAX_DEPTH > 4, "must use max depth at least 4");

    // Typedefs
    typedef std::vector<C> Context;
    typedef trainable_node<C, Restaurant> Node;

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

    TrainableFixedDepthHPYP(BaseMeasure* _H, double alpha)
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

        // LOG(INFO) << ""
        // for(d=0; d<=MAX_DEPTH; ++d) {
        //     LOG(500) << d << " " << discounts[d] << " " << alphas[d];
        // }
        // LOG(INFO) << "Alphabet size = " << H->cardinality();

        // Make the root node
        root = std::make_unique<Node>(0.5, 0.5);
        //nodes.push_back(root.get());
    }

    TrainableFixedDepthHPYP(BaseMeasure* _H)
        : TrainableFixedDepthHPYP(_H, default_alpha) {}

    // Forbid copy constructor and assignment
    TrainableFixedDepthHPYP(TrainableFixedDepthHPYP const&)            = delete;
    TrainableFixedDepthHPYP& operator=(TrainableFixedDepthHPYP const&) = delete;

    // Retrieve the root
    Node* getRoot() const { return root.get(); }

    size_t totalCustomers() const { return total_n_customers; }
    size_t totalTables()    const { return total_n_tables;    }

    size_t rootCustomers() const {
        auto root = getRoot();
        //return restaurant.getC(root->crp);
        return root->getC();
    }

    size_t rootTables() const {
        auto root = getRoot();
        //return restaurant.getT(root->crp);
        return root->getT();
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
            if(!node->has(*iter)) {
                node = node->make(*iter,
                                  discounts.at(depth),
                                  alphas.at(depth));
                nodes.push_back(node);
            } else {
                node = node->get(*iter);
            }
            node_storage[depth++] = node;
            if (depth == MAX_DEPTH || iter == start) {
                node_storage_size = depth;
                return;
            }
            --iter;
        };
        node_storage_size = depth;
    }

    void fill_prob_array(T obs) {
        prob_storage[0] = H->prob(obs);
        size_t depth = 1;
        while(depth < node_storage_size) {
            CHECK(node_storage.at(depth) != nullptr);
            prob_storage[depth] = node_storage.at(depth)->prob(
                obs,
                prob_storage.at(depth-1)
                );
            ++depth;
        }
    }

    void resample_hyperparameters() {
#pragma omp parallel for
        for (size_t m=0; m < nodes.size(); ++m) {
            nodes.at(m)->optimize(rng::get());
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
            CHECK( node_storage.at(depth) != nullptr );
            new_table = node_storage.at(depth)->add_customer(
                obs,
                prob_storage.at(depth-1),
                rng::get()
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
            removed_table = node->remove_customer(obs, rng::get());
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
        //p = pred(node, obs, p, discounts.at(depth), alphas.at(depth));
        p = node->prob(obs, p);
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
                // p = restaurant.computeProbability(node->crp, obs, p,
                //                                   discounts.at(depth),
                //                                   alphas.at(depth));
                p = node->prob(obs, p);
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

    double log_prob(const Context& prefix, T obs) const {
        CHECK(prefix.size() > 0);
        return log(prob(prefix, obs));
    }

    // discrete_distribution<T> dist(const Context& prefix) const {
    //     discrete_distribution<T> ret;
    //     for(size_t t = 0; t < H->cardinality(); ++t) {
    //         ret.push_back_prob( t, prob(prefix, t) );
    //     }
    //     return ret;
    // }

    size_t cardinality() const {
        return H->cardinality();
    }
};

}

#endif
