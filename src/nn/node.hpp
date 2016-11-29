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

#ifndef __NN_NODE_HPP__
#define __NN_NODE_HPP__

#include <unordered_map>
#include <vector>
#include <memory>

#include <nn/log.hpp>
#include <nn/utils.hpp>

#include <cereal/types/unordered_map.hpp>

namespace nn {

    template<typename T, typename R>
    struct trainable_node {
        R r;
        std::unordered_map<T, std::unique_ptr<trainable_node>> kids;

        static constexpr double d_str   {1.0};
        static constexpr double d_beta  {1.0};
        static constexpr double c_shape {10.0};
        static constexpr double c_rate  {0.1};

        trainable_node(double d, double a)
            : r(d_str, d_beta, c_shape, c_rate, d, a) {};

        trainable_node(trainable_node const&)            = delete;
        trainable_node& operator=(trainable_node const&) = delete;
        ~trainable_node() {}

        trainable_node* get_or_null(T t) const {
            if(kids.count(t) > 0) return kids.at(t).get();
            return nullptr;
        }

        trainable_node* get(T t) const {
            return kids.at(t).get();
        }

        trainable_node* make(T t, double d, double a) {
            kids[t] = std::make_unique<trainable_node>(d, a);
            return kids[t].get();
        }

        trainable_node* get_or_make(T t, double d, double a) {
            if(kids.count(t) == 0) { return make(t,d,a); }
            return get(t);
        }

        template<typename RNG>
        void optimize(RNG& rng) {
            r.resample_hyperparameters(rng);
        }

        bool has(T t) const { return kids.count(t) > 0; }

        double prob(T t, double p0) const { return r.prob(t, p0); }

        template<typename Engine>
        bool add_customer(T t, double p0, Engine& eng) {
            auto delta = r.increment(t, p0, eng);
            if (delta) return true;
            return false;
        }

        template<typename Engine>
        bool remove_customer(T t, Engine& eng) {
            auto delta = r.decrement(t, eng);
            if (delta < 0) return true;
            return false;
        }

        size_t getC()    const { return r.num_customers();  };
        size_t getC(T t) const { return r.num_customers(t); };
        size_t getT()    const { return r.num_tables();     };
        size_t getT(T t) const { return r.num_tables(t);    };

        std::string str(const T& t) {
            std::stringstream ss;
            ss << "c: "   << getC();
            ss << " cw: " << getC(t);
            ss << " t: "  << getT();
            ss << " tw: " << getT(t);
            return ss.str();
        }

        std::vector<T> getTypeVector() const {
            std::vector<T> ret;
            for(auto it = r.begin(); it != r.end(); ++it) {
                ret.push_back(it->first);
            }
            return ret;
        }
    };

    template<typename T, typename R>
    struct hash_node {
        R r;
        std::unordered_map<T, std::unique_ptr<hash_node>> kids;

        void* crp  {nullptr};   // CRP sufficient statistics
        void* data {nullptr};   // additional data required by the CRP

        hash_node() {
            this->crp = r.getFactory().make();
        }
        hash_node(hash_node const&) = delete;
        hash_node& operator=(hash_node const&) = delete;
        ~hash_node() { r.getFactory().recycle(crp); }

        hash_node* get_or_null(T t) const {
            if(kids.count(t) > 0) {
                return kids.at(t).get();
            }
            return nullptr;
        }

        hash_node* get(T t) const {
            return kids.at(t).get();
        }

        hash_node* make(T t) {
            kids[t] = std::make_unique<hash_node>();
            return kids[t].get();
        }

        hash_node* get_or_make(T t) {
            if(kids.count(t) == 0) {
                return make(t);
            }
            return get(t);
        }

        size_t getC()    const { return r.getC(crp);    };
        size_t getC(T t) const { return r.getC(crp, t); };
        size_t getT()    const { return r.getT(crp);    };
        size_t getT(T t) const { return r.getT(crp, t); };

        std::string str(const T& t) {
            std::stringstream ss;
            ss << "c: "   << r.getC(crp);
            ss << " cw: " << r.getC(crp, t);
            ss << " t: "  << r.getT(crp);
            ss << " tw: " << r.getT(crp, t);
            return ss.str();
        }

        std::vector<T> getTypeVector() const {
            std::vector<T> ret;
            for(auto t : r.getTypeVector(crp)) {
                if(r.getC(crp, t) > 0) {
                    ret.push_back(t);
                }
            }
            return ret;
        }
    };

    template<typename T, typename R>
    struct vector_node {
        R r;
        std::vector<std::unique_ptr<vector_node>> kids;

        void* crp  {nullptr};   // CRP sufficient statistics
        void* data {nullptr};   // additional data required by the CRP

        vector_node(size_t size) {
            this->crp = r.getFactory().make();
            this->kids.resize(size, nullptr); // inserts elements until there are size of them
        }

        ~vector_node() { r.getFactory().recycle(crp); }

        vector_node* get(T t) const {
            assert(t < kids.size());
            assert(t >= 0);
            return kids.at(t).get();
        }

        vector_node* make(T t) {
            assert(t < kids.size());
            assert(t >= 0);
            //kids[t] = std::make_unique<vector_node>(r);
            kids[t] = std::unique_ptr<vector_node>(new vector_node(r));
            return kids[t].get();
        }

        vector_node* get_or_make(T t) {
            DLOG(INFO) << "t=" << t << " kids size = " << kids.size();
            assert(t < kids.size());
            if(kids[t] == nullptr) {
                return make(t);
            }
            return get(t);
        }

        size_t getC() const { return r.getC(crp); };
        size_t getC(T t) const { return r.getC(crp, t); };
        size_t getT() const { return r.getT(crp); };
        size_t getT(T t) const { return r.getT(crp, t); };

        std::string str(const T& t) {
            std::stringstream ss;
            ss << "c: " << r.getC(crp);
            ss << " cw: " << r.getC(crp, t);
            ss << " t: " << r.getT(crp);
            ss << " tw: " << r.getT(crp, t);
            return ss.str();
        }

        std::vector<T> getTypeVector() const {
            std::vector<T> ret;
            for(auto t : r.getTypeVector(crp)) {
                if(r.getC(crp, t) > 0) {
                    ret.push_back(t);
                }
            }
            return ret;
        }

        // Forbid copy constructor and assignment
        vector_node(vector_node const&) = delete;
        vector_node& operator=(vector_node const&) = delete;
    };

    template<typename C, // context
             typename E, // emission
             typename R  // restaurant
             > // emission
    struct pyp_node {
        const R& r;

        std::unordered_map<E, std::unique_ptr<pyp_node>> kids;

        bool ephemeral {false}; // used to flag temporary nodes for deletion
        void* crp  {nullptr};   // CRP sufficient statistics
        void* data {nullptr};   // additional data required by the CRP

        pyp_node(const R& _r) : r(_r) {
            crp = _r.getFactory().make();
        }

        ~pyp_node() { r.getFactory().recycle(crp); }

        pyp_node* next(E t) {
            auto iter = kids.find(t);
            if(iter != kids.end()) {
                return kids[t].get();
            }
            return nullptr;
        }

        pyp_node* constNext(E t) const {
            auto iter = kids.find(t);
            if(iter == kids.end()) {
                return nullptr;
            } else {
                return kids.at(t).get();
            }
        }

        pyp_node* make(E t) {
            //auto node = std::make_unique<pyp_node>(r);
            auto node = std::unique_ptr<pyp_node>(new pyp_node(r));
            auto ret = node.get();
            kids.emplace(t, std::move(node));
            return ret;
        }

        // WARNING: the caller is responsible for freeing the node
        pyp_node* make_ephemeral(E t) const {
            auto node = new pyp_node(r);
            node->ephemeral = true;
            return node;
        }

        size_t getC()    const  { return r.getC(crp);    };
        size_t getC(E t) const  { return r.getC(crp, t); };
        size_t getT()    const  { return r.getT(crp);    };
        size_t getT(E t) const  { return r.getT(crp, t); };

        std::string str(E t) {
            std::stringstream ss;
            ss << "c: "   << r.getC(crp);
            ss << " cw: " << r.getC(crp, t);
            ss << " t: "  << r.getT(crp);
            ss << " tw: " << r.getT(crp, t);
            return ss.str();
        }

        std::vector<E> getTypeVector() const {
            std::vector<E> ret;
            for(auto t : r.getTypeVector(crp)) {
                if(r.getC(crp, t) > 0) {
                    ret.push_back(t);
                }
            }
            return ret;
        }

        // Forbid copy constructor and assignment
        pyp_node(pyp_node const&) = delete;
        pyp_node& operator=(pyp_node const&) = delete;
    };
};

#endif
