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

#ifndef __NN_DISCRETE_DISTRIBUTION_HPP__
#define __NN_DISCRETE_DISTRIBUTION_HPP__

#include <string>
#include <unordered_map>

#include <boost/algorithm/string/join.hpp>

#include <nn/utils.hpp>
#include <nn/rng.hpp>
#include <nn/stat.hpp>

namespace nn {
    template<typename T>
    class unnormalized_discrete_distribution {
        std::vector<double> ps;
        std::vector<T> ts;

    public:
        void push_back_log_prob(T t, double lp) {
            ts.push_back(t);
            ps.push_back(lp);
        }

        void push_back_prob(T t, double p) {
            ts.push_back(t);
            double lp = std::log(p);
            ps.push_back(lp);
        }

        double get_weight(size_t i) const     { return exp(ps.at(i)); }
        double get_log_weight(size_t i) const { return ps.at(i);      }

        T get_type(size_t i) const            { return ts.at(i);      }

        size_t get_index(T t) const {
            auto it = std::find(ts.begin(), ts.end(), t);
            return it - ts.begin();
        }

        std::vector<T> get_types() const { return ts;        }
        size_t size()              const { return ps.size(); }

        size_t sample_index() const {
            return sample_unnormalized_lnpdf(ps, nn::rng::get());
        }
        T sample_type() { return get_type(sample_index()); }
        typename std::vector<T>::const_iterator begin() const { return ts.begin(); }
        typename std::vector<T>::const_iterator end() const  { return ts.end();   }

        void log() const {
            for(auto i=0; i<ps.size(); ++i) {
                LOG(INFO) << i << " " << ts.at(i) << " " << ps.at(i);
            }
        }
    };

    template<typename T>
    class discrete_distribution {
        std::vector<double> ps;
        std::vector<T> ts;
        double Z {-std::numeric_limits<double>::infinity()};

    public:
        void push_back_log_prob(T t, double lp) {
            ts.push_back(t);
            ps.push_back(lp);
            log_plus_equals(Z, lp);
        }

        void push_back_prob(T t, double p) {
            ts.push_back(t);
            double lp = log(p);
            ps.push_back(lp);
            log_plus_equals(Z, lp);
        }

        double get_prob(size_t i)       const { return exp(ps.at(i) - Z); }
        double get_log_prob(size_t i)   const { return ps.at(i) - Z;      }

        double get_weight(size_t i)     const { return exp(ps.at(i));     }
        double get_log_weight(size_t i) const { return ps.at(i);          }

        double get_log_partition()      const { return Z;                 }

        T get_type(size_t i)            const { return ts.at(i);          }

        size_t get_index(T t) const {
            auto it = std::find(ts.begin(), ts.end(), t);
            return it - ts.begin();
        }

        std::vector<double> get_probs() const { return ps;        }
        std::vector<T>      get_types() const { return ts;        }
        size_t              size()      const { return ps.size(); }

        size_t sample_index() const {
            return sample_unnormalized_lnpdf(ps, nn::rng::get());
        }

        T sample_type() const {
            return get_type(sample_index());
        }

        size_t argmax() {
            size_t ret = 0;
            double max = ps[0];
            for(size_t k = 1; k < ts.size(); ++k) {
                if(ps[k] > max) {
                    ret = k;
                    max = ps[k];
                }
            }
            return ret;
        }

        typename std::vector<T>::const_iterator begin() const {
            return ts.begin();
        }

        typename std::vector<T>::const_iterator end() const {
            return ts.end();
        }

        std::string str() const {
            std::string ret;
            for(size_t i = 0; i < ps.size(); ++i) {
                ret += "p(";
                ret += std::to_string(get_type(i));
                ret += ")=";
                ret += std::to_string(get_prob(i));
                ret += " ";
            }
            return ret;
        }
    };

    template<typename T>
    class weighted_histogram {
        double total {0};
        std::map<T,double> counts;

    public:
        double prob(T t) {
            auto c = counts[t];
            return (double)c/(double)total;
        }
        void observe(T t, double c) {
            counts[t] += c;
            total += c;
        }
        void remove(T t, double c) {
            counts[t] -= c;
            total -= c;
        }
        T get_max() {
            T ret;
            double max = 0.0;
            for(const auto& keyval : counts) {
                if(keyval.second > max) {
                    ret = keyval.first;
                    max = keyval.second;
                }
            }
            return ret;
        }
        double get_total() {
            return total;
        }
        typename std::map<T,double>::const_iterator
        begin() const {
            return counts.begin();
        }
        typename std::map<T,double>::const_iterator
        end() const {
            return counts.end();
        }
        std::string str() {
            std::vector<std::string> ds;
            for(auto keyval : counts) {
                std::string s("p(");
                s.append(std::to_string(keyval.first));
                s.append(")=");
                s.append(std::to_string(prob(keyval.first)));
                ds.push_back( s );
            }
            return boost::algorithm::join(ds, " ");
        }
    };

    template<typename T>
    class histogram {
        size_t total {0};
        std::map<T,size_t> counts;

    public:
        void clear() {
            total = 0;
            counts.clear();
        }
        double prob(T t) const {
            auto c = counts.at(t);
            return static_cast<double>(c) /
                static_cast<double>(total);
        }

        size_t count(T t) const {
            if(counts.count(t) > 0) {
                return counts.at(t);
            } else {
                return 0;
            }
        }
        void observe(T t) {
            counts[t] += 1;
            total += 1;
        }
        void remove(T t) {
            counts[t] -= 1;
            total -= 1;
        }
        T get_max() {
            T ret {};
            size_t max = 0;
            for(const auto& keyval : counts) {
                if(keyval.second > max) {
                    ret = keyval.first;
                    max = keyval.second;
                }
            }
            return ret;
        }
        size_t get_total() const {
            return total;
        }
        typename std::map<T,size_t>::const_iterator
        begin() const {
            return counts.begin();
        }
        typename std::map<T,size_t>::const_iterator
        end() const {
            return counts.end();
        }
        std::string str() {
            std::vector<std::string> ds;
            for(auto keyval : counts) {
                std::stringstream ss;
                ss << "p(" << keyval.first << ")=" << prob(keyval.first);
                ds.push_back( ss.str() );
            }
            return boost::algorithm::join(ds, " ");
        }

        std::string count_str() {
            std::vector<std::string> ds;
            for(auto keyval : counts) {
                std::stringstream ss;
                ss << "c(" << keyval.first << ")=" << keyval.second;
                ds.push_back(ss.str());
            }
            return boost::algorithm::join(ds, " ");
        }
    };

    template<typename T>
    class IndexedNormalizedDiscreteDistribution {
        std::vector<double> ps;
        std::vector<T> ts;
        double Z {0.0};
    public:
        IndexedNormalizedDiscreteDistribution() {}

        void push_back(T t, double p) {
            ts.push_back(t);
            ps.push_back(p);
            Z += p;
        }

        double get_prob(unsigned i) const { return ps[i]; }

        // WARNING: this is slow and should be avoided
        double get_prob(T t) const {
            for(auto j = 0; j < ts.size(); ++j) {
                if(ts[j] == t) {
                    return ps[j];
                }
            }
            std::cerr << "should never get here" << std::endl;
            exit(1);
            return -1.0;
        }

        T get_type(unsigned i) const { return ts[i]; }

        std::size_t get_index(T t) const {
            auto it = std::find(ts.begin(), ts.end(), t);
            return it - ts.begin();
        }

        //	double get_probs() const { return ps; }
        std::vector<double> get_probs() const { return ps; }
        std::vector<T> get_types() const { return ts; }
        std::size_t size() const { return ps.size(); }

        bool check() const { if(Z > 0.99 && Z < 1.01) return true; return false; }

        int sample_index() const { return sample_normalized_pdf(ps, nn::rng::get()); }
    };

    template<typename T>
    class IndexedUnnormalizedDiscreteDistribution {
        std::vector<double> ps;
        std::vector<T> ts;

        double Z;

        const bool log_space;

    public:
        IndexedUnnormalizedDiscreteDistribution(bool _log_space) : log_space(_log_space) {
            if(log_space) {
                Z = -std::numeric_limits<double>::infinity();
            } else {
                Z = 0.0;
            }
        }

        void push_back(T t, double p) {
            ts.push_back(t);
            ps.push_back(p);
            if(log_space) {
                log_plus_equals(Z, p);
            } else {
                Z += p;
            }
        }

        double get_prob(unsigned i) const {
            if(log_space) {
                return exp(ps[i] - Z);
            } else {
                return ps[i]/Z;
            }
        }

        double get_log_prob(unsigned i) const {
            if(log_space) {
                return ps[i] - Z;
            } else {
                return log(ps[i]/Z);
            }
        }

        T get_type(unsigned i) const {
            return ts[i];
        }

        int get_index(T t) const {
            auto it = std::find(ts.begin(), ts.end(), t);
            return it - ts.begin();
        }

        std::vector<double> get_probs() const {
            return ps;
        }

        std::vector<T> get_types() const {
            return ts;
        }

        std::size_t size() const {
            return ps.size();
        }

        int sample_index() const {
            if(log_space) {
                return sample_unnormalized_lnpdf(ps, nn::rng::get());
            } else {
                return sample_unnormalized_pdf(ps, nn::rng::get());
            }
        }

        T sample_type() const {
            return get_type(sample_index());
        }

        bool is_log_space() const { return log_space; }
    };
}

#endif
