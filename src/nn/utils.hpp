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

#ifndef __NN_UTILS_HPP__
#define __NN_UTILS_HPP__

#include <string>
#include <functional>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/format.hpp>
#include <boost/functional/hash.hpp>
#include <boost/utility/string_ref.hpp>

namespace std {
    template <> struct hash<boost::string_ref> {
        size_t operator()(boost::string_ref const& sr) const {
            return boost::hash_range(sr.begin(), sr.end());
        }
    };

    template <> struct hash<std::pair<bool,std::string>> {
        size_t operator()( std::pair<bool,std::string> const &p) const {
            return boost::hash_value(p);
        }
    };

    template <> struct hash<std::vector<size_t>> {
        size_t operator()( std::vector<size_t> const& vec ) const {
            return boost::hash_range(vec.begin(), vec.end());
        }
    };

    template <> struct hash<list<size_t>> {
        std::size_t operator()(const list<size_t>& vec) const {
            return boost::hash_range(vec.begin(), vec.end());
        }
    };

    template <> struct hash<std::pair<size_t,std::vector<size_t>>> {
        size_t operator()(const std::pair<size_t,std::vector<size_t>>& p) const {
            std::size_t seed = 0;
            boost::hash_combine(seed, p.first);
            boost::hash_combine(seed,
                                boost::hash_range(p.second.begin(),
                                                  p.second.end()));
            return seed;
        }
    };
};

namespace nn {

    typedef std::vector<size_t> size_t_vec;

    size_t_vec from_vec(const std::vector<size_t_vec>& prefix,
                        const size_t_vec& last,
                        size_t BOS,
                        size_t EOS,
                        size_t SPACE) {
        size_t_vec key {BOS};
        for (const auto& w : prefix) {
            for (auto c_it = w.begin()+1; c_it != w.end()-1; ++c_it) {
                key.push_back(*c_it);
            }
            key.push_back(SPACE);
        }
        for (auto c_it = last.begin()+1; c_it != last.end()-1; ++c_it) {
            key.push_back(*c_it);
        }
        key.push_back(EOS);
        return key;
    }

    size_t increment(size_t base, size_t& i) {
        if (i==base-1) return base;
        return ++i;
    }

    std::vector<std::vector<size_t>> enum_seq(size_t base, size_t len) {
        std::vector<std::vector<size_t>> ret;

        std::vector<size_t> seq;
        seq.resize(len); // 0 initialized

        while(true) {
            ret.push_back(seq);
            size_t idx = len;
            while(increment(base, seq[--idx]) == base) {
                seq[idx] = 0;
                if (idx == 0) return ret;
            }
        }

        return ret;
    }

    std::string d2s(double x) {
        return boost::str(boost::format("%.2f") % x);
    }

    template<typename T>
    std::string vec2size_t_str(const std::vector<T> vec) {
        std::ostringstream oss;
        oss << "( ";
        if (!vec.empty()) {
            // Convert all but the last element to avoid a trailing ","
            std::copy(vec.begin(), vec.end(), std::ostream_iterator<size_t>(oss, " "));

            // Now add the last element with no delimiter
            oss << vec.back();
        }
        oss << ")";
        return oss.str();
    }

    template<typename T>
    std::string vec2str(const std::vector<T> vec) {
        std::ostringstream oss;
        oss << "( ";
        if (!vec.empty()) {
            // Convert all but the last element to avoid a trailing ","
            std::copy(vec.begin(), vec.end()-1, std::ostream_iterator<T>(oss, " "));

            // Now add the last element with no delimiter
            oss << vec.back();
        }
        oss << " )";
        return oss.str();
    }

    void combinations_r_recursive(const std::vector<size_t> &elems, size_t req_len,
                                  std::vector<size_t> &pos, size_t depth,
                                  size_t margin,
                                  std::vector<std::vector<size_t>>& ret
        )
    {
        // Have we selected the number of required elements?
        if (depth >= req_len) {
            //for (size_t ii = 0; ii < pos.size(); ++ii)
            //	cout << elems[pos[ii]];
            //cout << endl;

            std::vector<size_t> comb;
            for(size_t ii = 0; ii < pos.size(); ++ii) {
                comb.push_back( pos[ii] );
            }
            ret.push_back(comb);

            return;
        }

        // Try to select new elements to the right of the last selected one.
        for (size_t ii = margin; ii < elems.size(); ++ii) {
            pos[depth] = ii;
            combinations_r_recursive(elems, req_len, pos, depth + 1, ii, ret);
        }
        return;
    }

    void combinations_r(const std::vector<size_t> &elems, size_t req_len, std::vector<std::vector<size_t>>& ret)
    {
        assert(req_len > 0 && req_len <= elems.size());
        std::vector<size_t> positions(req_len, 0);
        combinations_r_recursive(elems, req_len, positions, 0, 0, ret);
    }

    inline bool readable(std::string path) {
        std::ifstream infile(path);
        return infile.good();
    }

    template<typename T>
    class table {
        std::unordered_map<T, size_t> tab;
    public:
        size_t index(T t) {
            if(tab.count(t) > 0) {
                return tab.at(t);
            } else {
                size_t ret = tab.size();
                tab[t] = ret;
                return ret;
            }
        }
        T type(size_t index) {
            return tab.at(index);
        }
        size_t size() {
            return tab.size();
        }
    };

    /**
     * Alias: std::vector<double>
     */
    typedef std::vector<double> d_vec;

    /**
     * Alias: std::vector<std::vector<double> >
     */
    typedef std::vector<d_vec> d_vec_vec;

    /**
     * Alias: std::vector<unsigned int>
     */
    typedef std::vector<unsigned int> ui_vec;

    /**
     * Alias: std::vector<std::vector<unsigned int> >
     */
    typedef std::vector<ui_vec> ui_vec_vec;

    /**
     * Handy typedefs
     */
    //typedef int32_t l_type;
    typedef size_t l_type;

    template <template<class,class,class...> class C, typename K, typename V, typename... Args>
    V GetWithDef(const C<K,V,Args...>& m, K const& key, const V & defval)
    {
        typename C<K,V,Args...>::const_iterator it = m.find( key );
        if (it == m.end())
            return defval;
        return it->second;
    }

    template<typename T>
    std::string vec2str(std::list<T> v) {
        std::stringstream ss;
        assert(v.size() > 0);
        auto i = 0;
        for(auto elem : v) {
            ss << elem;
            ++ i;
            if(i<v.size()) {
                ss << " ";
            }
        }
        return ss.str();
    }

    // Return a vector of values from an unordered_map
    template<typename T, typename U>
    std::vector<U> map_values(std::unordered_map<T,U> map) {
        std::vector<U> ret;
        for(auto keyval : map) {
            ret.push_back( keyval.second );
        }
        return ret;
    }

    std::vector<bool> str2bools(std::string seg) {
        std::vector<bool> bs;
        for(auto c : seg) {
            if (c == '0') { bs.push_back(false); }
            else          { bs.push_back(true);  }
        }
        return bs;
    }

    std::string bools2str(std::vector<bool> bools) {
        std::string ret;
        for(auto b : bools) {
            ret.append( std::to_string(b) );
        }
        return ret;
    }

    template<typename T>
    bool compare_by_val(const std::pair<T,double>& a, const std::pair<T,double>& b) {
        return b.second < a.second;
    }

    boost::string_ref get_safe_substr(boost::string_ref str, unsigned t, unsigned delta) {
        unsigned len;
        if(t+delta > str.size()) {
            len = str.size() - t;
        } else {
            len = delta;
        }
        return str.substr(t, len);
    }

    std::list<boost::string_ref> get_word_list(boost::string_ref input, std::vector<bool> seg) {
        std::list<boost::string_ref> ret;
        auto word_start = 0;
        auto word_len = 1;
        for(size_t t = 0; t < input.size(); t++) {
            if(seg[t]) {
                boost::string_ref w = input.substr(word_start, word_len);
                ret.push_back(w);
                word_start += word_len;
                word_len = 1;
            } else {
                word_len += 1;
            }
        }
        return ret;
    }

    std::vector<boost::string_ref> get_words(boost::string_ref input, std::vector<bool> seg) {
        std::vector<boost::string_ref> ret;
        auto word_start = 0;
        auto word_len = 1;
        for(size_t t = 0; t < input.size(); t++) {
            if(seg[t]) {
                boost::string_ref w = input.substr(word_start, word_len);
                ret.push_back(w);
                word_start += word_len;
                word_len = 1;
            } else {
                word_len += 1;
            }
        }
        return ret;
    }

    std::vector<std::vector<boost::string_ref>> get_words(std::vector<boost::string_ref> input, std::vector<bool> seg) {
        std::vector<std::vector<boost::string_ref>> ret;

        auto start = 0;
        for(auto sent : input) {
            auto len = sent.size();
            std::vector<bool> sub_seg(seg.begin() + start, seg.begin() + start + len - 1);
            sub_seg.push_back(1);
            start += (len-1);
            ret.push_back( get_words(sent, sub_seg) );
        }

        return ret;
    }

    inline std::string makeProgressBarString(double percentDone, int total=80) {
        std::ostringstream out;

        int numDone = (int)floor(total * percentDone);

        out << "[";
        for (int i=0; i<numDone; i++) {
            out << "=";
        }
        out << ">";
        for (int i=numDone; i<total; i++) {
            out << " ";
        }
        out << "]";
        return out.str();
    }

    /**
     * Sum the elements in a sequence.
     *
     * @tparam T iteratable sequence type; must have a const_iterator member
     *           as well as begin() and end() methods.
     * @param in sequence to compute the sum of
     * @returns the sum of the elements in the sequence
     */
    template<typename T>
    inline typename T::value_type sum(const T& in) {
        typename T::value_type sum = 0;
        for (typename T::const_iterator i = in.begin(); i != in.end(); ++i) {
            sum += *i;
        }
        return sum;
    }

    template<typename A, typename B>
    std::pair<B,A> flip_pair(const std::pair<A,B> &p) {
        return std::pair<B,A>(p.second, p.first);
    }

    template<typename A, typename B>
    std::multimap<B,A> flip_map(const std::map<A,B> &src) {
        std::multimap<B,A> dst;
        std::transform(src.begin(), src.end(), std::inserter(dst, dst.begin()),
                       flip_pair<A,B>);
        return dst;
    }

    template <typename Iter, typename Cont>
    bool is_last(Iter iter, const Cont& cont) {
        return (iter != cont.end()) && (next(iter) == cont.end());
    }
};

#endif
