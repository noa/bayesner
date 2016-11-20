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

#ifndef __NN_DATA_HPP__
#define __NN_DATA_HPP__

#include <string>
#include <sstream>
#include <vector>
#include <functional>
#include <map>
#include <set>
#include <fstream>

#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

#include <nn/mutable_symtab.hpp>
#include <nn/utils.hpp>

namespace nn {

    enum class Annotation {
        FULL, SEMI, NONE, UNDEF
    };

    typedef size_t                           sym;
    typedef std::vector<sym>                 syms;
    typedef std::pair<sym, syms>             obs_t;
    typedef std::vector<syms>                phrase;
    typedef std::vector<phrase>              utt;
    typedef std::tuple<syms, utt>            segmented_utterance;
    typedef std::vector<segmented_utterance> corpus;
    typedef std::tuple<sym, phrase>          tagged_phrase;
    typedef std::vector<tagged_phrase>       dictionary;

    const size_t CONTEXT_WORD = (size_t) - 1;

    std::vector<size_t> join(typename std::vector<syms>::const_iterator start,
                             typename std::vector<syms>::const_iterator stop,
                             size_t BOS, size_t SPACE, size_t EOS
        ) {
        std::vector<size_t> joined {BOS};
        size_t tot_syms {0};
        size_t n_words {0};
        for(auto it = start; it != stop; ++it) {
            CHECK(it->front() == BOS) << "sequence doesn't begin with BOS";
            CHECK(it->back()  == EOS) << "sequence doesn't end with EOS";
            CHECK(it->size() > 2)     << "sequence too short";
            auto word = *it;
            for(auto i = 1; i < word.size()-1; ++i) {
                joined.push_back(word.at(i));
                tot_syms ++;
            }
            if ((it+1) != stop) { joined.push_back(SPACE); }
            n_words ++;
        }
        joined.push_back(EOS);
        CHECK(joined.size() > 2)                     << "empty sequence";
        CHECK(joined.size() == (tot_syms+n_words+1)) << "logic error: " << joined.size() << " vs " << (tot_syms+n_words+1);
        return joined;
    }

    syms flatten(const phrase& p, size_t BOS, size_t EOS, size_t SPACE) {
        syms ret;
        size_t len = p.size();
        size_t i = 0;
        ret.push_back(BOS);
        for(auto subseq : p) {
            for(auto s : subseq) {
                ret.push_back(s);
            }
            if (i == len-1) ret.push_back(EOS);
            else            ret.push_back(SPACE);
            ++i;
        }
        return ret;
    }

    syms read_spaced_word(std::string str) {
        std::vector<std::string> tokens;
        CHECK(str.size() > 0) << "empty string: " << str;
        boost::algorithm::split(tokens, str, boost::is_any_of(" "), boost::token_compress_on);
        syms ret;
        CHECK(tokens.size() > 0) << "bad string: " << str;
        for(auto t : tokens) {
            ret.push_back( std::stoul(t) );
        }
        CHECK(ret.size() > 0) << "bad string: " << str;
        return ret;
    }

    std::vector<phrase> get_observations(sym _tag, const dictionary& dict) {
        std::vector<phrase> ret;
        for(auto entry : dict) {
            auto tag = std::get<0>(entry);
            if(tag == _tag) {
                auto p = std::get<1>(entry);
                ret.push_back(p);
            }
        }
        return ret;
    }

    template<typename IntVec>
    std::string get_string(
        IntVec encoded,
        const std::unordered_map<sym, std::string>& sym_map
        ) {
        std::string ret;
        for(size_t i = 1; i < encoded.size()-1; ++i) {
            size_t sym = encoded[i];
            CHECK( sym_map.count(sym) ) << "missing symbol: " << sym;
            std::string ch = sym_map.at(sym);
            ret.append(ch);
        }
        return ret;
    }

    template<typename IntVec, typename Map>
    std::string get_string(IntVec w, Map* map) {
        std::string ret;
        for(size_t i = 1; i < w.size()-1; ++i) {
            size_t sym = w[i];
            CHECK( map->has(sym) ) << "missing symbol: " << sym;
            std::string ch = map->val(sym);
            ret.append(ch);
        }
        return ret;
    }

    std::vector<syms> get_observation(const segmented_utterance& instance) {
        std::vector<syms> ret;
        auto phrases = std::get<1>(instance);
        for(const auto& p : phrases) {
            for(const auto& w : p) {
                ret.push_back(w);
            }
        }
        return ret;
    }

    // TODO: each word is wrapped below. Doesn't the filter assume they're not
    //       wrapped?
    // noa: re:above should be OK
    std::vector<syms> get_observation(const segmented_utterance& instance,
                                      const mutable_symbol_table<>& symtab,
                                      size_t BOS, size_t EOS) {
        CHECK( std::get<0>(instance).size() > 0 ) << "empty utt";
        std::vector<syms> ret;
        auto phrases = std::get<1>(instance);
        for(const auto& p : phrases) {
            for(const auto& w : p) {
                ret.push_back( wrap(w, symtab, BOS, EOS) );
            }
        }
        return ret;
    }

    std::vector<phrase> get_observations(const corpus& c) {
        std::vector<std::vector<syms>> ret;
        for(const auto& instance : c) {
            auto phrases = std::get<1>(instance);
            std::vector<syms> words;
            for(const auto& p : phrases) {
                for(const auto& w : p) {
                    words.push_back(w);
                }
            }
            ret.push_back(words);
        }
        return ret;
    }

    // WARNING: This assumes zero (0) is the context tag
    std::vector<std::string> get_conll_tag_strs(
        const syms& tags,
        const syms& lens,
//        const std::unordered_set<size_t>& gaz_tags,
        sym context_tag,
        const std::unordered_map<size_t, std::string>& tag_desc
        ) {
        //LOG(INFO) << "tags.size = " << tags.size();
        //LOG(INFO) << "lens.size = " << lens.size();
        CHECK(tags.size() == lens.size()) << "unexpected size";
        std::vector<std::string> ret;
        size_t i = 0;
        for (auto it = tags.begin(); it != tags.end(); ++it) {
            CHECK(i < lens.size()) << "indexing error";
            auto tag = *it;
            auto l = lens.at(i);
            CHECK(l > 0) << "lens must be >= 0";
            CHECK(tag_desc.count(tag) > 0) << "missing tag: " << tag;
            if (tag == context_tag) {
                CHECK(l == 1);
                ret.push_back( tag_desc.at(tag) );
            } else {
                ret.push_back( "B-"+tag_desc.at(tag) );
                for(size_t j=0; j<l-1; ++j) {
                    ret.push_back( "I-"+tag_desc.at(tag) );
                }
            }
            ++i;
        }
        return ret;
    }


    void write_tagging_conll(std::ofstream& of,
                             phrase words,
                             syms pred_tags,
                             syms pred_lens, // how many words each of the pred tags spans
                             syms gold_tags,
                             syms gold_lens,
//                             std::unordered_set<size_t> gaz_tags,
                             sym context_tag,
                             std::unordered_map<size_t, std::string> sym_desc,
                             std::unordered_map<size_t, std::string> tag_desc
        ) {
        //LOG(INFO) << "[write tagging conll]";
        //LOG(INFO) << "tag_desc.size() = " << tag_desc.size();

        auto pred_tag_strs = get_conll_tag_strs(pred_tags, pred_lens,
                                                context_tag, tag_desc);
        auto gold_tag_strs = get_conll_tag_strs(gold_tags, gold_lens,
                                                context_tag, tag_desc);

        CHECK(pred_tag_strs.size() == gold_tag_strs.size());
        CHECK(gold_tag_strs.size() == words.size()-1);

        for (size_t i = 0; i < words.size()-1; ++i) {
            auto str = get_string(words.at(i), sym_desc);
            of << str << " ";
            of << gold_tag_strs.at(i) << " ";
            of << pred_tag_strs.at(i) << std::endl;
        }
        of << std::endl;
    }

    std::vector<phrase> get_observations(sym _tag, const dictionary& dict, size_t max_instances) {
        size_t count = 0;
        std::vector<phrase> ret;
        for(auto i : get_observations(_tag, dict)) {
            ret.push_back(i);
            count += 1;
            if (count > max_instances) break;
        }
        return ret;
    }

    struct instance {
        syms        chars;
        phrase      words;
        syms        tags;
        Annotation  obs { Annotation::UNDEF };
        std::vector<size_t> lens; // how many words each tag spans

        instance() {}
        void clear() {
            chars.clear();
            words.clear();
            tags.clear();
            lens.clear();
            obs = Annotation::UNDEF;
        }

        instance(const segmented_utterance& utt,
                 const nn::uint_str_table& tagtab,
                 const nn::uint_str_table& symtab,
                 size_t BOS_sym, size_t EOS_sym, size_t SPACE_sym,
                 syms EOS_word) {
            words = get_observation(utt, symtab, BOS_sym, EOS_sym);
            words.push_back(EOS_word);
            chars = flatten(words, BOS_sym, EOS_sym, SPACE_sym);
            for(const auto& s : std::get<0>(utt)) {
                tags.push_back(s);
            }
            for(const auto& p : std::get<1>(utt)) {
                lens.push_back(p.size());
            }
            CHECK(lens.size() > 0) << "empty lens! tag size = " << tags.size() << " words.size() = " << words.size();
            CHECK(tags.size() > 0) << "empty tags!";
            CHECK(lens.size() == tags.size()) << "size mismatch!";
        }

        void log(const std::unordered_map<size_t,std::string>& sym_desc) {
            auto it = words.begin();
            for(size_t i = 0; i < tags.size(); ++i) {
                std::stringstream ss;
                auto it2 = it;
                ss << get_string(*it2, sym_desc);
                for(size_t j = 1; j < lens.at(i)-1; ++j) {
                    ss << " " << get_string(*it2, sym_desc);
                    ++it2;
                }
                LOG(INFO) << "tag=" << tags.at(i) << " len=" << lens.at(i) << " words=" << ss.str();
                it += lens.at(i);
            }
        }
    };

    typedef std::vector<instance> instances;

    std::vector<phrase> read_unlabeled(std::string path) {
        std::vector<phrase> ret;

        if(path != "") {
            CHECK(nn::readable(path)) << "bad path";
            std::string line;
            syms tags;
            std::ifstream file(path);
            while (std::getline(file, line)) {
                std::vector<std::string> tokens;
                boost::algorithm::split(tokens,line,boost::is_any_of("\t"),boost::token_compress_on);
                CHECK(tokens.size() > 1) << "bad input line: " << line;
                phrase p;
                for(auto word : tokens) {
                    p.push_back( read_spaced_word(word) );
                }
                ret.push_back( p );
            }
        } else {
            LOG(INFO) << "Warning! Empty path for unlabeled data";
        }

        return ret;
    }

    // One tagged phrase per line
    // <TAG><TAB><W1><TAB><W2> ...
    // where
    // <TAG> = a number
    // <Wi> = a sequence of space-separated numbers
    void read_conll(std::string path, corpus& ret) {
        if(path != "") {
            CHECK(nn::readable(path)) << "bad path";
            std::string line;
            syms tags;
            utt phrases;
            std::ifstream file(path);
            while (std::getline(file, line)) {
                if (line.length() == 0) { // end of an instance
                    if(tags.size() > 0 && phrases.size() > 0) {
                        ret.push_back( std::make_tuple(tags, phrases) );
                        tags.clear();
                        phrases.clear();
                    }
                    continue;
                }
                std::vector<std::string> tokens;
                boost::algorithm::split(tokens,line,boost::is_any_of("\t"),boost::token_compress_on);
                CHECK(tokens.size() > 1) << "bad input line: " << line;
                auto tag = std::stoul(tokens[0]);
                tags.push_back( tag );
                phrase p;
                for(size_t i = 1; i < tokens.size(); ++i) {
                    p.push_back( read_spaced_word(tokens[i]) );
                }
                CHECK(p.size() > 0) << "empty phrase! line: " << line;
                phrases.push_back(p);
            }
            if (tags.size() > 0 && phrases.size() > 0) {
                ret.push_back( std::make_tuple(tags, phrases) );
            }
        }
    }

    std::map<size_t, std::string> read_sym_str_map(std::string path) {
        CHECK(nn::readable(path)) << "bad path: " << path;
        std::ifstream file(path);
        std::string line;
        std::map<size_t, std::string> ret;
        while (std::getline(file, line)) {
            std::vector<std::string> tokens;
            boost::algorithm::split(tokens,line,boost::is_any_of("\t "),boost::token_compress_on);
            CHECK(tokens.size() == 2) << "bad input line: " << line;
            ret[std::stoul(tokens[0])] = tokens[1];
        }
        return ret;
    }

    std::map<size_t, std::string> read_tag_str_map(std::string path) {
        CHECK(nn::readable(path)) << "bad path";
        std::ifstream file(path);
        std::string line;
        std::map<size_t, std::string> ret;
        while (std::getline(file, line)) {
            std::vector<std::string> tokens;
            boost::algorithm::split(tokens,line,boost::is_any_of("\t "),boost::token_compress_on);
            CHECK(tokens.size() == 2) << "bad input line: " << line;
            ret[std::stoul(tokens[1])] = tokens[0];
        }
        return ret;
    }

    dictionary read_dict(std::string path) {
        dictionary ret;
        if(path != "") {
            CHECK(nn::readable(path)) << "bad path";
            std::ifstream file(path);
            std::string line;
            while (std::getline(file, line)) {
                std::vector<std::string> tokens;
                boost::algorithm::split(tokens,line,boost::is_any_of(" "),boost::token_compress_on);
                sym tag = std::stoul(tokens[0]);
                syms word;
                for(size_t i = 1; i < tokens.size(); ++i) {
                    word.push_back( std::stoul(tokens[i]) );
                }
                phrase words { word };
                ret.emplace_back(tag, words);
            }
        }
        return ret;
    }

    dictionary read_gaz(std::string path,
                        uint_str_table tab,
                        size_t BOS,
                        size_t EOS) {
        dictionary ret;
        if(path != "") {
            LOG(INFO) << "reading gazetteer from: " << path;
            CHECK(nn::readable(path)) << "bad path";
            std::string line;
            std::ifstream file(path);
            while (std::getline(file, line)) {
                std::vector<std::string> tokens;
                boost::algorithm::split(tokens,
                                        line,
                                        boost::is_any_of("\t"),
                                        boost::token_compress_on);
                CHECK(tokens.size() > 1) << "bad input line: " << line;
                auto tag = std::stoul(tokens[0]);
                phrase p;
                for(size_t i = 1; i < tokens.size(); ++i) {
                    p.push_back(wrap(read_spaced_word(tokens[i]), tab, BOS, EOS));
                }
                ret.emplace_back(tag, p);
            }
        }
        return ret;
    }

    dictionary read_gaz(std::string path) {
        std::set<sym> types;
        return read_gaz(path);
    }

    std::set<sym> count_syms(const std::vector<phrase>& data) {
        std::set<sym> ret;
        for(auto sent : data) {
            for(auto word : sent) {
                for(auto c : word) {
                    ret.insert(c);
                }
            }
        }
        return ret;
    }

    std::set<sym> count_syms(const corpus& c) {
        std::set<sym> ret;
        for(auto seg_utt : c) {
            for(auto p : std::get<1>(seg_utt)) {
                for(auto w : p) {
                    for(auto c : w) {
                        ret.insert(c);
                    }
                }
            }
        }
        return ret;
    }

    std::set<sym> count_syms(const dictionary& d) {
        std::set<sym> ret;
        for(auto tp : d) {
            for(auto w : std::get<1>(tp)) {
                for(auto c : w) {
                    ret.insert(c);
                }
            }
        }
        return ret;
    }

    std::string instance_string(segmented_utterance instance,
                                const std::unordered_map<size_t,std::string>& tag_desc,
                                const std::unordered_map<size_t,std::string>& sym_desc
        ) {
        std::stringstream ss;
        size_t i,j;
        auto tags = std::get<0>(instance);
        auto phrs = std::get<1>(instance);
        for(i=0;i<tags.size();++i){
            auto t = tags.at(i);
            auto p = phrs.at(i);
            if(t==CONTEXT_WORD) {
                CHECK(p.size() == 1) << "unexpected length";
                ss << get_string(p.at(0), sym_desc);
                ss << " ";
            } else {
                ss << "( " << tag_desc.at(t) << " ";
                for(j=0; j<p.size(); ++j) {
                    ss << get_string(p.at(j), sym_desc);
                    ss << " ";
                }
                ss << ") ";
            }
        }
        return ss.str();
    }

    size_t count_syms(const std::vector<phrase>& train,
                      const corpus& test,
                      const dictionary& dict,
                      const dictionary& gaz) {

        std::vector<std::set<sym>> sets;
        sets.push_back( count_syms(train) );
        sets.push_back( count_syms(test) );
        sets.push_back( count_syms(dict) );
        sets.push_back( count_syms(gaz) );
        std::set<sym> uni;
        for(const auto& s : sets) {
            for(const auto& i : s) {
                uni.insert(i);
            }
        }
        return uni.size();
    }

    template<typename P, typename I, typename C, typename M>
    void write_state(std::vector<P> state,
                     std::vector<I> instances,
                     const C& corpus,
                     const M& model,
                     std::string path
        ) {
        CHECK(state.size() == instances.size()) << "size mismatch";
        CHECK(state.size() > 0);
        std::ofstream of (path);
        CHECK(of.is_open()) << "problem opening path: " << path;
        for(auto n = 0; n < state.size(); ++n) {
            auto p = state.at(n);
            auto i = instances.at(n);
            auto tags = model.get_tags(p);
            auto lens = model.get_lens(p);
            CHECK(tags.size() > 0);
            CHECK(lens.size() > 0);
            write_tagging_conll(of, i.words,
                                tags, lens,
                                i.tags, i.lens,
                                corpus.get_other_key(),
                                corpus.symtab.get_map(),
                                corpus.tagtab.get_map());
        }
        of.close();
    }
};

#endif
