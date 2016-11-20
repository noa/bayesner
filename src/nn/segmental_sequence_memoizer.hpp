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

#ifndef __NN_SEGMENTAL_SEQUENCE_MEMOIZER_HPP__
#define __NN_SEGMENTAL_SEQUENCE_MEMOIZER_HPP__

#include <vector>
#include <memory>
#include <unordered_map>
#include <sstream>

#include <boost/lexical_cast.hpp>

#include <nn/mutable_symtab.hpp>
#include <nn/discrete_distribution.hpp>
#include <nn/data.hpp>
#include <nn/seq_model.hpp>
#include <nn/simple_seq_model.hpp>
#include <nn/adapted_seq_model_prefix.hpp>

namespace nn {

    bool is_range(std::string s) {
        std::replace( s.begin(), s.end(), '-', '0');
        bool is_a_range = false;
        try {
            boost::lexical_cast<double>(s);
            is_a_range = true;
        }
        catch(boost::bad_lexical_cast &) {
            // if it throws, it's not a number.
        }
        return is_a_range;
    }

    bool is_number(std::string s) {
        std::replace( s.begin(), s.end(), ',', '.');
        bool is_a_number = false;
        try {
            boost::lexical_cast<double>(s);
            is_a_number = true;
        }
        catch(boost::bad_lexical_cast &) {
            // if it throws, it's not a number.
        }
        return is_a_number;
    }

    syms process_context_word(syms raw_w,
                              mutable_symbol_table<>* symtab) {
        syms ret;
        std::string s;

        for(auto i = 1; i < raw_w.size() - 1; ++i) {
            auto sym = raw_w[i];
            s.append( symtab->val(sym) );
        }

        std::transform(s.begin(), s.end(), s.begin(), ::tolower);

        for(auto i = 0; i < s.size(); ++i) {
            std::string ch = s.substr(i,1);
            ret.push_back( symtab->key(ch) );
        }
        return ret;
    }

    enum class FilterProposal { CHUNK, BASELINE, HYBRID, PROP1 };

    template<typename base_t = HashIntegralMeasure<sym>,
             typename tran_t = FixedDepthHPYP<sym, syms, base_t>,
             typename emit_t = adapted_seq_model_prefix<>>
    class segmental_sequence_memoizer {
        typedef phrase Context;

        const syms BOS; // beginning of string obs
        const syms EOS; // end of string obs

        const sym context_tag;
        const sym eos_tag;

        bool frozen { false };
        typename emit_t::param emit_param;

        std::unique_ptr<base_t> H {nullptr};                 // base distrib
        std::unique_ptr<tran_t> T {nullptr};                 // context model
        std::unordered_map<sym, std::unique_ptr<emit_t>> E;  // emission models

        FilterProposal prop { FilterProposal::HYBRID };

        const double STOP_PROB {0.9};

        const uint_str_table& symtab;
        const uint_str_table& tagtab;

        // Filter diagnostics:
        size_t n_sampled_between_start {0};
        size_t n_sampled_between_stop  {0};
        size_t n_sampled_inside_stop   {0};
        size_t n_sampled_inside_cont   {0};

    public:
        segmental_sequence_memoizer(syms _BOS,
                                    syms _EOS,
                                    sym _context_tag,
                                    const uint_str_table& _symtab,
                                    const uint_str_table& _tagtab)
            : BOS(_BOS), EOS(_EOS), context_tag(_context_tag),
              symtab(_symtab), tagtab(_tagtab) {}

        template<typename Corpus>
        segmental_sequence_memoizer(const Corpus& corpus)
            : BOS(corpus.get_bos_obs()),
              EOS(corpus.get_eos_obs()),
              context_tag { corpus.get_other_key() },
              eos_tag     { corpus.tagtab.size()   },
              symtab(corpus.symtab),
              tagtab(corpus.tagtab) {

                  typename emit_t::param emit_param;
                  emit_param.discount = 0.5;
                  emit_param.alpha    = 1.0;
                  emit_param.nsyms    = symtab.size();
                  emit_param.BOS      = corpus.get_bos_key();
                  emit_param.EOS      = corpus.get_eos_key();
                  emit_param.SPACE    = corpus.get_space_key();
                  set_emit_param(emit_param);
                  size_t num_tags_total = tagtab.size();
                  for(auto kv : tagtab.get_map()) {
                      LOG(INFO) << kv.first << " : " << kv.second;
                  }
                  LOG(INFO) << "Total number of tags: " << num_tags_total;
                  H = std::make_unique<base_t>();
                  std::set<size_t> tags;
                  for(auto k : tagtab.get_key_set()) {
                      tags.insert(k);
                      auto emit_model = std::make_unique<emit_t>(emit_param);
                      add_emission_model( k, std::move(emit_model) );
                      if (k == context_tag) {
                          const double a = 2.0;
                          LOG(INFO) << "Prior for context tag " << context_tag
                                    << " = " << a;
                          H->add(k, a);
                      }
                      else {
                          const double a = 1.0;
                          LOG(INFO) << "Prior for tag " << k
                                    << " = " << a;
                          H->add(k, a);
                      }
                  }
                  // Add EOS
                  CHECK(!tags.count(eos_tag)) << "logic error";
                  H->add(eos_tag, 1.0);
                  auto tran_model = std::make_unique<tran_t>( H.get() );
                  add_transition_model( std::move(tran_model) );
                  LOG(INFO) << "Num emission models: " << num_emission_model();
                  CHECK(num_emission_model() == corpus.tagtab.size()) << "tag size mismatch";
                  freeze(); // don't add any emission models past this point
              }

        segmental_sequence_memoizer(segmental_sequence_memoizer const&) = delete;
        segmental_sequence_memoizer& operator=(segmental_sequence_memoizer const&) = delete;

        std::unordered_set<size_t> tag_set() const {
            std::unordered_set<size_t> ret;
            for(const auto& kv : E) {
                ret.insert(kv.first);
            }
            return ret;
        }

        bool consistent()       { return true;     }
        size_t num_tags() const { return E.size(); }

        void set_symtab(const nn::mutable_symbol_table<>& symtab_) {
            symtab = symtab_;
        }

        void set_tagtab(const nn::mutable_symbol_table<>& tagtab_) {
            tagtab = tagtab_;
        }

        struct particle {
            std::vector<syms> words;    // words in the current phrase

            bool in_phrase {false};     // if we're extending a phrase
            bool done      {false};     // reached EOS
            syms tags;                  // predicted tags
            std::vector<size_t> lens;   // span (# of words) for each tag

            std::vector<syms> context;  // all previous words, used for predictions
            sym context_tag;

            void dlog() {
                std::stringstream ss;
                for(size_t i = 0; i < lens.size(); ++i) {
                    ss << "(tag=" << tags.at(i) << ", len=" << lens.at(i) << ")";
                }
                if(in_phrase) { ss << "(in phrase)"; }
            }

            size_t pos_idx(size_t t) const {
                size_t ret = 0;
                size_t pos = 0;
                for(auto i=0; i<tags.size(); ++i) {
                    auto len = lens.at(i);
                    pos += len;
                    if (t < pos) return i;
                    ret ++;
                }
                CHECK(false) << "logic error";
                return 0;
            }

            bool end_at_pos(size_t t) const {
                size_t ret = 0;
                size_t pos = 0;
                for(auto i=0; i<tags.size(); ++i) {
                    auto len = lens.at(i);
                    pos += len;
                    if (t < pos) {
                        if (t == pos-1) return true;
                        else            return false;
                    }
                    ret ++;
                }
                CHECK(false) << "t: " << t << " len(tags): " << tags.size()
                             << "; len(lens): " << lens.size();
                return false;
            }

            size_t tag_at_pos(size_t t) const {
                auto idx = pos_idx(t);
                CHECK(idx < tags.size()) << "idx: " << idx << " tags.size(): " << tags.size();
                return tags.at(idx);
            }

            size_t len_at_pos(size_t t) const {
                auto idx = pos_idx(t);
                CHECK(idx < lens.size()) << "idx: " << idx << " lens.size(): " << lens.size();
                return lens.at(idx);
            }

            void start(size_t tag, syms w) {
                in_phrase = true;

                tags.push_back(tag);
                lens.push_back(1);

                words.push_back(w);
            }

            void add(size_t tag, syms w) {
                in_phrase = false;

                tags.push_back(tag);
                lens.push_back(1);
            }

            void stop(syms w) {
                in_phrase = false;
                lens.back()++;
                words.clear();
            }

            // start a segment then change our mind
            void emergencyStop(syms w) {
                in_phrase = false;
                words.clear();
            }

            void stopEOS(syms w) {
                done = true;
                in_phrase = false;
                words.clear();
            }

            void cont(syms w) {
                CHECK(in_phrase) << "cont while not in phrase";
                lens.back()++;
                words.push_back(w);
            }
        };

        syms get_tags(const particle& p) const { return p.tags; }
        syms get_lens(const particle& p) const { return p.lens; }

        particle make_particle(const instance& i) const {
            particle p;
            p.tags = i.tags;
            p.lens = i.lens;
            return p;
        }

        particle make_particle(const std::vector<size_t>& tags,
                               const std::vector<size_t>& lens) const {
            particle p;
            p.tags = tags;
            p.lens = lens;
            return p;
        }

        std::vector<particle> make_particles(const instances& is) const {
            std::vector<particle> ret;
            for(auto i : is) { ret.push_back( make_particle(i) ); }
            return ret;
        }

        void init(particle& p) const {
            p.in_phrase = false;
            p.tags.clear();
            p.tags.reserve(128);
            p.lens.clear();
            p.lens.reserve(128);
            p.words.clear();
            p.context.clear();
            p.context.reserve(128);
            p.context.push_back(BOS);
            p.context_tag = context_tag;
            //p.symtab = symtab;
        }

        void add_transition_model( std::unique_ptr<tran_t> _T ) {
            T = std::move( _T );
        }

        void add_emission_model( sym tag, std::unique_ptr<emit_t> _E ) {
            LOG(INFO) << "adding emission model for tag " << tag;
            E.emplace(tag, std::move(_E));
        }

        bool has_emission_model( sym tag ) {
            return E.count(tag);
        }

        void set_emit_param( typename emit_t::param p ) {
            emit_param = p;
        }

        void resample_hyperparameters() {
            LOG(INFO) << "Resampling hyperparameters...";
            for(auto it = E.begin(); it != E.end(); ++it) {
                it->second->resample_hyperparameters(rng::get());
            }
        }

        size_t num_emission_model() { return E.size(); }

        emit_t* get_emission_model( sym tag ) const {
            // if (!has_emission_model(tag)) {
            //     if (frozen) { LOG(FATAL) << "unknown tag: " << tag; }
            //     auto model = std::make_unique<emit_t>(emit_param);
            //     add_emission_model( tag, std::move(model) );
            // }
            return E.at(tag).get();
        }

        void freeze()                  { frozen = true;  }
        tran_t* get_transition_model() { return T.get(); }

        void set_proposal(FilterProposal _prop) {
            prop = _prop;
        }

        bool sanity(const particle& p,
                    const phrase& input,
                    const syms& gold_tags,
                    const syms& gold_lens) {
            CHECK(p.tags.size() > 0) << "empty tag list";
            CHECK(p.lens.size() > 0) << "empty lens list";
            CHECK(p.done) << "incomplete particle!!!";
            size_t total = 0;
            size_t i = 0;

            i = 0;
            for(auto l : p.lens) {
                auto tag = p.tags[i++];
                total += l;
            }

            // The -1 below excludes the final <EOS> observation
            CHECK(total == input.size()-1) << "size mismatch! " << total << " vs " << (input.size()-1);
            CHECK(p.tags.size() == p.lens.size()) << "size mismatch: (tags) " << p.tags.size() << " (lens) " << p.lens.size();
            return true;
        }

        void observe_gazetteer(const syms& tags,
                               const syms& lens,
                               const phrase& words) {
            CHECK(tags.size() > 0) << "no tags";
            CHECK(lens.size() > 0) << "no lens";
            CHECK(tags.size() == lens.size()) << "size mismatch! tags.len = "
                                              << tags.size() << " lens.len = "
                                              << lens.size();
            phrase::const_iterator it = words.begin();
            for (auto i=0; i<tags.size(); ++i) {
                auto tag = tags.at(i);
                auto len = lens.at(i);
                auto segment = nn::join(it,
                                        it+len,
                                        emit_param.BOS,
                                        emit_param.SPACE,
                                        emit_param.EOS);
                auto H = get_emission_model(tag)->get_base();
                H->observe(segment);
                std::advance(it, len);
            }
            CHECK(it == words.end()-1);
        }

        void observe(const particle& p, const phrase& words) {
            observe(p.tags, p.lens, words);
        }

        // Example input:
        //   seq_t tags { 0, 1, 0, tagtab.EOS };
        //   seq_t lens { 1, 2, 1 };
        //   phrase words {seq1, seq2, seq3, seq4, EOS};
        void observe(const syms& tags, const syms& lens, const phrase& words) {
            CHECK(tags.size() > 0) << "no tags";
            CHECK(lens.size() > 0) << "no lens";
            CHECK(tags.size() == lens.size()) << "size mismatch! tags.len = " << tags.size() << " lens.len = " << lens.size();

            // TODO: remove this sanity check (slow)
            size_t tot_len {0};
            for (auto l : lens) tot_len += l;
            CHECK(tot_len == words.size()-1) << "bad particle; len mismatch";

            size_t total {0};
            size_t i;
            //DLOG(INFO) << "observing tags...";
            phrase context {BOS};

            auto it = words.begin();
            for(i=0; i<tags.size(); ++i) {
                auto tag  = tags[i];
                auto len  = lens[i];
                total    += len;

                // Observe tag in context
                T->observe(context, tag);

                // Update the context:
                update_context(tag, *it, context);

                // Observe the emission:
                get_emission_model(tag)->observe(it, it+len);

                //it += len;
                std::advance(it, len);
            }

            std::advance(it, 1);
            CHECK(it == words.end()) << "reached end";

            // Observe the EOS tag in the transition model
            T->observe(context, eos_tag);
        }

        void update_context(size_t tag, const syms& word, phrase& context) const {
            if (tag == context_tag) {
                context.push_back( word );
            } else {
                syms tag_vec { 0, tag, 0 };
                context.push_back( tag_vec );
            }
        }

        void remove(const particle& p, const phrase& words) {
            DLOG(INFO)               << "observing...";
            CHECK(p.tags.size() > 0) << "no tags";
            CHECK(p.lens.size() > 0) << "no lens";
            CHECK(p.tags.size() == p.lens.size()) << "size mismatch! tags.len = " << p.tags.size() << " lens.len = " << p.lens.size();

            // TODO: remove this sanity check (slow)
            size_t tot_len {0};
            for (auto l : p.lens) tot_len += l;
            CHECK(tot_len == words.size()-1) << "bad particle; len mismatch";

            //LOG(INFO) << "words.size() == " << words.size();
            //LOG(INFO) << "tags.size() == "  << p.tags.size();

            size_t total {0};
            size_t i;
            phrase context {BOS};
            size_t word_pos {0};
            auto it = words.begin();
            auto dist = std::distance(it, words.end());
            CHECK(dist == tot_len+1) << "bad distance: " << dist;
            CHECK(it != words.end()) << "bad iterator";

            for(i=0;i<p.tags.size();++i) {
                auto tag  = p.tags.at(i);
                auto len  = p.lens.at(i);
                total    += len;
                word_pos += len;

                //LOG(INFO) << "tag = " << tag << " len = " << len;

                // Remove tag in context:
                //LOG(INFO) << "removing tag in context...";
                T->remove(context, tag);

                // Update the context:
                update_context(tag, *it, context);

                // Remove the emission:
                //LOG(INFO) << "removing the emission...";
                //LOG(INFO) << "len = " << len;
                CHECK(it != words.end()) << "bad iterator";
                get_emission_model(tag)->remove(it, it+len);

                std::advance(it, len);
            }

            std::advance(it, 1);
            CHECK(it == words.end()) << "reached end";

            // Remove the EOS tag in the transition model
            T->remove(context, eos_tag);
        }

        unnormalized_discrete_distribution<size_t>
        get_transition_dist(const phrase& context) const {
            unnormalized_discrete_distribution<size_t> ret;
            for(auto tag : tagtab.get_key_set()) {
                ret.push_back_prob( tag,
                                    T->prob(context, tag) );
            }
            return ret;
        }

        discrete_distribution<std::pair<sym,bool>>
        get_between_prop(const unnormalized_discrete_distribution<size_t>& Q_trans,
                         const phrase& context,
                         const syms& obs) const {
            typedef std::pair<sym,bool> Event;
            Event e;
            discrete_distribution<Event> Q;
            for(size_t i=0; i < Q_trans.size(); ++i) {
                double tlp = Q_trans.get_log_weight(i);
                auto tag = Q_trans.get_type(i);
                e.first  = tag;

                // E-Y -- start and immediately stop
                double q1 = tlp + E.at(tag)->log_prob(obs);
                e.second = false;
                Q.push_back_log_prob( e, q1 );

                // I-Y -- start a phrase and stop later
                if(e.first != context_tag) {
                    syms prefix(obs.begin(), obs.end()-1);
                    double q2 = tlp + E.at(tag)->log_prefix_prob(prefix);
                    e.second = true;
                    Q.push_back_log_prob( e, q2 );
                }
            }

            return Q;
        }

        // The previous tag is E-X
        // We must now pick the next tag, which may be E-Y or I-Y
        double between_extend(particle& p, const syms& obs) const {
            if (obs == EOS) {
                return T->log_prob(p.context, eos_tag);
            }
            auto Q_trans = get_transition_dist(p.context);
            auto Q       = get_between_prop(Q_trans, p.context, obs);
            auto j       = Q.sample_index();
            auto e       = Q.get_type(j);
            auto tag     = e.first;
            auto start   = e.second;
            auto lp      = Q.get_log_prob(j);
            auto i       = Q_trans.get_index(tag);

            if (start) {
                p.start(tag, obs);
                return Q_trans.get_log_weight(i)-lp;
            } else {
                p.add(tag, obs);
                update_context(tag, obs, p.context);
                return Q_trans.get_log_weight(i)+E.at(tag)->log_prob(obs)-lp;
            }

            CHECK(false) << "logic error";
            return 0.0;
        }

        void swap(particle& dst, const particle& src) const {
            init(dst);
            dst.tags = src.tags;
            dst.lens = src.lens;
        }

        double baseline_inside_extend(particle& p, const syms& obs) const {
            if(obs == EOS) {
                auto tag = p.tags.back();
                auto lep = E.at(tag)->log_prob(p.words, obs); // pay emission
                update_context(tag, obs, p.context);
                auto ltp = T->log_prob(p.context, eos_tag);   // pay transition
                p.stopEOS(obs);
                return ltp + lep;
            }
            std::bernoulli_distribution d(STOP_PROB);
            auto b = d( nn::rng::get() );
            if(b) { // emit E-X: stop
                auto tag = p.tags.back();
                auto log_emit_prob = E.at(tag)->log_prob(p.words, obs);
                p.stop(obs);
                update_context(tag, obs, p.context);
                return log_emit_prob - log(STOP_PROB);
            } else { // emit I-X: continue
                p.cont(obs);
                return -log(1.0-STOP_PROB);
            }
        }

        double baseline_inside_extend(particle& p,
                                      const syms& obs,
                                      size_t t // position in the observation
                                      ) const {
            //LOG(INFO) << "[inside extend score]";

            if(obs == EOS) {
                auto tag = p.tags.back();
                auto lep = E.at(tag)->log_prob(p.words, obs); // pay emission
                update_context(tag, obs, p.context);
                auto ltp = T->log_prob(p.context, eos_tag);   // pay transition
                p.stopEOS(obs);
                return ltp + lep;
            }

            auto b = p.end_at_pos(t);

            if(b) { // emit E-X: stop
                //LOG(INFO) << "E-X";
                auto tag = p.tag_at_pos(t);
                auto log_emit_prob = E.at(tag)->log_prob(p.words, obs);
                p.words.clear();
                p.in_phrase = false;
                update_context(tag, obs, p.context);
                return log_emit_prob - log(STOP_PROB);
            } else { // emit I-X: continue
                //LOG(INFO) << "I-X";
                p.in_phrase = true;
                p.words.push_back(obs);
                return -log(STOP_PROB);
            }
        }

        double between_extend(particle& p,
                              const syms& obs,
                              size_t t) const {
            //LOG(INFO) << "[between extend score]";
            CHECK(p.words.size() == 0) << "logic";

            if (obs == EOS) {
                return T->log_prob(p.context, eos_tag);
            }

            auto Q_trans = get_transition_dist(p.context);
            auto Q       = get_between_prop(Q_trans, p.context, obs);

            std::pair<sym,bool> e;
            auto tag = p.tag_at_pos(t);
            auto len = p.len_at_pos(t);
            auto cont = (len > 1);

            e.first  = tag;
            e.second = cont;

            auto i  = Q.get_index(e);
            auto lp = Q.get_log_prob(i);
            auto tran_idx = Q_trans.get_index(tag);

            // LOG(INFO) << "i: " << i << " lp: " << lp
            //           << "; tag: " << tag
            //           << "; len: " << len;

            if ( cont ) {
                //LOG(INFO) << "[between] cont";
                p.words.push_back(obs);
                p.in_phrase = true;
                update_context(tag, obs, p.context);
                return Q_trans.get_log_weight(tran_idx) - lp;
            } else {
                //LOG(INFO) << "[between] stop";
                p.in_phrase = false;
                update_context(tag, obs, p.context);
                return Q_trans.get_log_weight(tran_idx) + E.at(tag)->log_prob(obs) - lp;
            }

            CHECK(false) << "logic error";
            return 0.0;
        }

        void log_stats() const {}

        double score(particle& p, const syms& obs, size_t t) const {
            if (t==0) CHECK(p.context.size() == 1);

            //LOG(INFO) << "[score]";

            switch(prop) {
            case FilterProposal::HYBRID: {
                if (p.in_phrase) return baseline_inside_extend(p, obs, t);
                else             return between_extend(p, obs, t);
            }
            default: CHECK(false) << "logic error";
            }
            CHECK(false) << "sanity";
        }

        double extend(particle& p, const syms& obs) const {
            switch(prop) {
            case FilterProposal::HYBRID: {
                if (p.in_phrase) return baseline_inside_extend(p, obs);
                else             return between_extend(p, obs);
            }
            default: CHECK(false) << "unsupported proposal";
            };
            CHECK(false) << "sanity";
        }

        struct writer {
            std::string prefix;
            instances test;

            sym context_tag;
            std::unordered_map<size_t, std::string> sym_desc;
            std::unordered_map<size_t, std::string> tag_desc;

            writer(std::string _prefix,
                   instances _test,
                   sym _context_tag,
                   std::unordered_map<size_t, std::string> _sym_desc,
                   std::unordered_map<size_t, std::string> _tag_desc)
                : prefix(_prefix),
                  test(_test),
                  context_tag(_context_tag),
                  sym_desc(_sym_desc),
                  tag_desc(_tag_desc) {
                LOG(INFO) << "Context tag = " << context_tag;
                CHECK(tag_desc.size() > 0);
                CHECK(sym_desc.size() > 0);
            }

            void operator()(size_t iter,
                            typename std::vector<particle>::const_iterator begin,
                            typename std::vector<particle>::const_iterator end) {
                auto instance = test.begin();
                std::string path(prefix+"_"+std::to_string(iter)+".conll");
                std::ofstream of (path);
                CHECK(of.is_open()) << "Problem opening: "     << path;
                LOG(INFO)           << "Writing predictions: " << path;
                CHECK(sym_desc.size() > 0);
                CHECK(tag_desc.size() > 0);
                for(auto particle = begin; particle != end; ++particle) {
                    write_tagging_conll(of,
                                        instance->words,
                                        particle->tags,
                                        particle->lens,
                                        instance->tags,
                                        instance->lens,
                                        context_tag,
                                        sym_desc,
                                        tag_desc);
                    instance ++;
                }
                CHECK(instance == test.end());
                of.close();
            }
        };
    };
}

#endif
