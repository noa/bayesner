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

#ifndef __NN_HIDDEN_SEQUENCE_MEMOIZER_HPP__
#define __NN_HIDDEN_SEQUENCE_MEMOIZER_HPP__

#include <vector>

#include <nn/mutable_symtab.hpp>
#include <nn/adapted_seq_model.hpp>

namespace nn {
    enum class HSMProposal { BASELINE };

    //template<typename base_type = Uniform<sym>,
    template<typename base_type = HashIntegralMeasure<sym>,
             typename tran_type = FixedDepthHPYP<sym,
                                                 syms,
                                                 base_type>,
             typename emit_type = adapted_seq_model<>
             >
    class hidden_sequence_memoizer {
        enum class TType { START, EXTEND };

        HSMProposal prop { HSMProposal::BASELINE };

        const uint_str_table& symtab;
        const uint_str_table& tagtab;

        std::unique_ptr<base_type> H {nullptr};
        std::unique_ptr<tran_type> T {nullptr};
        std::unordered_map<sym, std::unique_ptr<emit_type>> E;

        bool frozen { false };

        const syms BOS; // beginning of string obs
        const syms EOS; // end of string obs

        const sym context_tag;  // model  "other" tag index

        const sym context_idx;
        const sym eos_idx;

        size_t n_transition_observed {0};
        size_t n_emission_observed   {0};

        static constexpr double default_emit_adaptor_alpha    {1.0};
        static constexpr double default_emit_adaptor_discount {0.5};

    public:
        static std::unordered_map<size_t,double> get_default_emit_adaptor_alpha(
            const uint_str_table& tag_table
            ) {
            std::unordered_map<size_t,double> ret;
            for(auto tag : tag_table.get_key_set()) {
                ret[tag] = default_emit_adaptor_alpha;
            }
            return ret;
        }

        static std::unordered_map<size_t,double> get_default_emit_adaptor_discount(
            const uint_str_table& tag_table
            ) {
            std::unordered_map<size_t,double> ret;
            for(auto tag : tag_table.get_key_set()) {
                ret[tag] = default_emit_adaptor_discount;
            }
            return ret;
        }

        template<typename Corpus>
        hidden_sequence_memoizer(const Corpus& corpus) :
            hidden_sequence_memoizer(corpus,
                                     get_default_emit_adaptor_alpha(corpus.tagtab),
                                     get_default_emit_adaptor_discount(corpus.tagtab),
                                     1.0) {}

        template<typename Corpus>
        hidden_sequence_memoizer(const Corpus& corpus,
                                 std::unordered_map<size_t,double> emit_adaptor_alpha,
                                 std::unordered_map<size_t,double> emit_adaptor_discount,
                                 double tran_alpha) :
            BOS(corpus.get_bos_obs()),
            EOS(corpus.get_eos_obs()),
            symtab(corpus.symtab),
            tagtab(corpus.tagtab),
            context_tag(corpus.get_other_key()),
            context_idx(corpus.get_other_key()*2),
            eos_idx(corpus.get_other_key()*2+1) {

            typename emit_type::param emit_param;
            emit_param.nsyms = symtab.size();
            emit_param.BOS   = corpus.get_bos_key();
            emit_param.EOS   = corpus.get_eos_key();
            emit_param.SPACE = corpus.get_space_key();

            LOG(INFO) << "[HSM] EOS idx     = " << eos_idx;
            LOG(INFO) << "[HSM] context idx = " << context_idx;

            for(auto tag : tagtab.get_key_set()) {
                emit_param.alpha    = emit_adaptor_alpha.at(tag);
                emit_param.discount = emit_adaptor_discount.at(tag);
                add_emission_model(tag, emit_param);
            }
            auto num_idx = num_emission_model();
            freeze(); // don't add any emission models past this point

            // Add transition model
            H = std::make_unique<base_type>();

            // Prior
            for(auto tag=0; tag < tagtab.size(); ++tag) {
                auto idx = tag_idx(tag, TType::START);
                if(tag == context_tag) {
                    H->add(idx, 20.0);  // "other"
                    H->add(idx+1, 1.0); // EOS
                } else {
                    H->add(idx, 5.0);
                    idx = tag_idx(tag, TType::EXTEND);
                    H->add(idx, 2.5);
                }
            }
            CHECK(H->cardinality() == tagtab.size()*2);

            auto tran_model = std::make_unique<tran_type>( H.get() );
            add_transition_model( std::move(tran_model) );
        }

        hidden_sequence_memoizer(hidden_sequence_memoizer const&)            = delete;
        hidden_sequence_memoizer& operator=(hidden_sequence_memoizer const&) = delete;

        struct particle {
            std::vector<size_t> tags;
            std::vector<syms> context;

            void add(size_t tag) {
                tags.push_back(tag);
            }
        };

        auto get_context_idx() -> decltype(context_idx) const {
            return context_idx;
        }

        auto get_eos_idx() -> decltype(eos_idx) const { return eos_idx;  }
        size_t num_emission_model()             const { return E.size(); }

        void init(particle& p) const {
            p.tags.clear();
            p.tags.reserve(128);
            CHECK(p.tags.size() < 1);
            p.context.clear();
            p.context.reserve(128);
            p.context.push_back(BOS);
            CHECK(p.context.size() == 1);
        }

        void add_transition_model( std::unique_ptr<tran_type> _T ) {
            T = std::move( _T );
        }

        void add_emission_model( sym tag,
                                 typename emit_type::param emit_param ) {
            CHECK(!frozen) << "usage";
            auto tag_str        = tagtab.val(tag);
            LOG(INFO) << "[HSM] emission model " << tag << " (" << tag_str << ")";
            auto emit_model = std::make_unique<emit_type>(emit_param);
            E.emplace(tag, std::move(emit_model));
        }

        emit_type* get_emission_model(sym tag) const { return E.at(tag).get(); }
        bool has_emission_model(sym tag)             { return E.count(tag);    }

        void       freeze()                          { frozen = true;          }
        tran_type* get_transition_model()            { return T.get();         }

        void set_proposal(HSMProposal p)             { prop = p;               }

        size_t tag_idx(size_t tag, TType t) const {
            CHECK(!(tag == context_tag && t == TType::EXTEND));
            if(t == TType::START) {
                // 0 -> 0
                // 1 -> 2
                // 2 -> 4
                return tag*2;
            } else {
                // 0 -> 1
                // 1 -> 3
                // 2 -> 5
                return tag*2+1;
            }
        }

        size_t idx_tag(size_t idx) const {
            // 0 -> 0
            // 1 -> 0
            // 2 -> 1
            // 3 -> 1
            return idx / 2;
        }

        std::map<size_t, size_t> get_idx_tag_map() const {
            std::map<size_t, size_t> ret;
            for (auto tag=0; tag<tagtab.size(); ++tag) {
                auto idx = tag_idx(tag, TType::START);
                ret[idx] = tag;
                if(tag != context_tag) {
                    idx = tag_idx(tag, TType::EXTEND);
                    ret[idx] = tag;
                }
            }
            return ret;
        }

        std::set<size_t> get_begin_idx() {
            std::set<size_t> ret;
            for(auto tag=0; tag<tagtab.size(); ++tag) {
                if(tag != context_tag) {
                    auto idx = tag_idx(tag, TType::START);
                    ret.insert(idx);
                }
            }
            return ret;
        }

        std::set<size_t> get_extend_idx() {
            std::set<size_t> ret;
            for(auto tag=0; tag<tagtab.size(); ++tag) {
                if(tag != context_tag) {
                    auto idx = tag_idx(tag, TType::EXTEND);
                    ret.insert(idx);
                }
            }
            return ret;
        }

        size_t name_len(std::vector<size_t>::const_iterator start,
                        std::vector<size_t>::const_iterator end) const {
            auto start_idx = *start;
            auto len = 1;
            for(auto it = start+1; it != end; ++it) {
                auto idx = *it;
                if(idx == start_idx + 1) {
                    ++len;
                } else {
                    break;
                }
            }
            return len;
        }

        std::vector<size_t> get_tags(const particle& p) const {
            std::vector<size_t> ret;
            auto it = p.tags.begin();
            while(it != p.tags.end()) {
                auto idx = *it;
                if(idx == context_idx) {
                    ret.push_back( context_tag );
                    std::advance(it, 1);
                } else if(idx % 2 == 0) {
                    auto type = idx_tag( idx );
                    ret.push_back( type );
                    auto len = name_len(it, p.tags.end());
                    std::advance(it, len);
                } else {
                    CHECK(false) << "logic";
                }
            }
            return ret;
        }

        std::vector<size_t> get_lens(const particle& p) const {
            std::vector<size_t> ret;
            auto it = p.tags.begin();
            while(it != p.tags.end()) {
                auto idx = *it;
                if(idx == context_tag) {
                    ret.push_back(1);
                    std::advance(it, 1);
                } else {
                    auto len = name_len(it, p.tags.end());
                    ret.push_back(len);
                    std::advance(it, len);
                }
            }
            return ret;
        }

        std::vector<particle> make_particles(instances corpus) {
            std::vector<particle> ret;
            for(instance i : corpus) {
                ret.push_back( make_particle(i.tags, i.lens) );
            }
            return ret;
        }

        particle make_particle(const syms& tags,
                               const syms& lens) const {
            particle p;
            init(p);
            for(auto i=0; i<tags.size(); ++i) {
                auto tag = tags.at(i);
                auto len = lens.at(i);
                CHECK(len > 0) << "bad len";
                auto idx = tag_idx(tag, TType::START);
                CHECK(idx != eos_idx) << "logic error";
                p.tags.push_back( idx );
                for(auto j=1; j<len; j++) {
                    CHECK(tag != context_tag) << "logic error";
                    auto idx = tag_idx(tag, TType::EXTEND);
                    CHECK(idx != eos_idx) << "logic error";
                    p.tags.push_back( idx );
                }
            }
            for(auto i=0; i<p.tags.size(); ++i) {
                auto idx = p.tags.at(i);
                CHECK(idx != eos_idx) << "logic error";
            }
            return p;
        }

        void observe_gazetteer(const syms& tags,
                               const syms& lens,
                               const phrase& words) {
            auto it = words.begin();
            auto tot_len {0};
            for(size_t i=0; i<tags.size(); ++i) {
                auto tag = tags.at(i);
                auto len = lens.at(i);
                tot_len += len;
                auto start_idx = tag_idx(tag, TType::START);
                for(size_t j=0; j<len; ++j) {
                    get_emission_model(tag)->get_base()->observe(*it);
                    it = std::next(it);
                }
                if (tag == context_tag) continue;
                phrase context {start_idx};
                auto extend_idx = tag_idx(tag, TType::EXTEND);
                for(size_t j=1; j<len; ++j) {
                    T->observe(context, extend_idx);
                    n_transition_observed ++;
                    syms tag_seq { extend_idx };
                    context.push_back(tag_seq);
                }
                T->observe(context, context_idx);
                n_transition_observed ++;
            }
            CHECK(tot_len == words.size()-1) << tot_len << " vs " << words.size();
            CHECK(it == words.end()-1);
        }

        void observe(const syms& tags,
                     const syms& lens,
                     const phrase& words) {
            // create the particle
            particle p = make_particle(tags, lens);
            CHECK(p.tags.size() == words.size()-1) << "size mismatch";
            observe(p, words); // observing particle
        }

        bool consistent() const {
            auto nc = T->totalCustomers();
            CHECK(nc == n_transition_observed) << "customer count mistmatch: "
                                               << nc << " vs "
                                               << n_transition_observed;
            return true;
        }

        template<typename Context>
        void update_context(Context &c, size_t idx, syms obs) const {
            if(idx == context_idx) {
                c.push_back(obs);
            } else {
                syms tag_seq {idx};
                c.push_back(tag_seq);
            }
        }

        void observe(const particle& p, const phrase& words) {
            const auto& tags = p.tags;
            phrase context {BOS};
            for(auto i=0; i<tags.size(); ++i) {
                auto idx  = tags.at(i);
                auto tag  = idx_tag(idx);
                auto word = words.at(i);
                CHECK(idx != eos_idx) << "logic error";

                // Observe tag in context
                T->observe(context, idx);
                ++n_transition_observed;

                // Update the transition model context
                update_context(context, idx, word);
                CHECK(context.size() > i);

                // Observe the emission
                get_emission_model(tag)->observe(word);
                ++n_emission_observed;
            }

            // Observe the final EOS tag
            T->observe(context, eos_idx);
            ++n_transition_observed;
        }

        void log_stats() {
            LOG(INFO) << "n_transition_observed = " << n_transition_observed;
            LOG(INFO) << "n_emission_observed   = " << n_emission_observed;
        }

        void remove(const particle& p, const phrase& words) {
            const auto& tags = p.tags;
            phrase context {BOS};
            for(auto i=0; i<tags.size(); ++i) {
                auto idx  = tags.at(i);
                auto tag  = idx_tag(idx);
                auto word = words.at(i);

                // Observe tag in context:
                T->remove(context, idx);

                // Update the transition model context:
                update_context(context, idx, word);

                // Observe the emission:
                get_emission_model(tag)->remove(word);
            }

            // Observe the final EOS tag
            T->remove(context, eos_idx);
        }

        unnormalized_discrete_distribution<size_t>
        tran_dist(particle& p) const {
            CHECK(p.context.size() > 0);
            unnormalized_discrete_distribution<size_t> ret;
            auto prev_tag = idx_tag(p.tags.back());
            for(auto tag : tagtab.get_key_set()) {
                auto idx = tag_idx(tag, TType::START);
                ret.push_back_log_prob(idx,
                                       T->log_prob(p.context, idx));

                // Extend tag:
                if(tag==prev_tag && tag != context_tag) {
                    idx = tag_idx(tag, TType::EXTEND);
                    ret.push_back_log_prob(idx,
                                           T->log_prob(p.context, idx));
                }
            }
            return ret;
        }

        void log_particle(const particle& p) const {
            std::string context;
            for(auto it = p.context.begin();
                it != p.context.end();
                ++it) {
                auto c = *it;
                if(c.size() < 2) {
                    auto idx = c[0];
                    auto tag = idx_tag(idx);
                    if(idx % 2 == 0) context += (" B-" + tagtab.val(tag));
                    else             context += (" I-" + tagtab.val(tag));
                } else {
                    context += (" " + get_string(c, symtab.get_map()));
                }
            }
            LOG(INFO) << "Context:";
            LOG(INFO) << context;
            std::string tags;
            for(auto it = p.tags.begin();
                it != p.tags.end();
                ++it) {
                auto idx = *it;
                auto tag = idx_tag(idx);
                auto tag_str = tagtab.val(tag);
                tags += ("(" + std::to_string(idx) + ", " + tag_str + ")");
            }
            LOG(INFO) << "Tags:";
            LOG(INFO) << tags;
        }

        double baseline_extend(particle& p, const syms& obs) const {
            if(obs == EOS) {
                return T->log_prob(p.context, eos_idx);
            }
            auto p_t = tran_dist(p);
            decltype(p_t) p_e;
            decltype(p_t) Q;
            auto lnZ { -std::numeric_limits<double>::infinity() };
            for(auto i = 0; i < p_t.size(); ++i) {
                auto idx = p_t.get_type(i);
                auto ltp = p_t.get_log_weight(i);
                auto tag = idx_tag(idx);
                auto lep = E.at(tag)->log_prob(obs);
                p_e.push_back_log_prob(idx, lep);
                auto lw = ltp + lep;
                lnZ = log_add(lnZ, lw);
                Q.push_back_log_prob(idx, lw);
            }
            auto i = Q.sample_index();
            auto idx = Q.get_type(i);
            update_context(p.context, idx, obs);
            CHECK(idx != eos_idx) << "logic error";
            p.add(idx);
            return lnZ;
        }

        void resample_hyperparameters() {}

        void log_stats() const {}

        double baseline_score(particle & p,
                              const syms& obs,
                              size_t t) const {
            if(obs == EOS) {
                return T->log_prob(p.context, eos_idx);
            }
            auto p_t = tran_dist(p);
            auto lnZ { -std::numeric_limits<double>::infinity() };
            for(auto i = 0; i < p_t.size(); ++i) {
                auto idx = p_t.get_type(i);
                auto tag = idx_tag(idx);
                auto ltp = p_t.get_log_weight(i);
                auto lep = E.at(tag)->log_prob(obs);
                auto lw = ltp + lep;
                lnZ = log_add(lnZ, lw);
            }
            auto idx = p.tags.at(t);
            update_context(p.context, idx, obs);
            return lnZ;
        }

        void swap(particle& dst, const particle& src) const {
            init(dst);
            dst.tags = src.tags;
        }

        double extend(particle& p, const syms& obs) const {
            switch(prop) {
            case HSMProposal::BASELINE: {
                return baseline_extend(p, obs);
            }
            default: CHECK(false) << "unsupported proposal";
            };
            CHECK(false) << "sanity";
        }

        double score(particle& p, const syms& obs, size_t t) const {
            switch(prop) {
            case HSMProposal::BASELINE: {
                return baseline_score(p, obs, t);
            }
            default: CHECK(false) << "unsupported proposal";
            };
            CHECK(false) << "sanity";
        }

        struct writer {
            std::string prefix;
            const instances& test;

            sym context_tag;
            const std::unordered_map<size_t, std::string>& sym_desc;
            const std::unordered_map<size_t, std::string>& tag_desc;

            writer(std::string _prefix,
                   const instances& _test,
                   sym _context_tag,
                   const std::unordered_map<size_t, std::string>& _sym_desc,
                   const std::unordered_map<size_t, std::string>& _tag_desc)
                : prefix(_prefix),
                  test(_test),
                  context_tag(_context_tag),
                  sym_desc(_sym_desc),
                  tag_desc(_tag_desc) {}

            void operator()(size_t iter,
                            typename std::vector<particle>::const_iterator begin,
                            typename std::vector<particle>::const_iterator end) {
                CHECK(false) << "unimplemented";
            }
        };
    };
}

#endif
