// Nicholas Andrews
// noandrews@gmail.com

#pragma once

#ifndef __LATENT_SEGMENTAL_PYP_LM_HPP__
#define __LATENT_SEGMENTAL_PYP_LM_HPP__

#include <vector>
#include <memory>
#include <unordered_map>
#include <sstream>

#include <boost/lexical_cast.hpp>

#include "mutable_symtab.hpp"
#include "discrete_distribution.hpp"
#include "data.hpp"
#include "seq_model.hpp"
#include "simple_seq_model.hpp"
#include "adapted_seq_model_prefix.hpp"
#include "latent_seq_model.hpp"
#include "log_fixed_depth_hpyp.hpp"

namespace nn {

    class LatentSegmentalHPYP {
        enum class FilterProposal { BASELINE };

        const syms BOS;
        const syms EOS;

        const sym context_tag;
        const sym eos_tag;

        typedef std::vector<obs_t>                          context_t;
        typedef Uniform<sym>                                uniform_t;
        typedef FixedDepthHPYP<sym, sym, uniform_t>         emit_t;
        typedef LatentSequenceModel<sym, uniform_t, emit_t> base_t;
        typedef LogFixedDepthHPYP<obs_t, obs_t, base_t>     model_t;

        std::unique_ptr<base_t>  base  {nullptr};
        std::unique_ptr<model_t> model {nullptr};

        FilterProposal prop { FilterProposal::BASELINE };

        const double STOP_PROB {0.75};

        const uint_str_table& symtab;
        const uint_str_table& tagtab;

        // Filter diagnostics:
        size_t n_sampled_between_start {0};
        size_t n_sampled_between_stop  {0};
        size_t n_sampled_inside_stop   {0};
        size_t n_sampled_inside_cont   {0};

        // Emission model parameters:
        const size_t bos_sym;
        const size_t eos_sym;
        const size_t space_sym;

    public:
        template<typename Corpus>
        LatentSegmentalHPYP(const Corpus& corpus)
            : BOS(corpus.get_bos_obs()),
              EOS(corpus.get_eos_obs()),
              context_tag { corpus.get_other_key() },
              eos_tag { corpus.tagtab.size() },
              symtab(corpus.symtab),
              tagtab(corpus.tagtab),
              bos_sym(corpus.get_bos_key()),
              eos_sym(corpus.get_eos_key()),
              space_sym(corpus.get_space_key())
            {
                base = std::make_unique<base_t>(corpus.symtab.size(),
                                                corpus.tagtab.get_key_set());
                model = std::make_unique<model_t>(base.get());
            }

        LatentSegmentalHPYP(
            LatentSegmentalHPYP const&
            ) = delete;
        LatentSegmentalHPYP& operator=(
            LatentSegmentalHPYP const&
            ) = delete;

        struct particle {
            std::vector<syms> words;    // words in the current phrase

            bool in_phrase {false};     // if we're extending a phrase
            bool done      {false};     // reached EOS
            syms tags;                  // predicted tags
            std::vector<size_t> lens;   // span (# of words) for each tag

            context_t context;          // all previous words
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
                CHECK(false);
                return std::numeric_limits<size_t>::max();
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
                CHECK(idx < tags.size()) << "idx: " << idx
                                         << " tags.size(): " << tags.size();
                return tags.at(idx);
            }

            size_t len_at_pos(size_t t) const {
                auto idx = pos_idx(t);
                CHECK(idx < lens.size()) << "idx: " << idx
                                         << " lens.size(): " << lens.size();
                return lens.at(idx);
            }

            void start(size_t tag, syms w) {
                in_phrase = true;

                tags.push_back(tag);
                lens.push_back(1);

                words.push_back(w);
            }

            void add(size_t tag) {
                in_phrase = false;
                tags.push_back(tag);
                lens.push_back(1);
            }

            void stop() {
                in_phrase = false;
                lens.back()++;
                words.clear();
            }

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

        bool consistent() { return true; }

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

        void log_stats() const {
          LOG(INFO) << "Base log stats:";
            base->log_stats();
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
            //p.context.push_back(BOS);
            p.context_tag = context_tag;
        }

        void resample_hyperparameters() {}

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
            CHECK(total == input.size()-1) << "size mismatch! " << total
                                           << " vs " << (input.size()-1);
            CHECK(p.tags.size() == p.lens.size()) << "size mismatch: (tags) "
                                                  << p.tags.size() << " (lens) "
                                                  << p.lens.size();
            return true;
        }

      void remove(const instance& i) {
        remove(i.tags, i.lens, i.words);
      }

        void remove(const particle& p, const phrase& words) {
            remove(p.tags, p.lens, words);
        }

        void remove(const syms& tags, const syms& lens, const phrase& words) {
          CHECK(tags.size() > 0) << "no tags";
          CHECK(lens.size() > 0) << "no lens";
          CHECK(tags.size() == lens.size()) << "size mismatch! tags.len = "
                                            << tags.size()
                                            << " lens.len = " << lens.size();

          size_t tot_len {0};
          for (auto l : lens) tot_len += l;
            CHECK(tot_len == words.size()-1) << "bad particle; len mismatch";

            size_t total {0};
            size_t i;
            //phrase context {BOS};
            std::vector<obs_t> context; // TODO: add BOS back

            auto it = words.begin();
            for (i=0; i<tags.size(); ++i) {
              auto tag  = tags[i];
              auto len  = lens[i];
              auto segment = join(it, it+len);
              auto obs = std::make_pair(tag, segment);
              total += len;
              model->remove(context, obs);
              context.push_back(obs);
              std::advance(it, len);
            }

            std::advance(it, 1);
            CHECK(it == words.end()) << "reached end";
        }

      void observe(const instance& i) {
        observe(i.tags, i.lens, i.words);
      }

        void observe(const particle& p, const phrase& words) {
            observe(p.tags, p.lens, words);
        }

        void observe_gazetteer(const syms& tags, const syms& lens, const phrase& words) {
            CHECK(false) << "unimplemented";
        }

            void observe(const syms& tags, const syms& lens, const phrase& words) {
            CHECK(tags.size() > 0) << "no tags";
            CHECK(lens.size() > 0) << "no lens";
            CHECK(tags.size() == lens.size()) << "size mismatch! tags.len = "
                                              << tags.size()
                                              << " lens.len = " << lens.size();

            size_t tot_len {0};
            for (auto l : lens) tot_len += l;
            CHECK(tot_len == words.size()-1) << "bad particle; len mismatch";

            size_t total {0};
            size_t i;
            //phrase context {BOS};
            std::vector<obs_t> context; // TODO: add BOS back

            auto it = words.begin();
            for (i=0; i<tags.size(); ++i) {
                auto tag  = tags[i];
                auto len  = lens[i];
                auto segment = join(it, it+len);
                auto obs = std::make_pair(tag, segment);
                total += len;
                model->observe(context, obs);
                context.push_back(obs);
                std::advance(it, len);
            }

            std::advance(it, 1);
            CHECK(it == words.end()) << "reached end";
        }

        void swap(particle& dst, const particle& src) const {
            init(dst);
            dst.tags = src.tags;
            dst.lens = src.lens;
        }

        discrete_distribution<std::pair<size_t,bool>>
        get_base_between_prop(const context_t& context, const syms& seq) const {
            discrete_distribution<std::pair<size_t,bool>> Q;
            for (auto t : tagtab.get_key_set()) {
                auto e = std::make_pair(t, true);
                auto lq = base->log_prob(t, seq);
                Q.push_back_log_prob(e, lq);
                if (t != context_tag) {
                    auto prefix = seq;
                    e.second = false;
                    prefix[prefix.size()-1] = space_sym;
                    lq = base->log_prob(t, prefix);
                    Q.push_back_log_prob( e, lq );
                }
            }
            return Q;
        }

        discrete_distribution<std::pair<size_t,bool>>
        get_between_prop(const context_t& context, const syms& seq) const {
            discrete_distribution<std::pair<size_t,bool>> Q;
            // for (auto t : tagtab.get_key_set()) {
            //     auto e = std::make_pair(t, true);
            //     auto obs = std::make_pair(t, seq);
            //     auto lq = model->log_prob(context, obs); // log_cache + log_new
            //     Q.push_back_log_prob(e, lq);
            //     if (t != context_tag) {
            //         e.second = false;
            //         auto prefix = seq;
            //         prefix[prefix.size()-1] = space_sym;
            //         auto total_lq = model->log_new_prob(context, prefix); // prefix prob
            //         auto ret = base->match(t, prefix);
            //         for (auto r = ret.first; r != ret.second; ++r) {
            //             auto obs = std::make_pair(t, r->first);
            //             auto lq = model->log_cache_prob(context, obs);
            //             log_plus_equals(total_lq, lq);
            //         }
            //         Q.push_back_log_prob( e, total_lq );
            //     }
            // }
            CHECK(false) << "TODO";
            return Q;
        }

        // The previous tag is E-X
        // We must now pick the next tag, which may be E-Y or I-Y
        double between_extend(particle& p, const syms& seq) const {
            if (seq == EOS) {
                auto obs = std::make_pair(context_tag, EOS);
                return model->log_prob(p.context, obs);
            }
            auto Q    = get_between_prop(p.context, seq);
            auto j    = Q.sample_index();
            auto e    = Q.get_type(j);
            auto tag  = e.first;
            auto stop = e.second;
            auto lq   = Q.get_log_prob(j);
            if (!stop) {
                p.start(tag, seq);
                return - lq;
            } else {
                auto obs = std::make_pair(tag, seq);
                auto lp = model->log_prob(p.context, obs);
                update_context(obs, p.context);
                p.add(tag);
                return lp - lq;
            }
        }

        seq_t join(const std::vector<seq_t>& seqs) const {
            return nn::join(seqs.begin(), seqs.end(),
                            bos_sym, space_sym, eos_sym);
        }

        seq_t join(typename std::vector<syms>::const_iterator start,
                   typename std::vector<syms>::const_iterator stop) const {
            return nn::join(start, stop, bos_sym, space_sym, eos_sym);
        }

        void update_context(size_t tag,
                            const syms& seq,
                            context_t& context) const {
            update_context(std::make_pair(tag,seq), context);
        }

        void update_context(const obs_t& o, context_t& context) const {
            context.push_back(o);
        }

        double inside_extend(particle& p, const syms& seq) const {
            if (seq == EOS) {
                auto tag = p.tags.back();
                auto obs = std::make_pair(tag, join(p.words));
                auto eos = std::make_pair(context_tag, EOS);
                auto ret = model->log_prob(p.context, obs);
                update_context(obs, p.context);
                ret += model->log_prob(p.context, eos);
                p.stopEOS(seq);
                return ret;
            }
            std::bernoulli_distribution d(STOP_PROB);
            auto b = d( nn::rng::get() );
            if(b) { // emit E-X: stop
                auto tag = p.tags.back();
                p.words.push_back(seq);
                auto obs = std::make_pair(tag, join(p.words));
                auto lp = model->log_prob(p.context, obs);
                auto lq = log(STOP_PROB);
                update_context(obs, p.context);
                p.stop();
                return lp - lq;
            } else { // emit I-X: continue
                p.cont(seq);
                auto lq = log(1.0-STOP_PROB);
                return -lq;
            }
        }

        double inside_extend(particle& p,
                             const syms& obs,
                             size_t t // position in the observation
            ) const {
            CHECK(false) << "unimplemented";
            return 0.0;
        }

        double between_extend(particle& p,
                              const syms& obs,
                              size_t t) const {
            CHECK(false) << "unimplemented";
            return 0.0;
        }

        double score(particle& p,
                     const syms& obs,
                     size_t t) const {
            switch(prop) {
            case FilterProposal::BASELINE: {
                if (p.in_phrase) return inside_extend(p, obs, t);
                else             return between_extend(p, obs, t);
            }
            default: CHECK(false) << "logic error";
            }
            CHECK(false) << "sanity";
        }

        double extend(particle& p, const syms& obs) const {
            switch(prop) {
            case FilterProposal::BASELINE: {
                if (p.in_phrase) return inside_extend(p, obs);
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
