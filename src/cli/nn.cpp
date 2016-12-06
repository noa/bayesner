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

#include <memory>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <set>
#include <omp.h>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/timer/timer.hpp>

#include <gflags/gflags.h>

#include <nn/log.hpp>
#include <nn/pgibbs.hpp>
#include <nn/timing.hpp>
#include <nn/utils.hpp>
#include <nn/reader.hpp>
#include <nn/data.hpp>
#include <nn/evaluation.hpp>
#include <nn/generic_filter.hpp>
#include <nn/adapted_seq_model.hpp>
#include <nn/hidden_sequence_memoizer.hpp>
#include <nn/segmental_sequence_memoizer.hpp>
//#include <nn/latent_segmental_pyp_lm.hpp>

#include <cereal/types/memory.hpp>
#include <cereal/archives/binary.hpp>

using namespace nn;

DEFINE_string(unlabeled, "", "path to unlabeled data for training");
DEFINE_string(train, "data/conll/eng/train.utf8", "path to training data");
DEFINE_string(test,  "data/conll/eng/valid.utf8", "path to test data");
DEFINE_string(gazetteer, "", "path to gazetteer");
DEFINE_string(mixed, "", "path to partially annotated data for inference");
DEFINE_string(out_path, "pred.txt", "output path for predictions");
DEFINE_string(model_path, "model.ser", "path to save/load model");
DEFINE_uint64(max_gazetteer_train, 50000, "maximum number of gazetteer items");
DEFINE_string(bos, "<bos>", "beginning-of-string symbol");
DEFINE_string(eos, "<eos>", "end-of-string symbol");
DEFINE_string(space, "<space>", "space symbol");
DEFINE_string(unk, "<unk>", "unknown symbol");
DEFINE_string(context_tag, "O", "tag to use for context words");
DEFINE_uint64(nseeds, 1024, "number of seeds");
DEFINE_uint64(nthreads, 16, "number of threads");
DEFINE_bool(entity_only, false, "only train the entity models then quit");
DEFINE_bool(classify, false, "assume the segmentation is given");
DEFINE_bool(train_only, false, "only do training");
DEFINE_bool(test_only, false, "only run test");                            // TODO
DEFINE_bool(predict_loop, false, "predict tags for each line from stdin"); // TODO
DEFINE_bool(print_errors, false, "display errors");
DEFINE_bool(fine_grained_context, false, "predict fine-grained context tags");
DEFINE_uint64(status_interval, 500, "status interval (instances)");
DEFINE_uint64(sec_status_interval, 5, "status interval (seconds)");
DEFINE_bool(crossval, false, "cross validate train");
DEFINE_uint64(nfolds, 10, "number of cross val folds");
DEFINE_string(model, "seg", "hsm | seg | hseg");
DEFINE_string(parameters, "", "parameters for each named-entity model");
DEFINE_double(emission_adaptor_discount, 0.75, "Emission discount hyper");
DEFINE_double(emission_adaptor_alpha, 0.1, "Emission concentration hyper");
DEFINE_string(transition_model, "tag", "type of tag transition model");
DEFINE_string(emission_model, "simple_adapted", "type of phrasal model");
DEFINE_string(other_model, "all_other", "what to do with dictionary");
DEFINE_bool(observe_dictionary, true, "if the dictionary is observed");
DEFINE_bool(train_gazetteer_model, false, "if an entity model is trained");
DEFINE_string(resampling, "none", "which resampling method to use");
DEFINE_uint64(nparticles, 16, "number of particles used for gazetteer filter");
DEFINE_uint64(nmcmc_iter, 10, "number of MCMC iterations");
DEFINE_string(mode, "smc", "smc | pgibbs");

template<typename Model>
std::unique_ptr<Model> load_model() {
    std::string fn = FLAGS_model_path;
    std::unique_ptr<Model> model;
    {
        std::ifstream is(fn, std::ios::binary);
        cereal::BinaryInputArchive iarchive(is);
        iarchive(model);
    }
    return model;
}

bool check_output(nn::instances test, std::string path) {
    std::string line;
    std::ifstream infile;
    infile.open(path);
    if(!infile) { LOG(FATAL) << "Error reading: [" << path << "]"; }
    size_t num_output_tags {0};
    while (std::getline(infile, line)) {
        std::vector<std::string> toks;
        boost::split(toks, line, boost::is_any_of(" \n\t"));
        if (toks.size() == 3) { num_output_tags ++; }
    }
    infile.close();
    size_t num_test_tags {0};
    for(auto instance : test) {
        for (auto i = 0; i<instance.tags.size(); ++i) {
            auto len = instance.lens.at(i);
            num_test_tags += len;
        }
    }
    CHECK(num_output_tags == num_test_tags) << "size mismatch; num_output_tags = "
                                            << num_output_tags
                                            << " num_test_tags = "
                                            << num_test_tags;
    return true;
}

template<typename Model,
         typename Corpus = CoNLLCorpus<>
         >
std::unique_ptr<Model> train_model(const Corpus& corpus,
                                   const instances& train,
                                   const instances& gaz,
                                   const instances& unlabeled) {
    LOG(INFO) << "Observing training data: " << FLAGS_train;
    tic();
    auto m = std::make_unique<Model>(corpus);
    if (gaz.size() > 0) {
        LOG(INFO) << "Observing gazetteer...";
        histogram<sym> gaz_type_counts;
        for(const auto& g : gaz) {
            m->observe_gazetteer(g.tags, g.lens, g.words);
            for(auto i = 0; i < g.tags.size(); ++i) {
                auto tag  = g.tags.at(i);
                gaz_type_counts.observe(tag);
            }
        }
        LOG(INFO) << "Gazetteer type stats:";
        LOG(INFO) << gaz_type_counts.str();
    }
    histogram<size_t> tag_hist;
    size_t ntag = 0;
    double alen = 0;
    for (const auto& ex : train) {
        m->observe(ex.tags, ex.lens, ex.words);
        for(auto i=0; i<ex.tags.size(); ++i) {
            auto tag = ex.tags.at(i);
            auto len = ex.lens.at(i);
            alen += static_cast<double>(len);
            ntag += 1;
            tag_hist.observe(tag);
        }
    }
    m->log_stats();
    LOG(INFO) << "TRAIN tag histogram:";
    LOG(INFO) << tag_hist.count_str();
    alen /= static_cast<double>(ntag);
    LOG(INFO) << "TRAIN mean tag len: "   << alen;
    LOG(INFO) << "...done in: "           << prettyprint(toc());
    LOG(INFO) << "Serializing model to: " << FLAGS_model_path;
    {
        std::string fn = FLAGS_model_path;
        std::ofstream os(fn, std::ios::binary);
        cereal::BinaryOutputArchive oarchive(os);
        oarchive(m);
    }
    LOG(INFO) << "Done.";
    return m;
}

template<typename Model>
void test_model(Model* model,
                const instances& test,
                std::string out_path) {
  typedef typename Model::particle Particle;
  typedef generic_filter<Model, Particle> Filter;
  typename Filter::settings filter_config;
  filter_config.num_particles = FLAGS_nparticles;
  Filter filter(filter_config, *model);
  LOG(INFO) << "Writing predictions on test data: " << FLAGS_test;
  std::ofstream of(out_path);
  size_t idx  = 0;
  size_t ntag = 0;
  double alen = 0;
  double azf  = 0;
  double aess = 0;
  CHECK(of.is_open()) << "problem opening: " << out_path;
  histogram<size_t> tag_hist;
  progress_bar prog(test.size(), FLAGS_sec_status_interval);
  tic();
  for (const auto& i : test) {
    auto p  = filter.sample(i.words);
    azf    += filter.get_zero_frac();
    aess   += filter.sys.ess;
    auto tags = model->get_tags(p);
    auto lens = model->get_lens(p);

    for(auto len : lens) {
      alen += static_cast<double>(len);
      ntag += 1;
    }

    for(auto tag : tags) {
      tag_hist.observe(tag);
    }
    
    write_tagging_conll(of, i.words,
                        tags, lens,
                        i.tags, i.lens,
                        model->get_corpus().get_other_key(),
                        model->get_corpus().symtab.get_map(),
                        model->get_corpus().tagtab.get_map());

    if(idx % FLAGS_status_interval == 0) {
      azf /= static_cast<double>(FLAGS_status_interval);
      aess /= static_cast<double>(FLAGS_status_interval);
      alen /= static_cast<double>(ntag);
      ntag = 0;
      alen = 0;
      azf = 0;
      aess = 0;
    }
    prog++;
    idx++;
  }
  LOG(INFO) << "TEST tag histogram:";
  LOG(INFO) << tag_hist.count_str();
  LOG(INFO) << "...done in: " << prettyprint(toc());
  of.close();
  LOG(INFO) << "Predictions written to: " << out_path;
}

template<typename Model,
         typename Corpus = CoNLLCorpus<>
         >
void run_inference(const instances& train,
                   const instances& gaz,
                   const instances& unlabeled,
                   const instances& test,
                   std::string out_path,
                   const Corpus& corpus) {
    LOG(INFO) << "mode: " << FLAGS_mode;
    //LOG(INFO) << "Initializing filter...";

    if (FLAGS_mode == "smc") {
        LOG(INFO) << "Inference: SMC";
        std::unique_ptr<Model> model = train_model<Model, Corpus>(corpus,
                                                                  train,
                                                                  gaz,
                                                                  unlabeled);
        CHECK(model->consistent()) << "inconsistent model state";
        test_model<Model, Corpus>(model.get(), test, out_path);
    } else if (FLAGS_mode == "pgibbs") {
        LOG(INFO) << "Inference: particle Gibbs";
        Model model(corpus);
        typedef typename Model::particle Particle;
        typedef generic_filter<Model, Particle> Filter;
        typename Filter::settings filter_config;
        typedef particle_gibbs<Particle, Filter, Model, instances> PG;
        typename PG::settings pg_config;
        auto gold_particles = model.make_particles(test);
        pg_config.num_iter = FLAGS_nmcmc_iter;
        Filter filter(filter_config, model);
        auto sampler = std::make_unique<PG>(pg_config,
                                            train,
                                            unlabeled,
                                            test,
                                            &model,
                                            &filter);
        std::string path {"tmp"};
        typename Model::writer w(path,
                                 test,
                                 corpus.get_other_key(),
                                 corpus.symtab.get_map(),
                                 corpus.tagtab.get_map());
        sampler->add_writer_callback(w);
        sampler->run(FLAGS_sec_status_interval);
        auto particles = sampler->get_test_state();
        CHECK(particles.size() > 0)            << "no particles";
        CHECK(particles.size() == test.size()) << "size mismatch";
        LOG(INFO) << "writing state to: " << out_path;
        write_state(particles, test, corpus, model, out_path);
    } else {
        CHECK(false) << "unrecognized mode: " << FLAGS_mode;
    }
    check_output(test, out_path);
}

int main(int argc, char **argv) {
    using namespace nn;

    // Parse command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Log important flags
    LOG(INFO) << "Train path: "                  << FLAGS_train;
    LOG(INFO) << "Test path: "                   << FLAGS_test;
    LOG(INFO) << "Output path for predictions: " << FLAGS_out_path;
    LOG(INFO) << "Gazetteer path: "              << FLAGS_gazetteer;
    LOG(INFO) << "Number of particles: "         << FLAGS_nparticles;
    LOG(INFO) << "Resampling: "                  << FLAGS_resampling;
    LOG(INFO) << "Inference mode: "              << FLAGS_mode;

    CHECK(FLAGS_out_path != "") << "must supply output path for predictions!";

    if(!FLAGS_crossval) {
        CHECK(FLAGS_test     != "") << "must supply path to test data!";
    }

    // Log max num threads being used
    auto nthread = omp_get_max_threads();
    LOG(INFO) << "Maximum number of OMP threads: " << nthread;

    // Timer
    boost::timer::auto_cpu_timer t;

    // Initialize the random number generator(s)
    LOG(INFO) << "Initializing random number generator...";
    rng::init();

    // Run inference
    typedef hidden_sequence_memoizer<>    HSM;
    typedef segmental_sequence_memoizer<> SSM;
    //typedef LatentSegmentalHPYP           SLM;

    typedef CoNLLCorpus<> Corpus;

    if (FLAGS_test_only) {


      if (FLAGS_model == "hsm") {
        LOG(INFO) << "Model: hidden sequence memoizer";
        // Load serialized model
        auto model = load_model<HSM>();
        auto corpus = model->get_corpus();
        // Read test data
        LOG(INFO) << "Reading test data: " << FLAGS_test;
        auto test = corpus.read(FLAGS_test);
        LOG(INFO) << "Read " << test.size() << " test instances.";
        test_model<HSM>(model.get(), test, FLAGS_out_path);
      }
      else if (FLAGS_model == "seg") {
        LOG(INFO) << "Model: segmental sequence memoizer";
        // Load serialized model
        auto model = load_model<SSM>();
        auto corpus = model->get_corpus();
        // Read test data
        LOG(INFO) << "Reading test data: " << FLAGS_test;
        auto test = corpus.read(FLAGS_test);
        LOG(INFO) << "Read " << test.size() << " test instances.";
        test_model<SSM>(model.get(), test, FLAGS_out_path);
      }
      LOG(INFO) << "Done. Exiting...";
      return 0;
    }
    
    if (!FLAGS_crossval) {
        Corpus corpus(FLAGS_bos,
                      FLAGS_eos,
                      FLAGS_space,
                      FLAGS_unk,
                      FLAGS_context_tag);

        // Read training data
        LOG(INFO) << "Reading training data: " << FLAGS_train;
        auto train = corpus.read(FLAGS_train);
        LOG(INFO) << "Read " << train.size() << " training instances.";

        // Read gazetteer
        instances gaz;
        if (FLAGS_gazetteer != "") {
            LOG(INFO) << "Reading gazetteer: " << FLAGS_gazetteer;
            gaz = corpus.read(FLAGS_gazetteer);
        }

        // Optionally: read unlabeled data
        instances unlabeled;
        if (FLAGS_unlabeled != "") {
            LOG(INFO) << "Reading unlabeled data: " << FLAGS_unlabeled;
            unlabeled = corpus.read(FLAGS_unlabeled);
            CHECK(unlabeled.size() > 0);
        }

        // Freeze the symbol table
        corpus.symtab.freeze();
        corpus.frozen = true;

        // Freeze the tag table
        corpus.tagtab.freeze();

        // Print all the tags
        auto tagmap = corpus.tagtab.get_map();
        for(auto e : tagmap) {
            LOG(INFO) << e.first << " <-> " << e.second;
        }

        // Symbol statistics
        size_t nsym = corpus.symtab.size();
        LOG(INFO) << nsym << " symbols in the alphabet";

        if(FLAGS_train_only) {
          if(FLAGS_model == "hsm") {
              train_model<HSM>(corpus,train,gaz,unlabeled);
          } else if (FLAGS_model == "seg") {
              train_model<SSM>(corpus,train,gaz,unlabeled);
          }
          else {
            CHECK(false) << "unrecognized model: " << FLAGS_model;
          }
          LOG(INFO) << "All done; exiting.";
          return 0;
        }

        // Read test data
        LOG(INFO) << "Reading test data: " << FLAGS_test;
        auto test = corpus.read(FLAGS_test);
        LOG(INFO) << "Read " << test.size() << " test instances.";

        if (FLAGS_model == "hsm") {
            LOG(INFO) << "Model: hidden sequence memoizer";
            run_inference<HSM>(train, gaz, unlabeled, test, FLAGS_out_path, corpus);
        }
        else if (FLAGS_model == "seg") {
            LOG(INFO) << "Model: segmental sequence memoizer";
            run_inference<SSM>(train, gaz, unlabeled, test, FLAGS_out_path, corpus);
        }
        // else if (FLAGS_model == "slm") {
        //     LOG(INFO) << "Model: segmental PYP language model";
        //     run_inference<SLM>(train, gaz, unlabeled, test, FLAGS_out_path, corpus);
        // }
        else {
            CHECK(false) << "unrecognized model: " << FLAGS_model;
        }
    } else { // CROSS VALIDATION
        auto N = Corpus::num_instances(FLAGS_train);
        CHECK(N > 0);
        LOG(INFO) << "N = " << N;

        auto N_test_per_fold = N/FLAGS_nfolds;
        LOG(INFO) << "N test per fold = " << N_test_per_fold;

        std::set<size_t> indices;
        for (auto i = 0; i < N; ++i) indices.insert(i);

        std::vector<std::string> output_paths;

        for(auto fold = 0; fold < FLAGS_nfolds; ++fold) {
            LOG(INFO) << "Fold " << fold << " of " << FLAGS_nfolds;

            // Create test corpus
            std::set<size_t> test_indices;
            auto n = 0;
            while(indices.size() > 0 && n < N_test_per_fold) {
                auto i = pop(rng::get(), indices);
                test_indices.insert(i);
                n++;
            }
            CHECK(test_indices.size() > 0);
            CHECK(test_indices.size() < (N/2)) << test_indices.size();

            // Create training corpus
            std::set<size_t> train_indices;
            for (size_t i=0; i < N; ++i) {
                auto in_test = (test_indices.count(i) > 0);
                if (!in_test) { train_indices.insert(i); }
            }

            CHECK(train_indices.size() > 0);
            CHECK(test_indices.size()  > 0);
            CHECK(train_indices.size() + test_indices.size() == N);

            Corpus corpus(FLAGS_bos,
                          FLAGS_eos,
                          FLAGS_space,
                          FLAGS_unk,
                          FLAGS_context_tag);

            // Read gazetteer
            instances gaz;
            if (FLAGS_gazetteer != "") {
                LOG(INFO) << "Reading gazetteer: " << FLAGS_gazetteer;
                gaz = corpus.read(FLAGS_gazetteer);
            }

            // Optionally: read unlabeled data
            instances unlabeled;
            if(FLAGS_unlabeled != "") {
                LOG(INFO) << "Reading unlabeled data: " << FLAGS_unlabeled;
                unlabeled = corpus.read(FLAGS_unlabeled);
                CHECK(unlabeled.size() > 0);
            }

            auto ret = corpus.read(FLAGS_train, train_indices, test_indices);

            auto train = std::get<0>(ret);
            auto test = std::get<1>(ret);

            LOG(INFO) << "train.size() = " << train.size();
            LOG(INFO) << "test.size()  = " << test.size();

            CHECK(train.size() > 0);
            CHECK(test.size() > 0);
            CHECK(train.size() + test.size() == N) << "N = " << N
                                                   << " vs "
                                                   << train.size() + test.size();

            // Symbol statistics
            size_t nsym = corpus.symtab.size();
            LOG(INFO) << nsym << " symbols in the alphabet";

            std::string output_path {FLAGS_out_path+"."+std::to_string(fold)};
            LOG(INFO) << "Writing predictions: " << output_path;

            output_paths.push_back(output_path);

            if (FLAGS_model == "hsm") {
                LOG(INFO) << "Model: hidden sequence memoizer";
                run_inference<HSM>(train, gaz, unlabeled, test, output_path, corpus);
            }
            else if (FLAGS_model == "seg") {
                LOG(INFO) << "Model: segmental sequence memoizer";
                run_inference<SSM>(train, gaz, unlabeled, test, output_path, corpus);
            }
            // else if (FLAGS_model == "slm") {
            //     LOG(INFO) << "Model: segmental PYP language model";
            //     run_inference<SLM>(train, gaz, unlabeled, test, output_path, corpus);
            // }
            else {
                CHECK(false) << "unrecognized model: " << FLAGS_model;
            }
        } // end of fold loop

        LOG(INFO) << "Cross validation complete. Output paths:";
        for(auto path : output_paths) {
            LOG(INFO) << "\t" << path;
        }
    }
}
