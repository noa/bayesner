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

#ifndef __NN_PGIBBS_HPP__
#define __NN_PGIBBS_HPP__

#include <vector>

#include <nn/data.hpp>
#include <nn/csmc.hpp>
#include <nn/timing.hpp>

namespace nn {
    template<typename P,     // particle
             typename F,     // filter
             typename M,     // model
             typename O      // observations (iterable type)
             >
    struct particle_gibbs {
        struct settings { size_t num_iter {100}; };

        settings config;

        // Anneal
        bool annealed_gibbs {false};
        double anneal_exp {2.0};

        // Observations
        const O& train;
        const O& unlabeled;
        const O& test;

        M* model;
        F* filter;

        // State
        size_t epoch_iter {0};

        // Sampler state / sufficient statistics
        std::vector<P> state_train;
        std::vector<P> state_unlabeled;
        std::vector<P> state_test;

        bool initialized {false};

        // called between sampler iterations
        std::function<std::vector<P>(const O&)> initializer;
        std::vector<std::function<
            void(typename std::vector<P>::const_iterator,
                 typename std::vector<P>::const_iterator)
        >> evaluators;
        std::vector<std::function<
            void(typename std::vector<P>::const_iterator,
                 typename std::vector<P>::const_iterator)
        >> inspectors;
        std::vector<std::function<
            void(size_t,
                 typename std::vector<P>::const_iterator,
                 typename std::vector<P>::const_iterator)
        >> writers;

        particle_gibbs(settings _config,
                       const O& _train,
                       const O& _unlabeled,
                       const O& _test,
                       M* _model,
                       F* _filter)
            : config(_config),
              train(_train),
              unlabeled(_unlabeled),
              test(_test),
              model(_model),
              filter(_filter) {}

        double anneal(size_t iter) {
            auto T = static_cast<double>(config.num_iter);
            auto slope = 1.0/std::pow(T, anneal_exp);
            auto t = static_cast<double>(iter);
            return 1.0 - slope*std::pow(iter, anneal_exp);
        }

        auto get_train_state() -> decltype(state_train) {
            return state_train;
        }

        auto get_unlabeled_state() -> decltype(state_unlabeled) {
            return state_unlabeled;
        }

        auto get_test_state() -> decltype(state_test) {
            return state_test;
        }

        void set_initial_state(const std::vector<P> _state) {
            this->state = _state;
            CHECK(this->state.size() == this->data.size()) << "size mismatch!";
            this->initialized = true;
        }

        void init() {
            CHECK(!initialized) << "already initialized";
            LOG(INFO) << "Initializing sampler state...";
            nn::progress_bar train_prog(train.size(), 5);
            nn::tic();
            size_t i;
            LOG(INFO) << "Initiatializing train state... ";
            for(i = 0; i < train.size(); ++i) {
                auto p = filter->sample( train[i].tags,
                                         train[i].lens,
                                         train[i].words,
                                         train[i].obs );
                state_train.push_back(p);
                model->observe( p, train[i].words );
                train_prog++;
            }
            LOG(INFO) << "...done in: " << nn::prettyprint(nn::toc());

            if(unlabeled.size() > 0) {
                nn::progress_bar unlabeled_prog(unlabeled.size(), 5);
                nn::tic();
                LOG(INFO) << "Initiatializing unlabeled state... ";
                for (i = 0; i < unlabeled.size(); ++i) {
                    auto p = filter->sample( unlabeled[i].tags,
                                             unlabeled[i].lens,
                                             unlabeled[i].words,
                                             unlabeled[i].obs );
                    state_unlabeled.push_back(p);
                    model->observe( p, unlabeled[i].words );
                    unlabeled_prog++;
                }
                LOG(INFO) << "...done in: " << nn::prettyprint(nn::toc());
            }

            nn::progress_bar test_prog(test.size(), 5);
            nn::tic();

            LOG(INFO) << "initiatializing test state... ";
            for(i = 0; i < test.size(); ++i) {
                auto p = filter->sample( test[i].words );
                state_test.push_back(p);
                model->observe( p, test[i].words );
                test_prog++;
            }
            LOG(INFO) << "...done in: " << nn::prettyprint(nn::toc());
            write_test_state();
        }

        void pre_run_callbacks() const {
            LOG(INFO) << "Evaluation:";
            for(const auto& e : evaluators) {
                e(state_test.begin(), state_test.end());
            }
        }

        void run_eval() const {
            LOG(INFO) << "Evaluation:";
            for(const auto& e : evaluators) {
                e(state_test.begin(), state_test.end());
            }
        }

        void write_test_state() const {
            for(const auto& w : writers) {
                w(epoch_iter, state_test.begin(), state_test.end());
            }
        }

        void pre_sweep_callbacks() const { run_eval(); }

        void run(size_t status_interval) {
            // initialize sampler state
            init();

            LOG(INFO) << "Resampling parameters...";
            model->resample_hyperparameters();

            // run particle Gibbs
            size_t i, j;
            double mean_ESS;
            size_t n_instance_sampled = 0;
            size_t n_total_instance = train.size() + test.size() + unlabeled.size();
            nn::progress_bar prog(n_total_instance * config.num_iter,
                                  status_interval);
            nn::tic();
            for (epoch_iter = 1; epoch_iter < config.num_iter; ++epoch_iter) {
                pre_sweep_callbacks();
                LOG(INFO) << "[epoch " << epoch_iter << " of " << config.num_iter
                          << "] running Gibbs sweep...";
                mean_ESS = 0.0;
                LOG(INFO) << "Resampling train data...";
                for (j=0; j<train.size(); ++j) {
                    const auto& instance = train.at(j);
                    const auto& particle = state_train.at(j);
                    model->remove(particle, instance.words);
                    state_train[j] = filter->conditional_sample(particle,
                                                                instance.tags,
                                                                instance.words,
                                                                instance.obs);
                    if (train[j].obs != Annotation::FULL) {
                        mean_ESS += filter->get_ess();
                        n_instance_sampled ++;
                    }
                    model->observe(state_train[j], instance.words);
                    prog++;
                }

                if(unlabeled.size() > 0) {
                    LOG(INFO) << "Resampling unlabeled data...";
                    for (j=0; j<unlabeled.size(); ++j) {
                        model->remove(state_unlabeled[j], unlabeled[j].words);
                        state_unlabeled[j] = filter->conditional_sample(state_unlabeled[j],
                                                                        unlabeled[j].words);
                        mean_ESS += filter->get_ess();
                        n_instance_sampled ++;
                        const auto& particle = state_unlabeled.at(j);
                        model->observe(particle, unlabeled[j].words);
                        prog++;
                    }
                }

                LOG(INFO) << "Resampling test data...";
                for (j=0; j<test.size(); ++j) {
                    model->remove(state_test[j], test[j].words);
                    state_test[j] = filter->conditional_sample(state_test[j],
                                                               test[j].words);
                    mean_ESS += filter->get_ess();
                    n_instance_sampled ++;
                    const auto& particle = state_test.at(j);
                    model->observe(particle, test[j].words);
                    prog++;
                }
                mean_ESS /= n_instance_sampled;
                LOG(INFO) << "[mean ESS = " << mean_ESS << "]";
                write_test_state();
            }
            LOG(INFO) << "Final evaluation:";
            run_eval();
            LOG(INFO) << "...done in: " << nn::prettyprint(nn::toc());
        }

        template<typename Evaluator>
        void add_evaluation_callback(Evaluator&& evaluator) {
            evaluators.push_back( std::forward<Evaluator>(evaluator) );
        }

        template<typename Inspector>
        void add_inspector_callback(Inspector&& inspector) {
            inspectors.push_back( std::forward<Inspector>(inspector) );
        }

        template<typename Writer>
        void add_writer_callback(Writer&& writer) {
            writers.push_back( std::forward<Writer>(writer) );
        }

        template<typename Initializer>
        void set_initializer(Initializer&& init) {
            initializer = std::forward<Initializer>( init );
        }
    };
}

#endif
