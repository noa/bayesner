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

#ifndef __NN_EVAL_HPP__
#define __NN_EVAL_HPP__

#include <vector>
#include <map>
#include <set>

#include <nn/log.hpp>
#include <nn/data.hpp>

namespace nn {

    template<typename Particle>
    struct F1Result {
        size_t correctChunk {0};
        size_t foundGuessed {0};
        size_t foundCorrect {0};
        size_t correctTags  {0};
        size_t tokenCounter {0};

        std::map<size_t,size_t> correctChunkMap;
        std::map<size_t,size_t> foundGuessedMap;
        std::map<size_t,size_t> foundCorrectMap;

        const size_t context_tag;

        const std::set<size_t>         begin_tags;
        const std::set<size_t>         extend_tags;
        const std::map<size_t, size_t> tag_type;

        F1Result(size_t _context_tag,
                 std::set<size_t> _begin_tags,
                 std::set<size_t> _extend_tags,
                 std::map<size_t, size_t> _tag_type)
            : context_tag(_context_tag),
              begin_tags(_begin_tags),
              extend_tags(_extend_tags),
              tag_type(_tag_type)
            {};

        bool begin(size_t tag)   { return begin_tags.count(tag)  > 0; }
        bool extend(size_t tag)  { return extend_tags.count(tag) > 0; }
        bool context(size_t tag) { return tag == context_tag;         }
        size_t type(size_t tag)  { return tag_type.at(tag);           }

        bool endOfChunk(size_t prevTag,
                        size_t tag,
                        size_t prevType,
                        size_t type) {
            bool chunkEnd = false;

            if (begin(prevTag)  && begin(tag)  )   chunkEnd = true;
            if (begin(prevTag)  && context(tag))   chunkEnd = true;
            if (extend(prevTag) && begin(tag)  )   chunkEnd = true;
            if (extend(prevTag) && context(tag))   chunkEnd = true;

            if (!context(prevTag) && prevType != type) {
                chunkEnd = true;
            }

            return chunkEnd;
        }

        bool startOfChunk(size_t prevTag,
                          size_t tag,
                          size_t prevType,
                          size_t type) {
            bool chunkStart = false;

            if (begin(prevTag)   && begin(tag) )   chunkStart = true;
            if (extend(prevTag)  && begin(tag) )   chunkStart = true;
            if (context(prevTag) && begin(tag) )   chunkStart = true;
            if (context(prevTag) && extend(tag))   chunkStart = true;

            if (!context(tag) && prevType != type) chunkStart = true;

            return chunkStart;
        }

        bool observe(Particle guess, Particle gold) {
            auto tags = guess.tags;
            CHECK(tags.size() == gold.tags.size()) << "size mismatch";
            bool inCorrect {false};
            size_t lastCorrect = context_tag;
            size_t lastCorrectType {0};
            size_t lastGuessed = context_tag;
            size_t guessed;
            size_t correct;
            size_t lastGuessedType {0};
            size_t correctType;
            size_t guessedType;
            bool firstTag { true };
            for(auto i=0; i<tags.size(); ++i) {
                guessed = tags.at(i);
                guessedType = type(tags.at(i));
                correct = gold.tags.at(i);
                correctType = type(correct);

                if (inCorrect) {
                    if(endOfChunk(lastCorrect,correct,lastCorrectType,correctType) &&
                       endOfChunk(lastGuessed,guessed,lastGuessedType,guessedType) &&
                       lastGuessedType == lastCorrectType) {
                        inCorrect = false;
                        correctChunk++;
                        if(correctChunkMap.count(lastCorrectType)) {
                            correctChunkMap[lastCorrectType]++;
                        } else { correctChunkMap[lastCorrectType] = 1; }
                    } else if (
                        endOfChunk(lastCorrect,correct,lastCorrectType,correctType) !=
                        endOfChunk(lastGuessed,guessed,lastGuessedType,guessedType) ||
                        guessedType != correctType ) {
                        inCorrect = false;
                    }
                }

                if (startOfChunk(lastCorrect,correct,lastCorrectType,correctType) &&
                    startOfChunk(lastGuessed,guessed,lastGuessedType,guessedType) &&
                    guessedType == correctType) {
                    inCorrect = true;
                }

                if (startOfChunk(lastCorrect,correct,lastCorrectType,correctType)) {
                    foundCorrect ++;
                    if(foundCorrectMap.count(correctType)) {
                        foundCorrectMap[correctType]++;
                    }
                    else { foundCorrectMap[correctType] = 1; }
                }

                if (startOfChunk(lastGuessed,guessed,lastGuessedType,guessedType)) {
                    foundGuessed ++;
                    if(foundGuessedMap.count(guessedType)) {
                        foundGuessedMap[guessedType]++;
                    }
                    else { foundGuessedMap[guessedType] = 1; }
                }

                if ( true ) {
                    if (correct == guessed && guessedType == correctType) {
                        correctTags++;
                    }
                    tokenCounter++;
                }

                firstTag = false;

                lastGuessed = guessed;
                lastCorrect = correct;
                lastGuessedType = guessedType;
                lastCorrectType = correctType;
            }
            return true;
        }

        double precision() const {
            return 100.0*correctChunk/foundGuessed;
        }

        double recall() const {
            return 100.0*correctChunk/foundCorrect;
        }

        double f1() const {
            double p = precision();
            double r = recall();
            return (2*p*r)/(p+r);
        }

        template<typename Table>
        void log(const Table& tagtab) {
            double p = precision();
            double r = recall();
            double f = f1();
            double a = 100.0*correctTags/tokenCounter;
            LOG(INFO) << "Overall: precision: " << p
                      << "; recall: " << r
                      << "; F1: " << f
                      << "; accuracy: " << a;
            for (auto kv : foundCorrectMap) {
                auto tag = kv.first;
                auto tag_str = tagtab.val(tag);
                auto tagFoundCorrect = kv.second;

                if(!foundGuessedMap.count(tag)) {
                    foundGuessedMap[tag] = 0;
                }

                auto tagFoundGuessed = foundGuessedMap[tag];
                auto tagCorrectChunk = correctChunkMap[tag];

                double p = 100.0*tagCorrectChunk/tagFoundGuessed;
                double r = 100.0*tagCorrectChunk/tagFoundCorrect;
                double f = (2*p*r) / (p+r);

                LOG(INFO) << tag_str
                          << ": precision: " << p
                          << "; recall: " << r
                          << "; FB1: " << f;
            }
        }
    };

    // template<typename Particle>
    // auto compute_f1(typename std::vector<Particle>::const_iterator pred_start,
    //                 typename std::vector<Particle>::const_iterator pred_stop,
    //                 typename std::vector<Particle>::const_iterator gold_start,
    //                 typename std::vector<Particle>::const_iterator gold_stop)
    //     -> F1Result<Particle>
    // {
    //     F1Result<Particle> ret;
    //     return ret;
    // }

    template<typename Particle>
    class F1Evaluator {
        typename std::vector<Particle>::const_iterator gold_start;
        typename std::vector<Particle>::const_iterator gold_stop;
        const size_t context_tag;
        const std::set<size_t> begin_tags;
        const std::set<size_t> extend_tags;
        const std::map<size_t, size_t> tag_type;
        const uint_str_table& tagmap;

    public:
        F1Evaluator(typename std::vector<Particle>::const_iterator _gold_start,
                    typename std::vector<Particle>::const_iterator _gold_stop,
                    size_t _context_tag,
                    std::set<size_t> _begin_tags,
                    std::set<size_t> _extend_tags,
                    std::map<size_t, size_t> _tag_type,
                    const uint_str_table& _tagmap
                    )
            : gold_start(_gold_start),
              gold_stop(_gold_stop),
              context_tag(_context_tag),
              begin_tags(_begin_tags),
              extend_tags(_extend_tags),
              tag_type(_tag_type),
              tagmap(_tagmap)
        {}

        void operator()(typename std::vector<Particle>::const_iterator pred_start,
                        typename std::vector<Particle>::const_iterator pred_stop) {
            F1Result<Particle> result(context_tag,
                                      begin_tags,
                                      extend_tags,
                                      tag_type);
            auto pred = pred_start;
            auto gold = gold_start;
            while (pred != pred_stop) {
                result.observe(*pred, *gold);
                pred ++; gold ++;
            }
            result.log(tagmap);
        }
    };
};

#endif
