/*
 * Copyright 2008-2016 Jan Gasthaus
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

#ifndef __NN_RESTAURANTS_HPP__
#define __NN_RESTAURANTS_HPP__

#include <map>
#include <boost/multi_array.hpp>

#include <nn/rng.hpp>
#include <nn/stat.hpp>
#include <nn/mu.hpp>
#include <nn/utils.hpp>
#include <nn/restaurant_interface.hpp>

namespace nn {
    inline double computeHPYPPredictive(size_t cw,
                                        size_t tw,
                                        size_t c,
                                        size_t t,
                                        double parentProbability,
                                        double discount,
                                        double concentration) {
        if (c==0) {
            return parentProbability;
        } else {
            return (cw - discount*tw + (concentration + discount*t)
                    *parentProbability) / (c + concentration);
        }
    }

    inline double computeLogHPYPPredictive(size_t cw,
                                           size_t tw,
                                           size_t c,
                                           size_t t,
                                           double logParentProbability,
                                           double discount,
                                           double concentration) {
        if (c==0) {
            return logParentProbability;
        } else {
            auto log_denom = log(c + concentration);
            return log_add( log(cw-discount*tw) - log_denom,
                            logParentProbability+log(concentration + discount*t)
                            - log_denom );
        }
    }

    inline double computeHPYPLogCachedProb(size_t cw,
                                           size_t tw,
                                           size_t c,
                                           double discount,
                                           double concentration) {
        if (cw < 1) return NEG_INF;
        double numer = static_cast<double>(cw)-discount*static_cast<double>(tw);
        double denom = static_cast<double>(c)+concentration;
        return log(numer) - log(denom);
    }

    inline double computeHPYPLogNewProb(size_t c, size_t t,
                                        double logParentProbability,
                                        double discount, double concentration) {
        return log(concentration + discount*static_cast<double>(t))
            + logParentProbability - log(static_cast<double>(c)+concentration);
    }

    inline bool createTable(size_t cw, size_t tw, size_t t,
                            double parentProbability,
                            double discount, double concentration) {
        double incTProb = (concentration + discount*t) * parentProbability;
        incTProb = incTProb/(incTProb + cw - tw * discount);
        return nn::coin(incTProb, nn::rng::get());
    }

    /**
     * An HPYP restaurant that stores the full seating arrangement
     * using a straightforward STL data structure (see nested struct
     * Payload).
     *
     * For each type, the restaurant stores the number of customers
     * sitting around each individual table. The total number of
     * customers per type as well as the overall total number of
     * customers and tables are also stored for fast lookup.
     *
     * This restaurant allows for fast insertion and removal of
     * customers, but is not very memory efficient.
     */
    template <typename Dish>
    class SimpleFullRestaurant : public IAddRemoveRestaurant<Dish> {
    public:
        typedef typename std::vector<Dish>    TypeVector;
        typedef typename TypeVector::iterator TypeVectorIterator;

        SimpleFullRestaurant() : payloadFactory() {}
        ~SimpleFullRestaurant() {}

        l_type getC(void* payloadPtr, Dish type) const;
        l_type getC(void* payloadPtr) const;
        l_type getT(void* payloadPtr, Dish type) const;
        l_type getT(void* payloadPtr) const;

        double computeProbability(void* payloadPtr,
                                  Dish type,
                                  double parentProbability,
                                  double discount,
                                  double concentration) const;

        double computeLogProbability(void* payloadPtr,
                                     Dish type,
                                     double parentLogProbability,
                                     double discount,
                                     double concentration) const;

        double computeLogCacheProb(void* payloadPtr,
                                   Dish type,
                                   double discount,
                                   double concentration) const;

        double computeLogNewProb(void* payloadPtr,
                                 //Dish type,
                                 double parentLogProbability,
                                 double discount,
                                 double concentration) const;

        TypeVector getTypeVector(void* payloadPtr) const;

        const IPayloadFactory& getFactory() const;

        void updateAfterSplit(void* longerPayloadPtr,
                              void* shorterPayloadPtr,
                              double discountBeforeSplit,
                              double discountAfterSplit,
                              bool parentOnly = false) const;

        bool addCustomer(void*  payloadPtr,
                         Dish type,
                         double parentProbability,
                         double discount,
                         double concentration,
                         void* additionalData = nullptr
            ) const;

        bool addCustomer(bool sharing_table,
                         void*  payloadPtr,
                         Dish type,
                         double parentProbability,
                         double discount,
                         double concentration,
                         void* additionalData = nullptr
            ) const;

        bool logAddCustomer(void* payloadPtr,
                            Dish type,
                            double logParentProbability,
                            double discount,
                            double concentration,
                            void* additionalData = nullptr
            ) const;

        bool removeCustomer(void* payloadPtr,
                            Dish type,
                            double discount,
                            void* additionalData
            ) const;

        void* createAdditionalData(void* payloadPtr,
                                   double discount,
                                   double concentration
            ) const;

        void freeAdditionalData(void* additionalData) const;
        bool checkConsistency(void* payloadPtr) const;

        class Payload {
        public:
            Payload() : tableMap(), sumCustomers(0), sumTables(0) {}

            Payload( const Payload& other ) :
                tableMap( other.tableMap ),
                sumCustomers( other.sumCustomers ),
                sumTables( other.sumTables )
                {}

            std::map<Dish, std::pair<l_type, std::vector<l_type>>> tableMap;

            l_type sumCustomers;
            l_type sumTables;

            //void serialize(InArchive & ar, const unsigned int version);
            //void serialize(OutArchive & ar, const unsigned int version);
        };

        class PayloadFactory : public IPayloadFactory {
        public:
            void* make() const {
                return new Payload();
            };

            void recycle(void* payloadPtr) const {
                delete (Payload*)payloadPtr;
            }

            void* copy(void* payloadPtr) const {
                return new Payload(*(Payload*)payloadPtr);
            }

            //void save(void* payloadPtr, OutArchive& oa) const;
            //void* load(InArchive& ia) const;
        };

        const PayloadFactory payloadFactory;
    };

    template<typename Dish>
    class HistogramRestaurant : public IAddRemoveRestaurant<Dish> {
    public:
        typedef typename std::vector<Dish> TypeVector;
        typedef typename TypeVector::iterator TypeVectorIterator;

        HistogramRestaurant() : payloadFactory() {}
        ~HistogramRestaurant() {}

        l_type getC(void* payloadPtr, Dish type) const;
        l_type getC(void* payloadPtr)            const;
        l_type getT(void* payloadPtr, Dish type) const;
        l_type getT(void* payloadPtr)            const;

        double computeProbability(void*  payloadPtr,
                                  Dish type,
                                  double parentProbability,
                                  double discount,
                                  double concentration) const;

        double computeLogProbability(void*  payloadPtr,
                                     Dish type,
                                     double parentLogProbability,
                                     double discount,
                                     double concentration) const;

        typename IHPYPBaseRestaurant<Dish>::TypeVector getTypeVector(void* payloadPtr) const;

        const IPayloadFactory& getFactory() const;

        void updateAfterSplit(void* longerPayloadPtr,
                              void* shorterPayloadPtr,
                              double discountBeforeSplit,
                              double discountAfterSplit,
                              bool parentOnly = false) const;

        bool addCustomer(void*  payloadPtr,
                         Dish type,
                         double parentProbability,
                         double discount,
                         double concentration,
                         void* additionalData = NULL) const;

        bool logAddCustomer(void* payloadPtr,
                            Dish type,
                            double logParentProbability,
                            double discount,
                            double concentration,
                            void* additionalData = nullptr
            ) const;

        bool removeCustomer(void* payloadPtr,
                            Dish type,
                            double discount,
                            void* additionalData) const;

        void* createAdditionalData(void* payloadPtr,
                                   double discount,
                                   double concentration) const;

        void freeAdditionalData(void* additionalData) const;

        bool checkConsistency(void* payloadPtr) const;

        class Payload {
        public:
            typedef std::map<l_type, l_type> Histogram;

            struct Arrangement {
                Arrangement() : cw(0), tw(0), histogram() {}
                l_type cw;
                l_type tw;
                Histogram histogram;
            };

            typedef std::map<Dish, Arrangement> TableMap;

            Payload() : tableMap(), sumCustomers(0), sumTables(0) {}

            TableMap tableMap;
            l_type sumCustomers;
            l_type sumTables;
        };

        class PayloadFactory : public IPayloadFactory {

            void* make() const {
                return new Payload();
            };

            void recycle(void* payloadPtr) const {
                delete (Payload*)payloadPtr;
            }
        };

        const PayloadFactory payloadFactory;
    };

    template <typename Dish>
    l_type SimpleFullRestaurant<Dish>::getC(void* payloadPtr, Dish type) const {
        Payload& payload = *((Payload*)payloadPtr);
        auto it = payload.tableMap.find(type);
        if (it != payload.tableMap.end()) {
            return it->second.first; //cw
        } else {
            return 0;
        }
    }

    template <typename Dish>
    l_type SimpleFullRestaurant<Dish>::getC(void* payloadPtr) const {
        return ((Payload*)payloadPtr)->sumCustomers;
    }

    template <typename Dish>
    l_type SimpleFullRestaurant<Dish>::getT(void* payloadPtr, Dish type) const {
        Payload& payload = *((Payload*)payloadPtr);
        auto it = payload.tableMap.find(type);
        if (it != payload.tableMap.end()) {
            return it->second.second.size();
        } else {
            return 0;
        }
    }

    template <typename Dish>
    l_type SimpleFullRestaurant<Dish>::getT(void* payloadPtr) const {
        return ((Payload*)payloadPtr)->sumTables;
    }

    template <typename Dish>
    double SimpleFullRestaurant<Dish>::computeProbability(void*  payloadPtr,
                                                          Dish type,
                                                          double parentProbability,
                                                          double discount,
                                                          double concentration) const {
        Payload& payload = *((Payload*)payloadPtr);
        size_t cw = 0;
        size_t tw = 0;
        auto it = payload.tableMap.find(type);
        if (it != payload.tableMap.end()) {
            cw = (*it).second.first;
            tw = (*it).second.second.size();
        }
        return computeHPYPPredictive(cw, // cw
                                     tw, // tw
                                     payload.sumCustomers, // c
                                     payload.sumTables, // t
                                     parentProbability,
                                     discount,
                                     concentration);
    }

    template <typename Dish>
    double SimpleFullRestaurant<Dish>::computeLogProbability(void* payloadPtr,
                                                             Dish type,
                                                             double parentLogProbability,
                                                             double discount,
                                                             double concentration) const {
        Payload& payload = *((Payload*)payloadPtr);
        size_t cw = 0;
        size_t tw = 0;
        auto it = payload.tableMap.find(type);
        if (it != payload.tableMap.end()) {
            cw = (*it).second.first;
            tw = (*it).second.second.size();
        }
        return computeLogHPYPPredictive(cw, // cw
                                        tw, // tw
                                        payload.sumCustomers, // c
                                        payload.sumTables,    // t
                                        parentLogProbability,
                                        discount,
                                        concentration);
    }

    template <typename Dish>
    double SimpleFullRestaurant<Dish>::computeLogCacheProb(void* payloadPtr,
                                                           Dish type,
                                                           double discount,
                                                           double concentration) const {
        Payload& payload = *((Payload*)payloadPtr);
        size_t cw = 0;
        size_t tw = 0;
        auto it = payload.tableMap.find(type);
        if (it != payload.tableMap.end()) {
            cw = (*it).second.first;
            tw = (*it).second.second.size();
        }
        return computeHPYPLogCachedProb(cw, // cw
                                        tw, // tw
                                        payload.sumCustomers, // c
                                        discount,
                                        concentration);
    }

    template <typename Dish>
    double SimpleFullRestaurant<Dish>::computeLogNewProb(void* payloadPtr,
                                                         // Dish type,
                                                         double parentLogProbability,
                                                         double discount,
                                                         double concentration) const {
        Payload& payload = *((Payload*)payloadPtr);
        // int cw = 0;
        // int tw = 0;
        // auto it = payload.tableMap.find(type);
        // if (it != payload.tableMap.end()) {
        //   cw = (*it).second.first;
        //   tw = (*it).second.second.size();
        // }
        return computeHPYPLogNewProb(   payload.sumCustomers, // c
                                        payload.sumTables,    // t
                                        parentLogProbability,
                                        discount,
                                        concentration);
    }

    template <typename Dish>
    std::vector<Dish> SimpleFullRestaurant<Dish>::getTypeVector(
        void* payloadPtr) const {
        Payload& payload = *((Payload*)payloadPtr);
        std::vector<Dish> typeVector;
        typeVector.reserve(payload.tableMap.size());
        for (auto it = payload.tableMap.begin();
             it != payload.tableMap.end();
             ++it) {
            typeVector.push_back(it->first);
        }
        return typeVector;
    }

    template <typename Dish>
    const IPayloadFactory& SimpleFullRestaurant<Dish>::getFactory() const {
        return this->payloadFactory;
    }

    template <typename Dish>
    bool SimpleFullRestaurant<Dish>::addCustomer(
        bool sharing_table,
        void*  payloadPtr,
        Dish type,
        double parentProbability,
        double discount,
        double concentration,
        void*  additionalData
        ) const {

        CHECK(false) << "untested";

        CHECK(additionalData == nullptr);
        Payload& payload = *((Payload*)payloadPtr);
        auto& arrangement = payload.tableMap[type];
        std::vector<l_type>& tables = arrangement.second;

        ++payload.sumCustomers; // c
        ++arrangement.first; // cw

        //assert(!(arrangement.first == 1 && sharing_table));

        if (arrangement.first == 1) {
            // first customer sits at first table
            tables.push_back(1);
            ++payload.sumTables;
            return true;
        }

        // probs for old tables: \propto cwk - d
        d_vec tableProbs(tables.size() + 1, 0);
        for(size_t i = 0; i < tables.size(); ++i) {
            tableProbs[i] = tables[i] - discount;
        }

        // prob for new table: \propto (alpha + d*t)*P0
        // this can be 0 for the first customer if concentration=0, but that is ok
        tableProbs[tables.size()] = (concentration + discount*payload.sumTables)*parentProbability;

        // choose table for customer to sit at
        auto table = nn::sample_unnormalized_pdf(tableProbs, nn::rng::get());
        assert(table <= tables.size());

        if(table == tables.size()) {
            // sit at new table
            tables.push_back(1);
            ++payload.sumTables;
            return true;
        } else {
            // existing table
            ++tables[table];
            return false;
        }
    }

    template <typename Dish>
    bool SimpleFullRestaurant<Dish>::addCustomer(void*  payloadPtr,
                                                 Dish type,
                                                 double parentProbability,
                                                 double discount,
                                                 double concentration,
                                                 void*  additionalData
        ) const {

        //CHECK(additionalData == nullptr);
        Payload& payload = *((Payload*)payloadPtr);
        auto& arrangement = payload.tableMap[type];
        std::vector<l_type>& tables = arrangement.second;

        ++payload.sumCustomers; // c
        ++arrangement.first;    // cw

        if (arrangement.first == 1) { // first customer sits at first table
            tables.push_back(1);
            ++payload.sumTables;
            return true;
        }

        // probs for old tables: \propto cwk - d
        //d_vec tableProbs(tables.size() + 1, 0);
        d_vec tableProbs;
        for(size_t i = 0; i < tables.size(); ++i) {
            auto p = std::max(0.0,
                              static_cast<double>(tables.at(i)) - discount);
            tableProbs.push_back(p);
        }

        // prob for new table: \propto (alpha + d*t)*P0
        // this can be 0 for the first customer if concentration=0, but that is ok
        auto p = (concentration + discount * static_cast<double>(payload.sumTables)) * parentProbability;
        tableProbs.push_back(p);

        // choose table for customer to sit at
        // DLOG(INFO) << "here.";
        // auto rng = nn::rng::get();
        // DLOG(INFO) << "got rng.";

        // XXX DEBUG
        //for (auto i=0; i<tableProbs.size(); ++i) {
        //LOG(INFO) << "table " << i << " propto " << tableProbs.at(i);
        //}

        //LOG(INFO) << "sampling table from " << tableProbs.size() << " items";

        auto table = nn::sample_unnormalized_pdf(tableProbs, nn::rng::get());

        //LOG(INFO) << "sampled table = " << table;

        //CHECK(table <= tables.size());
        //CHECK(table >= 0);

        if(table == tables.size()) {
            // sit at new table
            tables.push_back(1);
            ++payload.sumTables;
            return true;
        } else {
            // existing table
            tables[table] += 1;
            return false;
        }

        //LOG(INFO) << "out of add customer";
    }

    template <typename Dish>
    bool SimpleFullRestaurant<Dish>::logAddCustomer(void* payloadPtr,
                                                    Dish type,
                                                    double logParentProbability,
                                                    double discount,
                                                    double concentration,
                                                    void* additionalData
        ) const {

        CHECK(additionalData == nullptr);
        Payload& payload = *((Payload*)payloadPtr);
        auto& arrangement = payload.tableMap[type];
        std::vector<l_type>& tables = arrangement.second;

        ++payload.sumCustomers; // c
        ++arrangement.first; // cw

        if (arrangement.first == 1) {
            // first customer sits at first table
            tables.push_back(1);
            ++payload.sumTables;
            return true;
        }

        // probs for old tables: \propto cwk - d
        // d_vec tableProbs(tables.size() + 1, 0);
        d_vec tableLogProbs;
        //tableProbs.resize(tables.size() + 1, 0);
        for(auto i = 0; i < tables.size(); ++i) {
            auto p = std::max(0.0, static_cast<double>(tables[i]) - discount);
            tableLogProbs.push_back(log(p));
        }

        // prob for new table: \propto (alpha + d*t)*P0
        // this can be 0 for the first customer if concentration=0, but that is ok
        //tableProbs[tables.size()] = log(concentration + discount*payload.sumTables) + logParentProbability;
        tableLogProbs.push_back( log(concentration + discount*static_cast<double>(payload.sumTables)) + logParentProbability );

        // choose table for customer to sit at
        auto table = nn::sample_unnormalized_lnpdf(tableLogProbs, nn::rng::get());
        //CHECK(table <= tables.size());

        if(table == tables.size()) {
            // sit at new table
            tables.push_back(1);
            ++payload.sumTables;
            return true;
        } else {
            // existing table
            ++tables[table];
            return false;
        }

    }

    template <typename Dish>
    bool SimpleFullRestaurant<Dish>::removeCustomer(void* payloadPtr, Dish type,
                                                    double discount,
                                                    void* additionalData) const {
        CHECK(additionalData == nullptr);

        Payload& payload = *((Payload*)payloadPtr);
        assert(payload.tableMap.count(type) == 1);
        auto& arrangement = payload.tableMap[type];
        std::vector<l_type>& tables = arrangement.second;

        // CHECK(payload.sumCustomers > 0);
        // CHECK(payload.sumTables    > 0);
        // CHECK(arrangement.first    > 0);
        // CHECK(tables.size()        > 0);

        --payload.sumCustomers; // c
        --arrangement.first;    // cw

        d_vec tableProbs(tables.begin(), tables.end()); // cast to double

        // chose a table to delete the customer from; prob proportional to table size
        auto table = nn::sample_unnormalized_pdf(tableProbs, nn::rng::get());
        //CHECK(table < tables.size());

        // remove customer from table
        --tables[table];
        if (tables[table] == 0) { // if table became empty
            tables.erase(tables.begin() + table); // drop from table list
            --payload.sumTables;
            return true;
        } else {
            return false;
        }
    }

    template <typename Dish>
    void* SimpleFullRestaurant<Dish>::createAdditionalData(void* payloadPtr,
                                                           double discount,
                                                           double concentration) const {
        return nullptr; // we don't need additional data
    }

    template <typename Dish>
    void SimpleFullRestaurant<Dish>::freeAdditionalData(void* additionalData) const {
        // nothing to be done
    }

    template <typename Dish>
    bool SimpleFullRestaurant<Dish>::checkConsistency(void* payloadPtr) const {
        Payload& payload = *((Payload*)payloadPtr);
        bool consistent = true;

        size_t sumCustomers = 0;
        size_t sumTables = 0;

        for (auto it = payload.tableMap.begin();
             it != payload.tableMap.end(); ++it) {
            size_t cw = sum(it->second.second);
            if (it->second.first != cw) {
                consistent = false;
                std::cerr << "sum_k(cwk) [" << cw << "] != cw [" << it->second.first << "]"
                          << std::endl;
            }
            sumCustomers += cw;
            sumTables += it->second.second.size();
        }

        consistent =    (sumCustomers == payload.sumCustomers)
            && (sumTables == payload.sumTables)
            && consistent;
        if (!consistent) {
            std::cerr << "Restaurant internally inconsistent!"
                      << " " << sumCustomers << "!=" << payload.sumCustomers
                      << ", " << sumTables << "!=" << payload.sumTables
                      << std::endl;
        }
        return consistent;
    }

    template <typename Dish>
    l_type HistogramRestaurant<Dish>::getC(void* payloadPtr, Dish type) const {
        Payload& payload = *((Payload*)payloadPtr);
        typename Payload::TableMap::iterator it = payload.tableMap.find(type);
        if (it != payload.tableMap.end()) {
            return it->second.cw; //cw
        } else {
            return 0;
        }
    }

    template <typename Dish>
    l_type HistogramRestaurant<Dish>::getC(void* payloadPtr) const {
        return ((Payload*)payloadPtr)->sumCustomers;
    }

    template <typename Dish>
    l_type HistogramRestaurant<Dish>::getT(void* payloadPtr, Dish type) const {
        Payload& payload = *((Payload*)payloadPtr);
        typename Payload::TableMap::iterator it = payload.tableMap.find(type);
        if (it != payload.tableMap.end()) {
            return it->second.tw; //tw
        } else {
            return 0;
        }
    }

    template <typename Dish>
    l_type HistogramRestaurant<Dish>::getT(void* payloadPtr) const {
        return ((Payload*)payloadPtr)->sumTables;
    }

    template <typename Dish>
    double HistogramRestaurant<Dish>::computeProbability(void*  payloadPtr,
                                                         Dish type,
                                                         double parentProbability,
                                                         double discount,
                                                         double concentration) const {
        Payload& payload = *((Payload*)payloadPtr);
        size_t cw = 0;
        size_t tw = 0;
        typename Payload::TableMap::iterator it = payload.tableMap.find(type);
        if (it != payload.tableMap.end()) {
            cw = (*it).second.cw;
            tw = (*it).second.tw;
        }
        return computeHPYPPredictive(cw, // cw
                                     tw, // tw
                                     payload.sumCustomers, // c
                                     payload.sumTables, // t
                                     parentProbability,
                                     discount,
                                     concentration);
    }

    template <typename Dish>
    double HistogramRestaurant<Dish>::computeLogProbability(void*  payloadPtr,
                                                            Dish type,
                                                            double parentLogProbability,
                                                            double discount,
                                                            double concentration) const {
        Payload& payload = *((Payload*)payloadPtr);
        size_t cw = 0;
        size_t tw = 0;
        typename Payload::TableMap::iterator it = payload.tableMap.find(type);
        if (it != payload.tableMap.end()) {
            cw = (*it).second.cw;
            tw = (*it).second.tw;
        }
        return computeLogHPYPPredictive(cw, // cw
                                        tw, // tw
                                        payload.sumCustomers, // c
                                        payload.sumTables, // t
                                        parentLogProbability,
                                        discount,
                                        concentration);
    }

    template <typename Dish>
    typename IHPYPBaseRestaurant<Dish>::TypeVector HistogramRestaurant<Dish>::getTypeVector(
        void* payloadPtr) const {
        Payload& payload = *((Payload*)payloadPtr);
        typename IHPYPBaseRestaurant<Dish>::TypeVector typeVector;
        typeVector.reserve(payload.tableMap.size());
        for (auto it = payload.tableMap.begin();
             it != payload.tableMap.end(); ++it) {
            typeVector.push_back(it->first);
        }
        return typeVector;
    }

    template <typename Dish>
    const IPayloadFactory& HistogramRestaurant<Dish>::getFactory() const {
        return this->payloadFactory;
    }

    template <typename Dish>
    void HistogramRestaurant<Dish>::updateAfterSplit(void* longerPayloadPtr,
                                                     void* shorterPayloadPtr,
                                                     double discountBeforeSplit,
                                                     double discountAfterSplit,
                                                     bool parentOnly) const {
        Payload& payload = *((Payload*)longerPayloadPtr);
        Payload& newParent = *((Payload*)shorterPayloadPtr);

        // make sure the parent is empty
        CHECK(newParent.sumCustomers == 0);
        CHECK(newParent.sumTables == 0);
        CHECK(newParent.tableMap.size() == 0);

        for(typename Payload::TableMap::iterator it = payload.tableMap.begin();
            it != payload.tableMap.end(); ++it) {
            Dish type = it->first;
            typename Payload::Arrangement& arrangement = payload.tableMap[type];
            typename Payload::Arrangement& parentArrangement = newParent.tableMap[type];

            if (arrangement.cw == 1) { // just one customer -- can't split
                // seat customer at his own table in parent
                ++newParent.sumCustomers; // c
                ++newParent.sumTables; // t
                parentArrangement.cw = 1; // cw
                parentArrangement.tw = 1; // cw
                parentArrangement.histogram[1] = 1; // cwk
            } else {
                // The correct thing to do is the following:
                // For each type s in this restaurant
                //      for each table with cwk customers
                //          sample a partition cwkj from a PYP(-d1d2,d2)
                //          make the resulting table the tables in this restaurant
                //          for each k, create a table in the parent restaurant and
                //          seat |cskj| customers on that table

                // make copy of old seating arrangement
                typename Payload::Arrangement oldArrangement = arrangement;

                if (!parentOnly) {
                    // remove old seating arrangement from this restaurant
                    payload.sumCustomers -= arrangement.cw;
                    payload.sumTables -= arrangement.tw;
                    arrangement.cw = 0;
                    arrangement.tw = 0;
                    arrangement.histogram.clear();
                }

                for (typename Payload::Histogram::iterator it = oldArrangement.histogram.begin();
                     it != oldArrangement.histogram.end();
                     ++it) { // for all table sizes
                    for (l_type k = 0; k < (*it).second; ++k) { // all tables of this size
                        std::vector<int> frag = sample_crp_c(discountAfterSplit,
                                                             -discountBeforeSplit,
                                                             (*it).first);
                        // add table to parent with cwk = frag.size()
                        parentArrangement.histogram[frag.size()] += 1;
                        parentArrangement.cw += frag.size();
                        newParent.sumCustomers += frag.size();
                        parentArrangement.tw += 1;
                        newParent.sumTables += 1;

                        if (!parentOnly) {
                            // add split tables to this restaurant
                            for (auto j = 0; j < frag.size(); ++j) {
                                int cwkj = frag[j];
                                arrangement.histogram[cwkj] += 1;
                                arrangement.cw += cwkj;
                                arrangement.tw += 1;
                                payload.sumTables += 1;
                                payload.sumCustomers += cwkj;
                            }
                        }
                    }
                }
            }
        }
    }

    template <typename Dish>
    bool HistogramRestaurant<Dish>::addCustomer(void*  payloadPtr,
                                                Dish type,
                                                double parentProbability,
                                                double discount,
                                                double concentration,
                                                void* additionalData) const {
        CHECK(additionalData == NULL);

        Payload& payload = *((Payload*)payloadPtr);
        typename Payload::Arrangement& arrangement = payload.tableMap[type];

        payload.sumCustomers += 1; // c
        arrangement.cw += 1; // cw
        if (arrangement.cw == 1) {
            // first customer sits at the first table
            // this special case is needed as otherwise things will break when alpha=0
            // and the restaurant has 0 customers in the singleton bucket
            arrangement.histogram[1] += 1;
            arrangement.tw += 1;
            payload.sumTables += 1;
            return true;
        }

        auto numBuckets = arrangement.histogram.size();
        d_vec tableProbs(numBuckets + 1, 0);
        std::vector<size_t> assignment(numBuckets,0);
        int i = 0;
        for(typename Payload::Histogram::iterator it = arrangement.histogram.begin();
            it != arrangement.histogram.end();
            ++it) {
            // prob for joining a table of size k: \propto (k - d)t[k]
            tableProbs[i] = ((*it).first - discount) * (*it).second;
            assignment[i] = (*it).first;
            ++i;
        }
        // prob for new table: \propto (alpha + d*t)*P0
        // this can be 0 for the first customer if concentration=0, but that is ok
        tableProbs[numBuckets] = (concentration +
                                  discount *
                                  static_cast<double>(payload.sumTables)) *
            parentProbability;

        // choose table for customer to sit at
        int sample = nn::sample_unnormalized_pdf(tableProbs, nn::rng::get());
        CHECK(sample <= numBuckets);

        if(sample == numBuckets) {
            // sit at new table
            arrangement.histogram[1] += 1;
            arrangement.tw += 1;
            payload.sumTables += 1;
            return true;
        } else {
            // existing table
            arrangement.histogram[assignment[sample]] -= 1;
            if (arrangement.histogram[assignment[sample]] == 0) {
                // delete empty bucket from histogram
                arrangement.histogram.erase(assignment[sample]);
            }
            arrangement.histogram[assignment[sample]+1] += 1;
            return false;
        }
    }

    template <typename Dish>
    bool HistogramRestaurant<Dish>::logAddCustomer(void* payloadPtr,
                                                   Dish type,
                                                   double parentLogProbability,
                                                   double discount,
                                                   double concentration,
                                                   void* additionalData) const {
        CHECK(additionalData == NULL);

        Payload& payload = *((Payload*)payloadPtr);
        typename Payload::Arrangement& arrangement = payload.tableMap[type];

        payload.sumCustomers += 1; // c
        arrangement.cw += 1; // cw
        if (arrangement.cw == 1) {
            // first customer sits at the first table
            // this special case is needed as otherwise things will break when alpha=0
            // and the restaurant has 0 customers in the singleton bucket
            arrangement.histogram[1] += 1;
            arrangement.tw += 1;
            payload.sumTables += 1;
            return true;
        }

        auto numBuckets = arrangement.histogram.size();
        d_vec tableProbs(numBuckets + 1, 0);
        std::vector<size_t> assignment(numBuckets,0);
        int i = 0;
        for(auto it = arrangement.histogram.begin();
            it != arrangement.histogram.end();
            ++it) {
            // prob for joining a table of size k: \propto (k - d)t[k]
            tableProbs[i] = log(((*it).first - discount) * (*it).second);
            assignment[i] = (*it).first;
            ++i;
        }

        // prob for new table: \propto (alpha + d*t)*P0
        // this can be 0 for the first customer if concentration=0, but that is ok
        tableProbs[numBuckets] = log(concentration + discount*static_cast<double>(payload.sumTables)) + parentLogProbability;
//  tableProbs[numBuckets] = nn::log_add(parentLogProbability + log(concentration), parentLogProbability + log(discount) + log(payload.sumTables));

        // choose table for customer to sit at
        auto sample = nn::sample_unnormalized_lnpdf(tableProbs, nn::rng::get());
        CHECK(sample <= numBuckets);

        if(sample == numBuckets) {
            // sit at new table
            arrangement.histogram[1] += 1;
            arrangement.tw += 1;
            payload.sumTables += 1;
            return true;
        } else {
            // existing table
            arrangement.histogram[assignment[sample]] -= 1;
            if (arrangement.histogram[assignment[sample]] == 0) {
                // delete empty bucket from histogram
                arrangement.histogram.erase(assignment[sample]);
            }
            arrangement.histogram[assignment[sample]+1] += 1;
            return false;
        }
    }

    template <typename Dish>
    bool HistogramRestaurant<Dish>::removeCustomer(void* payloadPtr,
                                                   Dish type,
                                                   double discount,
                                                   void* additionalData) const {
        CHECK(additionalData == NULL);

        Payload& payload = *((Payload*)payloadPtr);
        typename Payload::Arrangement& arrangement = payload.tableMap[type];

        arrangement.cw -= 1; // cw
        payload.sumCustomers -= 1; // c

        CHECK(arrangement.cw >= 0);
        CHECK(payload.sumCustomers >= 0);

        auto numBuckets = arrangement.histogram.size();
        int singletonBucket = -1; // invalid bucket
        d_vec tableProbs(numBuckets, 0);
        std::vector<int> assignment(numBuckets,0);

        int i = 0;
        for(auto it = arrangement.histogram.begin();
            it != arrangement.histogram.end();
            ++it) {
            // prob for choosing a bucket k*t[k]
            tableProbs[i] = (*it).first * (*it).second;
            assignment[i] = (*it).first;
            if ((*it).first == 1) {
                singletonBucket = i;
            }
            ++i;
        }

        // choose table for customer to sit at
        auto sample = nn::sample_unnormalized_pdf(tableProbs, nn::rng::get());
        CHECK(sample <= numBuckets);
        CHECK(tableProbs[sample] > 0);

        if (sample == singletonBucket) {
            CHECK(arrangement.histogram[1] > 0);
            // singleton -> drop table
            arrangement.histogram[1] -= 1;
            if (arrangement.histogram[1] == 0) {
                // delete empty bucket from histogram
                //arrangement.histogram.erase(1);
            }
            arrangement.tw -= 1;
            payload.sumTables -= 1;

            CHECK(arrangement.tw >= 0);
            CHECK(payload.sumTables >= 0);

            return true;
        } else {
            // non-singleton bucket
            arrangement.histogram[assignment[sample]] -= 1;
            CHECK(arrangement.histogram[assignment[sample]] >= 0);
            if (arrangement.histogram[assignment[sample]] == 0) {
                // delete empty bucket from histogram
                arrangement.histogram.erase(assignment[sample]);
            }
            arrangement.histogram[assignment[sample]-1] += 1;
            return false;
        }
    }

    template <typename Dish>
    void* HistogramRestaurant<Dish>::createAdditionalData(void* payloadPtr,
                                                          double discount,
                                                          double concentration) const {
        return NULL;
    }

    template <typename Dish>
    void HistogramRestaurant<Dish>::freeAdditionalData(void* additionalData) const {
        // nothing to be done
    }

    template <typename Dish>
    bool HistogramRestaurant<Dish>::checkConsistency(void* payloadPtr) const {
        Payload& payload = *((Payload*)payloadPtr);
        bool consistent = true;

        size_t sumCustomers = 0;
        size_t sumTables = 0;

        for (auto it = payload.tableMap.begin();
             it != payload.tableMap.end(); ++it) {
            size_t cw = 0;
            size_t tw = 0;
            for (auto hit = (*it).second.histogram.begin();
                 hit != (*it).second.histogram.end();
                 ++hit) {
                cw += (*hit).first * (*hit).second;
                tw += (*hit).second;
            }

            if (it->second.cw != cw || it->second.tw != tw) {
                consistent = false;
                LOG(INFO) << "sum_k(cwk) [" << cw << "] != cw [" << it->second.cw << "]";
            }
            sumCustomers += cw;
            sumTables += tw;
        }

        consistent =    (sumCustomers == payload.sumCustomers)
            && (sumTables == payload.sumTables)
            && consistent;

        if(!consistent) {
            LOG(INFO) << "Restaurant internally inconsistent!"
                      << " " << sumCustomers << "!=" << payload.sumCustomers
                      << ", " << sumTables << "!=" << payload.sumTables;
        }

        return consistent;
    }
}

#endif
