// Nicholas Andrews
// noandrews@gmail.com

#ifndef __CPYP_RESTAURANT_HPP__
#define __CPYP_RESTAURANT_HPP__

#include "rng.hpp"
#include "restaurants.hpp"
#include "crp.hpp"

namespace nn {

    template <typename Dish>
    class CRestaurant {
    public:
        typedef typename std::vector<Dish>    TypeVector;
        typedef typename TypeVector::iterator TypeVectorIterator;

        CRestaurant() : payloadFactory() {}
        ~CRestaurant() {}

        bool addCustomer(void* payloadPtr,
                         Dish type,
                         double p0,
                         double discount,
                         double strength,
                         void* additionalData = nullptr) const {
            Payload& payload = *((Payload*)payloadPtr);
            auto& dish_locs     = payload.dish_locs;
            auto& num_tables    = payload.num_tables;
            auto& num_customers = payload.num_customers;
            crp_table_manager<1>& loc = dish_locs[dish];
            bool share_table = false;
            if (loc.num_customers()) {
                const F p_empty = F(strength + num_tables * discount) * p0;
                const F p_share = F(loc.num_customers() - loc.num_tables() * discount);
                share_table = sample_bernoulli(p_empty, p_share, eng);
            }
            if (share_table) {
                unsigned n = loc.share_table(discount, eng);
                update_llh_add_customer_to_table_seating(n);
            } else {
                loc.create_table();
                update_llh_add_customer_to_table_seating(0);
                ++num_tables;
            }
            ++num_customers;
            return (share_table ? 0 : 1);
        }

        bool logAddCustomer(void* payloadPtr,
                            Dish type,
                            double log_p0,
                            double discount,
                            double concentration,
                            void* additionalData = nullptr) const {
            CHECK(false) << "unimplemented";
            return true;
        }

        bool removeCustomer(void* payloadPtr,
                            Dish type,
                            double discount,
                            void* additionalData) const {
            Payload& payload = *((Payload*)payloadPtr);
            auto& dish_locs     = payload.dish_locs;
            auto& num_tables    = payload.num_tables;
            auto& num_customers = payload.num_customers;
            crp_table_manager<1>& loc = dish_locs[dish];
            assert(loc.num_customers());
            if (loc.num_customers() == 1) {
                update_llh_remove_customer_from_table_seating(1);
                dish_locs.erase(dish);
                --num_tables_;
                --num_customers_;
                // q = 1 since this is the first customer
                return -1;
            } else {
                unsigned selected_table_postcount = 0;
                int delta = loc.remove_customer(eng, &selected_table_postcount).second;
                update_llh_remove_customer_from_table_seating(selected_table_postcount + 1);
                --num_customers;
                if (delta) --num_tables;
                return delta;
            }
        }

        l_type getC(void* payloadPtr, Dish type) const {
            Payload& payload = *((Payload*)payloadPtr);
            auto& dish_locs = payload.dish_locs;
            auto it = dish_locs.find(type);
            if (it == dish_locs.end()) return 0;
            return it->second.num_tables();
        }

        l_type getC(void* payloadPtr) const {
            Payload& payload = *((Payload*)payloadPtr);
            return payload.num_customers;
        }

        l_type getT(void* payloadPtr, Dish type) const {
            Payload& payload = *((Payload*)payloadPtr);
            auto& dish_locs     = payload.dish_locs;
            auto& num_tables    = payload.num_tables;
            auto& num_customers = payload.num_customers;
            auto it = dish_locs.find(dish);
            if (it == dish_locs.end()) return 0;
            return it->second.num_table();
        }

        l_type getT(void* payloadPtr) const {
            Payload& payload = *((Payload*)payloadPtr);
            return payload.num_tables;
        }

        // call this before changing the number of tables / customers
        void update_llh_add_customer_to_table_seating(size_t n,
                                                      size_t num_customers,
                                                      size_t num_tables,
                                                      double discount,
                                                      double strength) {
            unsigned t = 0;
            if (n == 0) t = 1;
            llh_ -= log(strength + num_customers);
            if (t == 1) llh_ += log(discount) + log(strength / discount + num_tables);
            if (n > 0) llh_ += log(n - discount);
        }

        // call this before changing the number of tables / customers
        void update_llh_remove_customer_from_table_seating(size_t n,
                                                           size_t num_customers,
                                                           size_t num_tables,
                                                           double discount,
                                                           double strength) {
            unsigned t = 0;
            if (n == 1) t = 1;
            llh_ += log(strength + num_customers - 1);
            if (t == 1) llh_ -= log(discount_) + log(strength_ / discount_ + num_tables_ - 1);
            if (n > 1) llh_ -= log(n - discount_ - 1);
        }

        double computeProbability(void* payloadPtr,
                                  Dish type,
                                  double p0,
                                  double discount,
                                  double strength) const {
            Payload& payload = *((Payload*)payloadPtr);
            auto& dish_locs     = payload.dish_locs;
            auto& num_tables    = payload.num_tables;
            auto& num_customers = payload.num_customers;
            if (num_tables == 0) return p0;
            auto it = dish_locs.find(dish);
            const F r = F(num_tables * discount + strength);
            if (it == dish_locs.end()) {
                return r * p0 / F(num_customers_ + strength);
            } else {
                return (F(it->second.num_customers() - discount *
                          it->second.num_tables()) + r * p0) /
                    F(num_customers + strength);
            }
        }

        double computeLogProbability(void* payloadPtr,
                                     Dish type,
                                     double log_p0,
                                     double discount,
                                     double concentration) const {
            return 0;
        }

        double computeLogCacheProb(void* payloadPtr,
                                   Dish type,
                                   double discount,
                                   double concentration) const {
            CHECK(false) << "unimplemented";
            return 0;
        }

        double computeLogNewProb(void* payloadPtr,
                                 double parentLogProbability,
                                 double discount,
                                 double concentration) const {
            CHECK(false) << "unimplemented";
            return 0;
        }

        TypeVector getTypeVector(void* payloadPtr) const {
            TypeVector v;
            return v;
        }

        const IPayloadFactory& getFactory() const {
            return this->payloadFactory;
        }

        class Payload {
        private:
            unsigned num_tables_;
            unsigned num_customers_;
            std::unordered_map<Dish, crp_table_manager<1>, DishHash> dish_locs;

        public:
            Payload() {}
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
        };

        const PayloadFactory payloadFactory;
    };
}

#endif
