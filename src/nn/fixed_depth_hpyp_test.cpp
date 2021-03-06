#include <gtest/gtest.h>

#include <nn/uniform.hpp>
#include <nn/fixed_depth_hpyp.hpp>
#include <nn/log_fixed_depth_hpyp.hpp>

#include <cereal/archives/binary.hpp>

TEST(FixedDepthHPYP, CerealWorks) {
    using namespace nn;
    typedef HashIntegralMeasure<size_t>        Base;
    typedef FixedDepthHPYP<size_t,size_t,Base> Model;
    rng::init(); // necessary for sampling table assignments
    std::string fn {"/tmp/fixed_depth_hpyp.cereal"};
    std::vector<size_t> test { 0, 1, 2, 3, 4 };
    double lp {0};
    {
        Base H;
        for(size_t i=0; i<5; ++i) H.add(i, 1.0);
        Model model1(H);
        std::vector<size_t> obs { 0, 1, 2, 2, 1, 3, 0, 1, 2, 3, 4 };
        for(auto it = obs.begin()+1; it != obs.end(); ++it) {
            model1.observe(obs.begin(), it, *it);
        }
        for(auto it = test.begin()+1; it != test.end(); ++it) {
            lp += model1.log_prob(test.begin(), it, *it);
        }
        std::ofstream os(fn, std::ios::binary);
        cereal::BinaryOutputArchive oarchive(os);
        oarchive( model1 );
    }
    {
        Model model2;
        std::ifstream is(fn, std::ios::binary);
        cereal::BinaryInputArchive iarchive(is);
        iarchive( model2 );
        ASSERT_EQ(model2.cardinality(), 5);
        double lp2 {0};
        for(auto it = test.begin()+1; it != test.end(); ++it) {
            lp2 += model2.log_prob(test.begin(), it, *it);
        }
        ASSERT_EQ(lp, lp2);
    }
}

TEST(LogFixedDepthHPYP, CerealWorks) {
    using namespace nn;
    typedef HashIntegralMeasure<size_t>           Base;
    typedef LogFixedDepthHPYP<size_t,size_t,Base> Model;
    rng::init(); // necessary for sampling table assignments
    std::string fn {"/tmp/log_fixed_depth_hpyp.cereal"};
    std::vector<size_t> test { 0, 1, 2, 3, 4 };
    double lp {0};
    {
        Base H;
        for(size_t i=0; i<5; ++i) H.add(i, 1.0);
        Model model1(H);
        std::vector<size_t> obs { 0, 1, 2, 2, 1, 3, 0, 1, 2, 3, 4 };
        for(auto it = obs.begin()+1; it != obs.end(); ++it) {
            model1.observe(obs.begin(), it, *it);
        }
        for(auto it = test.begin()+1; it != test.end(); ++it) {
            lp += model1.log_prob(test.begin(), it, *it);
        }
        std::ofstream os(fn, std::ios::binary);
        cereal::BinaryOutputArchive oarchive(os);
        oarchive( model1 );
    }
    {
        Model model2;
        std::ifstream is(fn, std::ios::binary);
        cereal::BinaryInputArchive iarchive(is);
        iarchive( model2 );
        ASSERT_EQ(model2.cardinality(), 5);
        double lp2 {0};
        for(auto it = test.begin()+1; it != test.end(); ++it) {
            lp2 += model2.log_prob(test.begin(), it, *it);
        }
        ASSERT_EQ(lp, lp2);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
