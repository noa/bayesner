#include <gtest/gtest.h>

#include <string>
#include <fstream>

#include <nn/uniform.hpp>

#include <cereal/archives/binary.hpp>

TEST(HashIntegralMeasure, CerealWorks) {
    using namespace nn;
    HashIntegralMeasure<size_t> H;
    H.add(0,1.0);
    H.add(3,5.0);
    std::string fn {"/tmp/hash_integral_measure.cereal"};
    {
        std::ofstream os(fn, std::ios::binary);
        cereal::BinaryOutputArchive oarchive(os);
        oarchive( H );
    }
    HashIntegralMeasure<size_t> G;
    {
        std::ifstream is(fn, std::ios::binary);
        cereal::BinaryInputArchive iarchive(is);
        iarchive( G );
    }
    ASSERT_EQ(G.w(0), 1.0);
    ASSERT_EQ(G.w(3), 5.0);
}

TEST(Uniform, CerealWorks) {
    using namespace nn;
    Uniform<size_t> H(5);
    std::string fn {"/tmp/uniform.cereal"};
    {
      std::ofstream os(fn, std::ios::binary);
      cereal::BinaryOutputArchive oarchive(os);
      oarchive( H );
    }
    Uniform<size_t> G;
    {
      std::ifstream is(fn, std::ios::binary);
      cereal::BinaryInputArchive iarchive(is);
      iarchive( G );
    }
    ASSERT_EQ(G.cardinality(), 5);
    ASSERT_EQ(G.prob(0), 1.0/5.0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
