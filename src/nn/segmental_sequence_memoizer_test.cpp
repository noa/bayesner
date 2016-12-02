#include <gtest/gtest.h>

#include <string>
#include <fstream>

#include <nn/segmental_sequence_memoizer.hpp>

#include <cereal/archives/binary.hpp>

TEST(SegmentalSequenceMemoizer, CerealWorks) {
    using namespace nn;
    std::string fn {"/tmp/segmental_sequence_memoizer.cereal"};
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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
