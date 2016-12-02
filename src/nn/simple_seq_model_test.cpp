#include <gtest/gtest.h>

#include <nn/data.hpp>
#include <nn/uniform.hpp>
#include <nn/fixed_depth_hpyp.hpp>
#include <nn/simple_seq_model.hpp>

#include <cereal/archives/binary.hpp>

TEST(SimpleSequenceModel, CerealWorks) {
    using namespace nn;
    rng::init();
    std::string fn {"/tmp/simple_sequence_model.cereal"};
    syms seq2 { 0, 1, 2, 3, 4 };
    double lp {0};
    {
        simple_seq_model<> model(5, 0, 4);
        syms seq1 { 0, 1, 1, 2, 4 };
        model.observe(seq1);
        lp = model.log_prob(seq2);
        std::ofstream os(fn, std::ios::binary);
        cereal::BinaryOutputArchive oarchive(os);
        oarchive( model );
    }
    {
        simple_seq_model<> model;
        {
            std::ifstream is(fn, std::ios::binary);
            cereal::BinaryInputArchive iarchive(is);
            iarchive( model );
        }
        ASSERT_EQ(model.get_initial_symbol(), 0);
        ASSERT_EQ(model.get_final_symbol(), 4);
        ASSERT_EQ(model.get_base().cardinality(), 5);
        double lp2 = model.log_prob(seq2);
        ASSERT_EQ(lp, lp2);
    }
}
