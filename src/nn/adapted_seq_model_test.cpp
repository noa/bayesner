#include <gtest/gtest.h>

#include <nn/data.hpp>
#include <nn/uniform.hpp>
#include <nn/fixed_depth_hpyp.hpp>
#include <nn/simple_seq_model.hpp>
#include <nn/adapted_seq_model.hpp>

#include <cereal/archives/binary.hpp>

TEST(AdaptedSequenceModel, CerealWorks) {
    using namespace nn;
    rng::init();
    std::string fn {"/tmp/adapted_sequence_model.cereal"};
    syms seq2 { 0, 2, 2, 1, 3, 3, 4 };
    double lp {0};
    {
        adapted_seq_model<>::param p;
        p.nsyms = 5;
        p.BOS = 0;
        p.EOS = 4;
        p.SPACE = 1;
        adapted_seq_model<> model(p);
        syms seq1 { 0, 3, 1, 3, 4 };
        model.observe(seq1);
        lp = model.log_prob(seq2);
        std::ofstream os(fn, std::ios::binary);
        cereal::BinaryOutputArchive oarchive(os);
        oarchive( model );
    }
    {
        adapted_seq_model<> model;
        {
            std::ifstream is(fn, std::ios::binary);
            cereal::BinaryInputArchive iarchive(is);
            iarchive( model );
        }
        double lp2 = model.log_prob(seq2);
        ASSERT_EQ(lp, lp2);
    }
}
