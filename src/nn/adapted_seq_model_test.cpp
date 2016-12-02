#include <gtest/gtest.h>

#include <nn/uniform.hpp>
#include <nn/fixed_depth_hpyp.hpp>
#include <nn/simple_seq_model.hpp>
#include <nn/adapted_seq_model.hpp>

#include <cereal/archives/binary.hpp>

TEST(SimpleSequenceModel, CerealWorks) {
    using namespace nn;
    rng::init();
    std::string fn {"/tmp/simple_sequence_model.cereal"};
    seq_t seq2 { 0, 1, 2, 3, 4 };
    double lp {0};
    {
        simple_seq_model<> model(5, 0, 4);
        seq_t seq1 { 0, 1, 1, 2, 4 };
        LOG(INFO) << "observing seq1...";
        model.observe(seq1);
        LOG(INFO) << "calc log prob of seq2...";
        lp = model.log_prob(seq2);
        LOG(INFO) << "serializing model...";
        std::ofstream os(fn, std::ios::binary);
        cereal::BinaryOutputArchive oarchive(os);
        oarchive( model );
    }
    {
        LOG(INFO) << "default constructing stub...";
        simple_seq_model<> model;
        {
            LOG(INFO) << "loading model...";
            std::ifstream is(fn, std::ios::binary);
            cereal::BinaryInputArchive iarchive(is);
            iarchive( model );
        }
        ASSERT_EQ(model.get_initial_symbol(), 0);
        ASSERT_EQ(model.get_final_symbol(), 4);
        ASSERT_EQ(model.get_base()->cardinality(), 5);
        LOG(INFO) << "calc log prob of seq2...";
        double lp2 = model.log_prob(seq2);
        ASSERT_EQ(lp, lp2);
        LOG(INFO) << "all done...";
    }
    LOG(INFO) << "all done...";
}

// TEST(AdaptedSequenceModel, CerealWorks) {
//     using namespace nn;
//     typedef HashIntegralMeasure<size_t>        Base;
//     typedef FixedDepthHPYP<size_t,size_t,Base> Model;
//     rng::init(); // necessary for sampling table assignments
//     std::string fn {"/tmp/fixed_depth_hpyp.cereal"};
//     std::vector<size_t> test { 0, 1, 2, 3, 4 };
//     double lp {0};
//     {
//         std::shared_ptr<Base> H = std::make_shared<Base>();
//         for(size_t i=0; i<5; ++i) H->add(i, 1.0);
//         Model model1(H);
//         std::vector<size_t> obs { 0, 1, 2, 2, 1, 3, 0, 1, 2, 3, 4 };
//         for(auto it = obs.begin()+1; it != obs.end(); ++it) {
//             model1.observe(obs.begin(), it, *it);
//         }
//         std::vector<size_t> test { 0, 1, 2, 3, 4 };
//         for(auto it = test.begin()+1; it != test.end(); ++it) {
//             lp += model1.log_prob(test.begin(), it, *it);
//         }
//         std::ofstream os(fn, std::ios::binary);
//         cereal::BinaryOutputArchive oarchive(os);
//         oarchive( model1 );
//     }
//     {
//         Model model2;
//         std::ifstream is(fn, std::ios::binary);
//         cereal::BinaryInputArchive iarchive(is);
//         iarchive( model2 );
//         double lp2 {0};
//         for(auto it = test.begin()+1; it != test.end(); ++it) {
//             lp2 += model2.log_prob(test.begin(), it, *it);
//         }
//         ASSERT_EQ(lp, lp2);
//     }
// }

// int main(int argc, char **argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }
