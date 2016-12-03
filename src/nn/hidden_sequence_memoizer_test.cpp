#include <gtest/gtest.h>

#include <string>
#include <fstream>

#include <nn/reader.hpp>
#include <nn/hidden_sequence_memoizer.hpp>

#include <cereal/archives/binary.hpp>

TEST(HiddenSequenceMemoizer, CerealWorks) {
    using namespace nn;
    std::string fn {"/tmp/hidden_sequence_memoizer.cereal"};
    {
        CoNLLCorpus<> corpus("<bos>",
                             "<eos>",
                             "<s>",
                             "<unk>",
                             "O");
        hidden_sequence_memoizer<> model(corpus);
        std::ofstream os(fn, std::ios::binary);
        cereal::BinaryOutputArchive oarchive(os);
        oarchive( model );
    }
    {
        hidden_sequence_memoizer<> model;
        std::ifstream is(fn, std::ios::binary);
        cereal::BinaryInputArchive iarchive(is);
        iarchive( model );
    }
}
