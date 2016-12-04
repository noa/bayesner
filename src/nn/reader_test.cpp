#include <gtest/gtest.h>

#include <string>
#include <fstream>

#include <nn/reader.hpp>

#include <cereal/archives/binary.hpp>

TEST(Reader, CerealWorks) {
    using namespace nn;
    typedef CoNLLCorpus<> Corpus;
    Corpus corpus1("<bos>",
                  "<eos>",
                  "<s>",
                  "<unk>",
                  "O");
    std::string fn {"/tmp/reader.cereal"};
    {
        std::ofstream os(fn, std::ios::binary);
        cereal::BinaryOutputArchive oarchive(os);
        oarchive( corpus1 );
    }
    {
        Corpus corpus2;
        std::ifstream is(fn, std::ios::binary);
        cereal::BinaryInputArchive iarchive(is);
        iarchive( corpus2 );
        ASSERT_EQ(corpus1.bos, corpus2.bos);
        ASSERT_EQ(corpus1.eos, corpus2.eos);
        ASSERT_EQ(corpus1.space, corpus2.space);
        ASSERT_EQ(corpus1.unk, corpus2.unk);
        ASSERT_EQ(corpus1.other_tag, corpus2.other_tag);
    }
}
