#include <gtest/gtest.h>

#include <string>
#include <fstream>

#include <nn/mutable_symtab.hpp>

#include <cereal/archives/binary.hpp>

TEST(MutableSymbolTable, CerealWorks) {
    using namespace nn;
    std::string fn {"/tmp/mutable_symbol_table.cereal"};
    {
        mutable_symbol_table<> table;
        table.add_key("one");
        table.add_key("two");
        std::ofstream os(fn, std::ios::binary);
        cereal::BinaryOutputArchive oarchive(os);
        oarchive( table );
    }
    {
        mutable_symbol_table<> table;
        std::ifstream is(fn, std::ios::binary);
        cereal::BinaryInputArchive iarchive(is);
        iarchive( table );
        ASSERT_EQ( table.val(0), "one" );
        ASSERT_EQ( table.val(1), "two" );
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
