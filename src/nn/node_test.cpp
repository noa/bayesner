#include <gtest/gtest.h>

#include <string>
#include <fstream>

#include <nn/log.hpp>
#include <nn/rng.hpp>
#include <nn/restaurants.hpp>
#include <nn/node.hpp>

#include <cereal/archives/binary.hpp>

TEST(HashNode, CerealWorks) {
    using namespace nn;
    typedef SimpleFullRestaurant<size_t> Restaurant;
    typedef hash_node<size_t, Restaurant> Node;
    rng::init(); // necessary for sampling table assignments
    std::string fn {"/tmp/node.cereal"};
    size_t nc, nt;
    size_t nc0, nt0;
    {
        //LOG(INFO) << "saving";
        Restaurant r;
        Node node;
        node.get_or_make(0);
        node.get_or_make(2);
        auto crp = node.get_payload();
        //LOG(INFO) << "adding customer";
        r.addCustomer(crp, 0, 0.5, 0.5, 0.5);
        //LOG(INFO) << "adding second customer";
        r.addCustomer(crp, 0, 0.5, 0.5, 0.5);
        nc = node.getC();
        nt = node.getT();
        nc0 = node.getC(0);
        nt0 = node.getT(0);
        //LOG(INFO) << "serializing";
        std::ofstream os(fn, std::ios::binary);
        cereal::BinaryOutputArchive oarchive(os);
        oarchive( node );
    }
    {
        //LOG(INFO) << "loading";
        Node node;
        std::ifstream is(fn, std::ios::binary);
        cereal::BinaryInputArchive iarchive(is);
        iarchive( node );
        ASSERT_TRUE(node.has(0));
        ASSERT_TRUE(node.has(2));
        ASSERT_TRUE(node.getC() == nc);
        ASSERT_TRUE(node.getT() == nt);
        ASSERT_TRUE(node.getC(0) == nc0);
        ASSERT_TRUE(node.getT(0) == nt0);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
