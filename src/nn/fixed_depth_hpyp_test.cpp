#include <gtest/gtest.h>

#include <nn/uniform.hpp>
#include <nn/fixed_depth_hpyp.hpp>

TEST(FixedDepthHPYP, CerealWorks) {
    using namespace nn;
    HashIntegralMeasure<size_t> H;
    for(size_t i=0; i<5; ++i) H.add(i, 1.0);
    FixedDepthHPYP<size_t,size_t,HashIntegralMeasure<size_t>> model(&H);

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
