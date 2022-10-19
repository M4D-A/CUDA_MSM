#include <inttypes.h>
#include "src/uint384.hpp"
#include "src/field377.hpp"
#include "src/bls12-377.hpp"
#include <gtest/gtest.h>

int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}