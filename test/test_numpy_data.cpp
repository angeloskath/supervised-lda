
#include <iostream>
#include <sstream>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "test/utils.hpp"

#include "NumpyFormat.hpp"

using namespace Eigen;


// T will be available as TypeParam in TYPED_TEST functions
template <typename T>
class TestNumpyData : public ParameterizedTest<T> {};


TYPED_TEST_CASE(TestNumpyData, ForFloatAndDouble);


TYPED_TEST(TestNumpyData, SimpleReadWrite) {
    MatrixX<TypeParam> A = MatrixX<TypeParam>::Random(10, 20);
    MatrixX<TypeParam> B;
    NumpyInput<TypeParam> input;

    std::stringstream ss;
    ss << NumpyOutput<TypeParam>(A);
    ss.seekg(0);
    ss >> input;

    B = input;

    ASSERT_TRUE(A==B);
}

TYPED_TEST(TestNumpyData, SimpleRead) {
    const unsigned char array_of_doubles[] = {
        147, 78, 85, 77, 80, 89, 1, 0, 70, 0, 123, 39, 100, 101, 115, 99, 114,
        39, 58, 32, 39, 60, 102, 56, 39, 44, 32, 39, 102, 111, 114, 116, 114,
        97, 110, 95, 111, 114, 100, 101, 114, 39, 58, 32, 70, 97, 108, 115,
        101, 44, 32, 39, 115, 104, 97, 112, 101, 39, 58, 32, 40, 50, 44, 32,
        50, 41, 44, 32, 125, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 10, 196,
        191, 59, 106, 239, 185, 228, 63, 226, 242, 207, 251, 10, 110, 238, 63,
        200, 117, 190, 187, 60, 139, 197, 63, 218, 253, 92, 142, 118, 64, 231,
        63
    };

    std::istringstream ss_doubles(
        std::string(
            reinterpret_cast<const char *>(array_of_doubles),
            sizeof(array_of_doubles)
        )
    );

    NumpyInput<double> ni_doubles;

    ss_doubles >> ni_doubles;
    MatrixX<double> doubles = ni_doubles;
    EXPECT_NEAR(0.647697, doubles(0, 0), 1e-4);
    EXPECT_NEAR(0.168312, doubles(1, 0), 1e-4);
    EXPECT_NEAR(0.950933, doubles(0, 1), 1e-4);
    EXPECT_NEAR(0.726619, doubles(1, 1), 1e-4);

    const unsigned char array_of_floats[] = {
        147, 78, 85, 77, 80, 89, 1, 0, 70, 0, 123, 39, 100, 101, 115, 99, 114,
        39, 58, 32, 39, 60, 102, 52, 39, 44, 32, 39, 102, 111, 114, 116, 114,
        97, 110, 95, 111, 114, 100, 101, 114, 39, 58, 32, 70, 97, 108, 115,
        101, 44, 32, 39, 115, 104, 97, 112, 101, 39, 58, 32, 40, 50, 44, 32,
        50, 41, 44, 32, 125, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 10, 72,
        192, 6, 62, 84, 199, 104, 63, 48, 211, 54, 63, 240, 224, 103, 63
    };

    std::istringstream ss_floats(
        std::string(
            reinterpret_cast<const char *>(array_of_floats),
            sizeof(array_of_floats)
        )
    );

    NumpyInput<float> ni_floats;

    ss_floats >> ni_floats;
    MatrixX<float> floats = ni_floats;
    EXPECT_NEAR(0.131592, floats(0, 0), 1e-4);
    EXPECT_NEAR(0.714159, floats(1, 0), 1e-4);
    EXPECT_NEAR(0.909291, floats(0, 1), 1e-4);
    EXPECT_NEAR(0.905776, floats(1, 1), 1e-4);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
