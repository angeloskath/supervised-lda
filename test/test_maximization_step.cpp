
#include <memory>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "test/utils.hpp"

#include "SupervisedLDA.hpp"
#include "ProgressVisitor.hpp"

using namespace Eigen;


// T will be available as TypeParam in TYPED_TEST functions
template <typename T>
class TestMaximizationStep : public ParameterizedTest<T> {};

TYPED_TEST_CASE(TestMaximizationStep, ForFloatAndDouble);

TYPED_TEST(TestMaximizationStep, Maximization) {
    std::mt19937 rng;
    rng.seed(0);

    SupervisedLDA<TypeParam> lda(0);

    std::vector<Progress<TypeParam> > progress;
    auto visitor = std::make_shared<FunctionVisitor<TypeParam> >(
        [&progress](Progress<TypeParam> p) {
            progress.push_back(p);
        }
    );

    lda.set_progress_visitor(visitor);

    MatrixX<TypeParam> expected_z_bar = MatrixX<TypeParam>::Random(10, 100);
    MatrixX<TypeParam> b(10, 100);
    VectorXi y(100);
    std::uniform_int_distribution<> dis(0, 5);
    for (int i=0; i<100; i++) {
        y[i] = dis(rng);
    }
    MatrixX<TypeParam> beta(10, 100);
    MatrixX<TypeParam> eta(10, 6);
    TypeParam L = 0.1;

    lda.m_step(
        expected_z_bar,
        b,
        y,
        beta,
        eta,
        L,
        1000,
        1e-3
    );

    ASSERT_GT(progress.size(), 2);
    for (int i=1; i<progress.size(); i++) {
        ASSERT_EQ(i-1, progress[i-1].partial_iteration);
        EXPECT_GT(progress[i-1].value, progress[i].value) << "Iteration:" << i;
    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
