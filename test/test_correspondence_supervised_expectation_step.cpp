#include <iostream>
#include <Eigen/Core>
#include <gtest/gtest.h>

#include "test/utils.hpp"

#include "e_step_utils.hpp"

using namespace Eigen;


// T will be available as TypeParam in TYPED_TEST functions
template <typename T>
class TestSupervisedCorrespondanceExpectationStep : public ParameterizedTest<T> {};

TYPED_TEST_CASE(TestSupervisedCorrespondanceExpectationStep, ForFloatAndDouble);

TYPED_TEST(TestSupervisedCorrespondanceExpectationStep, ComputeSupervisedCorrespondanceLikelihood) {
    for (int i=0; i<100; i++) {
        VectorX<TypeParam> Xtmp = VectorX<TypeParam>::Random(10).array().abs() * 5;
        VectorXi X = Xtmp.template cast<int>();
        VectorX<TypeParam> X_ratio = X.cast<TypeParam>() / X.sum();
        int y = 0;
        VectorX<TypeParam> alpha = VectorX<TypeParam>::Constant(5, 0.1);
        MatrixX<TypeParam> beta = MatrixX<TypeParam>::Random(5, 10);
        MatrixX<TypeParam> eta = MatrixX<TypeParam>::Random(5, 3);
        MatrixX<TypeParam> phi = MatrixX<TypeParam>::Constant(5, 10, 0.1);
        VectorX<TypeParam> gamma = VectorX<TypeParam>::Constant(5, X.sum() / 5.0);
        VectorX<TypeParam> tau = VectorX<TypeParam>::Constant(10, 1. / 10.0);

        // normalize beta, phi, eta
        phi.array() -= phi.minCoeff() - 0.001;
        phi.array().rowwise() /= phi.colwise().sum().array();
        beta.array() -= beta.minCoeff() - 0.001;
        beta.array().rowwise() /= beta.colwise().sum().array();
        eta.array() -= eta.minCoeff() - 0.001;
        eta.array().rowwise() /= eta.colwise().sum().array();

        TypeParam mu = 2.0;

        TypeParam likelihood = e_step_utils::compute_supervised_correspondence_likelihood(
            X,
            y,
            alpha,
            beta,
            eta,
            phi,
            gamma,
            tau,
            mu
        );
        
        ASSERT_FALSE(std::isnan(likelihood)) << phi.array().log();
        EXPECT_GT(0, likelihood);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

