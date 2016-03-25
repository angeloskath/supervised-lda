#include <iostream>
#include <Eigen/Core>
#include <gtest/gtest.h>

#include "test/utils.hpp"

#include "e_step_utils.hpp"

using namespace Eigen;


// T will be available as TypeParam in TYPED_TEST functions
template <typename T>
class TestApproximateSupervisedExpectationStep : public ParameterizedTest<T> {};

TYPED_TEST_CASE(TestApproximateSupervisedExpectationStep, ForFloatAndDouble);

TYPED_TEST(TestApproximateSupervisedExpectationStep, ComputeApproximateSupervisedPhi) {
    VectorXi X(10);
    X << 22, 49, 0, 2, 16, 35, 94, 3, 25, 10;
    VectorX<TypeParam> X_ratio = X.cast<TypeParam>() / X.sum();
    int  y = 0;

    MatrixX<TypeParam> phi = MatrixX<TypeParam>::Random(5, 10);
    VectorX<TypeParam> alpha = VectorX<TypeParam>::Constant(5, 0.1);
    MatrixX<TypeParam> beta = MatrixX<TypeParam>::Random(5, 10);
    MatrixX<TypeParam> eta = MatrixX<TypeParam>::Random(5, 3);
    VectorX<TypeParam> gamma = VectorX<TypeParam>::Constant(5, X.sum() / 5.0);
    
    // Normalize beta and phi
    beta.array() -= beta.minCoeff() - 0.001;
    beta.array().rowwise() /= beta.colwise().sum().array();

    phi.array() -= phi.minCoeff() - 0.001;
    phi.array().rowwise() /= phi.colwise().sum().array();
    
    MatrixX<TypeParam> phi_unsupervised = phi;
    MatrixX<TypeParam> phi_supervised = phi;
    MatrixX<TypeParam> phi_supervised_approximation = phi;
    MatrixX<TypeParam> phi_old = phi;

    MatrixX<TypeParam> h(5, 10);
    e_step_utils::compute_h<TypeParam>(X, X_ratio, eta, phi, h);

    TypeParam likelihood_baseline = e_step_utils::compute_supervised_likelihood(
        X,
        y,
        alpha,
        beta,
        eta,
        phi,
        gamma,
        h
    );

    // Compute phi with unsupervised method
    e_step_utils::compute_unsupervised_phi<TypeParam> (
        beta,
        gamma,
        phi_unsupervised
    );
    // Compute new likelihood
    TypeParam likelihood_unsupervised = e_step_utils::compute_supervised_likelihood(
        X,
        y,
        alpha,
        beta,
        eta,
        phi_unsupervised,
        gamma
    );
    
    // Compute phi with the supervised method
    e_step_utils::fixed_point_iteration<TypeParam> (
        X_ratio,
        y,
        beta,
        eta,
        gamma,
        h,
        phi_old,
        phi_supervised
    );
    // Compute new likelihood
    TypeParam likelihood_supervised = e_step_utils::compute_supervised_likelihood(
        X,
        y,
        alpha,
        beta,
        eta,
        phi_supervised,
        gamma
    );

    // Compute phi with the supervised approximation
    e_step_utils::compute_supervised_approximate_phi<TypeParam> (
        X_ratio,
        y,
        beta,
        eta,
        gamma,
        phi_supervised_approximation
    );
    // Compute new likelihood
    TypeParam likelihood_supervised_approximation = e_step_utils::compute_supervised_likelihood(
        X,
        y,
        alpha,
        beta,
        eta,
        phi_supervised_approximation,
        gamma
    );

    EXPECT_GT(likelihood_supervised_approximation, likelihood_baseline);
    EXPECT_GT(likelihood_supervised_approximation, likelihood_unsupervised);
    EXPECT_GT(likelihood_supervised, likelihood_unsupervised);
    // What should the following be
    //
    // EXPECT_GT(likelihood_supervised_approximation, likelihood_supervised);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
