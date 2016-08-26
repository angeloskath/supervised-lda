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
    VectorX<TypeParam> alpha = VectorX<TypeParam>::Constant(5, 0.2);
    MatrixX<TypeParam> beta = MatrixX<TypeParam>::Random(5, 10);
    MatrixX<TypeParam> eta = MatrixX<TypeParam>::Random(5, 3);
    VectorX<TypeParam> gamma(beta.rows());

    // Normalize beta and phi
    beta.array() -= beta.minCoeff() - 0.001;
    beta.array().rowwise() /= beta.colwise().sum().array();

    phi.array() -= phi.minCoeff() - 0.001;
    phi.array().rowwise() /= phi.colwise().sum().array();

    // Compute gamma according to the random phi
    e_step_utils::compute_gamma<TypeParam>(
        X,
        alpha,
        phi,
        gamma
    );

    // Make copies of phi
    MatrixX<TypeParam> phi_unsupervised = phi;
    MatrixX<TypeParam> phi_supervised = phi;
    MatrixX<TypeParam> phi_supervised_approximation = phi;

    // Make copies of gamma
    VectorX<TypeParam> gamma_unsupervised = gamma;
    VectorX<TypeParam> gamma_supervised = gamma;
    VectorX<TypeParam> gamma_supervised_approximation = gamma;

    VectorX<TypeParam> h(5);

    TypeParam likelihood_baseline = e_step_utils::compute_supervised_likelihood(
        X,
        y,
        alpha,
        beta,
        eta,
        phi,
        gamma
    );

    size_t fixed_point_iterations = 5;

    for (int i=0; i<50; i++) {
        // Compute phi with unsupervised method
        e_step_utils::compute_unsupervised_phi<TypeParam> (
            beta,
            gamma_unsupervised,
            phi_unsupervised
        );
        // Compute gamma
        e_step_utils::compute_gamma <TypeParam> (
            X,
            alpha,
            phi_unsupervised,
            gamma_unsupervised
        );
    }

    // Compute new likelihood
    TypeParam likelihood_unsupervised = e_step_utils::compute_supervised_likelihood(
        X,
        y,
        alpha,
        beta,
        eta,
        phi_unsupervised,
        gamma_unsupervised
    );

    for (int i=0; i<50; i++) {
        // Compute phi with the supervised method
        e_step_utils::compute_supervised_phi_gamma<TypeParam> (
            X,
            X_ratio,
            y,
            beta,
            eta,
            fixed_point_iterations,
            phi_supervised,
            gamma_supervised,
            h
        );
    }

    // Compute new likelihood
    TypeParam likelihood_supervised = e_step_utils::compute_supervised_likelihood(
        X,
        y,
        alpha,
        beta,
        eta,
        phi_supervised,
        gamma_supervised
    );

    TypeParam C = 1.0;
    for (int i=0; i<50; i++) {
        // Compute phi with the supervised approximation
        e_step_utils::compute_supervised_approximate_phi<TypeParam> (
            X_ratio,
            X.sum(),
            y,
            beta,
            eta,
            gamma_supervised_approximation,
            C,
            phi_supervised_approximation
        );

        // Compute gamma
        e_step_utils::compute_gamma<TypeParam> (
            X,
            alpha,
            phi_supervised_approximation,
            gamma_supervised_approximation
        );
    }
    // Compute new likelihood
    TypeParam likelihood_supervised_approximation = e_step_utils::compute_supervised_likelihood(
        X,
        y,
        alpha,
        beta,
        eta,
        phi_supervised_approximation,
        gamma_supervised_approximation
    );

    EXPECT_GT(likelihood_supervised_approximation, likelihood_baseline);
    // EXPECT_GT(likelihood_supervised_approximation, likelihood_unsupervised);
    // EXPECT_GT(likelihood_supervised, likelihood_unsupervised);
    //
    // What should the following be
    //
    // EXPECT_GT(likelihood_supervised_approximation, likelihood_supervised);
}
