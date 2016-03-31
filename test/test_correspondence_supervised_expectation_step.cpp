#include <iostream>
#include <Eigen/Core>
#include <gtest/gtest.h>

#include "test/utils.hpp"

#include "e_step_utils.hpp"

using namespace Eigen;


// T will be available as TypeParam in TYPED_TEST functions
template <typename T>
class TestSupervisedCorrespondenceExpectationStep : public ParameterizedTest<T> {};

TYPED_TEST_CASE(TestSupervisedCorrespondenceExpectationStep, ForFloatAndDouble);

TYPED_TEST(TestSupervisedCorrespondenceExpectationStep, ComputeSupervisedCorrespondenceLikelihood) {
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
        TypeParam portion = 0.1;

        TypeParam likelihood = e_step_utils::compute_supervised_correspondence_likelihood(
            X,
            y,
            alpha,
            beta,
            eta,
            phi,
            gamma,
            tau,
            mu,
            portion
        );
        
        ASSERT_FALSE(std::isnan(likelihood)) << phi.array().log();
        EXPECT_GT(0, likelihood);
    }
}

TYPED_TEST(TestSupervisedCorrespondenceExpectationStep, ComputeSupervisedCorrespondencePhi) {
    VectorXi X(10);
    X << 22, 49, 0, 2, 16, 35, 94, 3, 25, 10;
    VectorX<TypeParam> X_ratio = X.cast<TypeParam>() / X.sum();
    int  y = 0;

    MatrixX<TypeParam> phi = MatrixX<TypeParam>::Random(5, 10);
    VectorX<TypeParam> alpha = VectorX<TypeParam>::Constant(5, 0.1);
    MatrixX<TypeParam> beta = MatrixX<TypeParam>::Random(5, 10);
    MatrixX<TypeParam> eta = MatrixX<TypeParam>::Random(5, 3);
    VectorX<TypeParam> gamma = VectorX<TypeParam>::Constant(5, X.sum() / 5.0);
    VectorX<TypeParam> tau = VectorX<TypeParam>::Constant(10, 1. / 10.0);
    
    // Normalize beta phi and eta
    beta.array() -= beta.minCoeff() - 0.001;
    beta.array().rowwise() /= beta.colwise().sum().array();

    phi.array() -= phi.minCoeff() - 0.001;
    phi.array().rowwise() /= phi.colwise().sum().array();

    eta.array() -= eta.minCoeff() - 0.001;
    eta.array().rowwise() /= eta.colwise().sum().array();

    MatrixX<TypeParam> phi_unsupervised = phi;
    VectorX<TypeParam> gamma_unsupervised = gamma;
    TypeParam mu = 2.0;
    TypeParam portion = 0.1;

    TypeParam likelihood_baseline = e_step_utils::compute_supervised_correspondence_likelihood(
        X,
        y,
        alpha,
        beta,
        eta,
        phi,
        gamma,
        tau,
        mu,
        portion
    );

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
    TypeParam likelihood_unsupervised = e_step_utils::compute_supervised_correspondence_likelihood(
        X,
        y,
        alpha,
        beta,
        eta,
        phi_unsupervised,
        gamma,
        tau,
        mu,
        portion
    );

    for (int i=0; i<50; i++) {
        // Compute phi with correspondence supervised method
        e_step_utils::compute_supervised_correspondence_phi<TypeParam>(
            X,
            y,
            beta,
            eta,
            gamma,
            tau,
            phi
        );
        // Compute tau
        e_step_utils::compute_supervised_correspondence_tau<TypeParam>(
            X,
            y,
            eta,
            phi,
            tau
        );

        // Compute gamma
        e_step_utils::compute_gamma <TypeParam> (
            X,
            alpha,
            phi,
            gamma
        );
    }

    // Compute new likelihood
    TypeParam likelihood = e_step_utils::compute_supervised_correspondence_likelihood(
        X,
        y,
        alpha,
        beta,
        eta,
        phi,
        gamma,
        tau,
        mu,
        portion
    );

    EXPECT_GT(likelihood, likelihood_baseline);
    EXPECT_GT(likelihood, likelihood_unsupervised);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

