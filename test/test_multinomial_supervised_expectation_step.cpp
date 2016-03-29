#include <Eigen/Core>
#include <gtest/gtest.h>

#include "test/utils.hpp"

#include "Document.hpp"
#include "MultinomialSupervisedEStep.hpp"
#include "Parameters.hpp"
#include "e_step_utils.hpp"

using namespace Eigen;

// T will be available as TypeParam in TYPED_TEST functions
template <typename T>
class TestMultinomialSupervisedExpectationStep : public ParameterizedTest<T> {};

TYPED_TEST_CASE(TestMultinomialSupervisedExpectationStep, ForFloatAndDouble);

TYPED_TEST(TestMultinomialSupervisedExpectationStep, ComputeSupervisedMultinomialLikelihood) {
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

        // normalize beta, phi, eta
        phi.array() -= phi.minCoeff() - 0.001;
        phi.array().rowwise() /= phi.colwise().sum().array();
        beta.array() -= beta.minCoeff() - 0.001;
        beta.array().rowwise() /= beta.colwise().sum().array();
        eta.array() -= eta.minCoeff() - 0.001;
        eta.array().rowwise() /= eta.colwise().sum().array();

        TypeParam mu = 2.0;

        TypeParam likelihood = e_step_utils::compute_supervised_multinomial_likelihood(
            X,
            y,
            alpha,
            beta,
            eta,
            phi,
            gamma,
            mu
        );

        ASSERT_FALSE(std::isnan(likelihood)) << phi.array().log();
        EXPECT_GT(0, likelihood);
    }
}

TYPED_TEST(TestMultinomialSupervisedExpectationStep, ComputeSupervisedMultinomialPhi) {
    VectorXi X(10);
    X << 22, 49, 0, 2, 16, 35, 94, 3, 25, 10;
    VectorX<TypeParam> X_ratio = X.cast<TypeParam>() / X.sum();
    int  y = 0;

    MatrixX<TypeParam> phi = MatrixX<TypeParam>::Random(5, 10);
    VectorX<TypeParam> alpha = VectorX<TypeParam>::Constant(5, 0.1);
    MatrixX<TypeParam> beta = MatrixX<TypeParam>::Random(5, 10);
    MatrixX<TypeParam> eta = MatrixX<TypeParam>::Random(5, 3);
    VectorX<TypeParam> gamma = VectorX<TypeParam>::Constant(5, X.sum() / 5.0);
    
    // Normalize beta phi and eta
    beta.array() -= beta.minCoeff() - 0.001;
    beta.array().rowwise() /= beta.colwise().sum().array();

    phi.array() -= phi.minCoeff() - 0.001;
    phi.array().rowwise() /= phi.colwise().sum().array();

    eta.array() -= eta.minCoeff() - 0.001;
    eta.array().rowwise() /= eta.colwise().sum().array();
    
    MatrixX<TypeParam> phi_unsupervised = phi;
    MatrixX<TypeParam> phi_supervised = phi;
    MatrixX<TypeParam> phi_supervised_approximation = phi;
    MatrixX<TypeParam> phi_old = phi;

    TypeParam mu = 2.0;

    TypeParam likelihood_baseline = e_step_utils::compute_supervised_multinomial_likelihood(
        X,
        y,
        alpha,
        beta,
        eta,
        phi,
        gamma,
        mu
    );

    e_step_utils::compute_supervised_multinomial_phi(
        X,
        y,
        beta,
        eta,
        gamma,
        phi
    );

    TypeParam likelihood = e_step_utils::compute_supervised_multinomial_likelihood(
        X,
        y,
        alpha,
        beta,
        eta,
        phi,
        gamma,
        mu
    );
    
    EXPECT_GT(likelihood, likelihood_baseline);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
