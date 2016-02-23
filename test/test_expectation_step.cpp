
#include <cmath>
#include <vector>

#include <eigen3/Eigen/Core>
#include <gtest/gtest.h>

#include "test/utils.hpp"

#include "SupervisedLDA.hpp"


using namespace Eigen;


// T will be available as TypeParam in TYPED_TEST functions
template <typename T>
class TestExpectationStep : public ParameterizedTest<T> {};

TYPED_TEST_CASE(TestExpectationStep, ForFloatAndDouble);

TYPED_TEST(TestExpectationStep, ComputeH) {
    SupervisedLDA<TypeParam> lda(0);

    VectorXi X(10);
    X << 10, 9, 8, 7, 6, 5, 4, 3, 2, 1;

    MatrixX<TypeParam> eta = MatrixX<TypeParam>::Random(5, 3);
    MatrixX<TypeParam> phi = MatrixX<TypeParam>::Random(5, 10);
    phi.array() -= phi.minCoeff();
    phi = phi.array().rowwise() / phi.colwise().sum().array();
    MatrixX<TypeParam> h(5, 10);

    lda.compute_h(X, eta, phi, h);

    VectorX<TypeParam> hphi = (h.transpose() * phi).diagonal();

    for (int i=0; i<10; i++) {
        EXPECT_NEAR(hphi(0), hphi(i), 1e-6);
    }

    TypeParam hphi_actual = 0;
    for (int y=0; y<3; y++) {
        TypeParam p = 1;
        for (int n=0; n<10; n++) {
            TypeParam t = 0;
            for (int k=0; k<5; k++) {
                t += phi(k, n) * exp( X.cast<TypeParam>()(n)/X.sum() * eta(k, y) );
            }
            p *= t;
        }
        hphi_actual += p;
    }

    for (int i=0; i<10; i++) {
        EXPECT_NEAR(hphi_actual, hphi(i), 1e-6);
    }
}


TYPED_TEST(TestExpectationStep, ComputeLikelihood) {
    SupervisedLDA<TypeParam> lda(0);

    for (int i=0; i<100; i++) {
        VectorX<TypeParam> Xtmp = VectorX<TypeParam>::Random(10).array().abs() * 5;
        VectorXi X = Xtmp.template cast<int>();
        int y = 0;
        VectorX<TypeParam> alpha = VectorX<TypeParam>::Constant(5, 0.1);
        MatrixX<TypeParam> beta = MatrixX<TypeParam>::Random(5, 10);
        MatrixX<TypeParam> eta = MatrixX<TypeParam>::Random(5, 3);
        MatrixX<TypeParam> phi = MatrixX<TypeParam>::Constant(5, 10, 0.1);
        VectorX<TypeParam> gamma = VectorX<TypeParam>::Constant(5, X.sum() / 5.0);
        MatrixX<TypeParam> h(5, 10);

        // normalize beta, phi and initialize h
        phi.array() -= phi.minCoeff() - 0.001;
        phi.array().rowwise() /= phi.colwise().sum().array();
        beta.array() -= beta.minCoeff() - 0.001;
        beta.array().rowwise() /= beta.colwise().sum().array();
        lda.compute_h(X, eta, phi, h);

        TypeParam likelihood = lda.compute_likelihood(X, y, alpha, beta, eta, phi, gamma, h);

        ASSERT_FALSE(std::isnan(likelihood)) << phi.array().log();
        EXPECT_GT(0, likelihood);
    }
}


TYPED_TEST(TestExpectationStep, DocEStep) {
    SupervisedLDA<TypeParam> lda(0);

    VectorX<TypeParam> Xtmp = VectorX<TypeParam>::Random(10).array().abs() * 5;
    VectorXi X = Xtmp.template cast<int>();
    int y = 0;
    VectorX<TypeParam> alpha = VectorX<TypeParam>::Constant(5, 0.1);
    MatrixX<TypeParam> beta = MatrixX<TypeParam>::Random(5, 10);
    MatrixX<TypeParam> eta = MatrixX<TypeParam>::Random(5, 3);
    MatrixX<TypeParam> phi = MatrixX<TypeParam>::Random(5, 10);
    VectorX<TypeParam> gamma = VectorX<TypeParam>::Random(5).array().abs()*10;

    // normalize beta
    beta.array() -= beta.minCoeff() - 0.001;
    beta.array().rowwise() /= beta.colwise().sum().array();


    int fixed_point_iterations = 10;
    TypeParam convergence_tolerance = 0;
    std::vector<TypeParam> likelihoods(10);
    for (int i=0; i<10; i++) {
        likelihoods[i] = lda.doc_e_step(
            X,
            y,
            alpha,
            beta,
            eta,
            phi,
            gamma,
            fixed_point_iterations,
            i,
            convergence_tolerance
        );
    }
    for (int i=1; i<10; i++) {
        EXPECT_GT(likelihoods[i], likelihoods[i-1]);
    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
