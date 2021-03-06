
#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "test/utils.hpp"

#include "ldaplusplus/Document.hpp"
#include "ldaplusplus/em/FastSupervisedEStep.hpp"
#include "ldaplusplus/Parameters.hpp"
#include "ldaplusplus/em/SupervisedEStep.hpp"
#include "ldaplusplus/e_step_utils.hpp"

using namespace Eigen;
using namespace ldaplusplus;


// T will be available as TypeParam in TYPED_TEST functions
template <typename T>
class TestExpectationStep : public ParameterizedTest<T> {};

TYPED_TEST_CASE(TestExpectationStep, ForFloatAndDouble);

TYPED_TEST(TestExpectationStep, ComputeH) {
    VectorXi X(10);
    X << 10, 9, 8, 7, 6, 5, 4, 3, 2, 1;
    VectorX<TypeParam> X_ratio = X.cast<TypeParam>() / X.sum();

    MatrixX<TypeParam> eta = MatrixX<TypeParam>::Random(5, 3);
    MatrixX<TypeParam> phi = MatrixX<TypeParam>::Random(5, 10);
    phi.array() -= phi.minCoeff();
    phi = phi.array().rowwise() / phi.colwise().sum().array();
    VectorX<TypeParam> h(5);

    e_step_utils::compute_h<TypeParam>(X, X_ratio, eta, phi, h);

    VectorX<TypeParam> hphi = h.transpose() * phi;

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

    EXPECT_NEAR(hphi_actual, hphi(9), 1e-2);
}


TYPED_TEST(TestExpectationStep, ComputeLikelihood) {
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

        // normalize beta, phi
        phi.array() -= phi.minCoeff() - 0.001;
        phi.array().rowwise() /= phi.colwise().sum().array();
        beta.array() -= beta.minCoeff() - 0.001;
        beta.array().rowwise() /= beta.colwise().sum().array();

        TypeParam likelihood = e_step_utils::compute_supervised_likelihood(
            X,
            y,
            alpha,
            beta,
            eta,
            phi,
            gamma
        );

        ASSERT_FALSE(std::isnan(likelihood)) << phi.array().log();
        EXPECT_GT(0, likelihood);
    }
}


TYPED_TEST(TestExpectationStep, DocEStep) {
    return;

    VectorX<TypeParam> Xtmp = VectorX<TypeParam>::Random(10).array().abs() * 5;
    VectorXi X = Xtmp.template cast<int>();
    int y = 0;

    auto doc = std::make_shared<corpus::ClassificationDecorator>(
        std::make_shared<corpus::EigenDocument>(X),
        y
    );

    VectorX<TypeParam> alpha = VectorX<TypeParam>::Constant(5, 0.2);
    MatrixX<TypeParam> beta = MatrixX<TypeParam>::Random(5, 10);
    MatrixX<TypeParam> eta = MatrixX<TypeParam>::Zero(5, 3);

    // normalize beta
    beta.array() -= beta.minCoeff() - 0.001;
    beta.array().rowwise() /= beta.colwise().sum().array();

    auto model = std::make_shared<parameters::SupervisedModelParameters<TypeParam> >(
        alpha,
        beta,
        eta
    );

    size_t fixed_point_iterations = 5;
    TypeParam convergence_tolerance = -10;
    std::vector<TypeParam> likelihoods(10);
    for (int i=0; i<10; i++) {
        em::SupervisedEStep<TypeParam> e_step(
            i,
            convergence_tolerance,
            fixed_point_iterations
        );
        auto vp = std::static_pointer_cast<parameters::VariationalParameters<TypeParam> >(
            e_step.doc_e_step(
                doc,
                model
            )
        );

        likelihoods[i] = e_step_utils::compute_supervised_likelihood<TypeParam>(
            doc->get_words(),
            doc->get_class(),
            model->alpha,
            model->beta,
            model->eta,
            vp->phi,
            vp->gamma
        );
    }
    for (int i=1; i<10; i++) {
        EXPECT_GT(likelihoods[i], likelihoods[i-1]);
    }
}


TYPED_TEST(TestExpectationStep, FastDocEStep) {
    return;

    VectorX<TypeParam> Xtmp = VectorX<TypeParam>::Random(10).array().abs() * 5;
    VectorXi X = Xtmp.template cast<int>();
    int y = 0;

    auto doc = std::make_shared<corpus::ClassificationDecorator>(
        std::make_shared<corpus::EigenDocument>(X),
        y
    );

    VectorX<TypeParam> alpha = VectorX<TypeParam>::Constant(5, 0.1);
    MatrixX<TypeParam> beta = MatrixX<TypeParam>::Random(5, 10);
    MatrixX<TypeParam> eta = MatrixX<TypeParam>::Random(5, 3);

    // normalize beta
    beta.array() -= beta.minCoeff() - 0.001;
    beta.array().rowwise() /= beta.colwise().sum().array();

    auto model = std::make_shared<parameters::SupervisedModelParameters<TypeParam> >(
        alpha,
        beta,
        eta
    );

    VectorX<TypeParam> h(beta.rows());

    int fixed_point_iterations = 10;
    TypeParam convergence_tolerance = 0;
    std::vector<TypeParam> likelihoods(10);
    for (int i=0; i<10; i++) {
        em::FastSupervisedEStep<TypeParam> e_step(
            i,
            convergence_tolerance,
            fixed_point_iterations
        );
        auto vp = std::static_pointer_cast<parameters::VariationalParameters<TypeParam> >(
            e_step.doc_e_step(
                doc,
                model
            )
        );
        likelihoods[i] = e_step_utils::compute_supervised_likelihood<TypeParam>(
            doc->get_words(),
            doc->get_class(),
            model->alpha,
            model->beta,
            model->eta,
            vp->phi,
            vp->gamma,
            h
        );
    }
    for (int i=1; i<10; i++) {
        EXPECT_GT(likelihoods[i], likelihoods[i-1]);
    }
}
