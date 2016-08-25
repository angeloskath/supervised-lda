#include <iostream>
#include <cmath>
#include <stdlib.h>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "test/utils.hpp"

#include "MultinomialLogisticRegression.hpp"
#include "GradientDescent.hpp"

using namespace Eigen;

template <typename T>
class TestMultinomialLogisticRegression : public ParameterizedTest<T> {};

TYPED_TEST_CASE(TestMultinomialLogisticRegression, ForFloatAndDouble);

/**
  * In this test we check if the gradient is correct by appling
  * a finite difference method.
  */
TYPED_TEST(TestMultinomialLogisticRegression, Gradient) {
    // Gradient checking should only be made with a double type
    if (is_float<TypeParam>::value) {
        return;
    }
    
    // eta is typically of size KxC, where K is the number of topics and C the
    // number of different classes.
    // Here we choose randomly for conviency K=10 and C=5
    MatrixX<TypeParam> eta = MatrixX<TypeParam>::Random(10, 5);
    // X is of size KxD, where D is the total number of documents.
    // In our case we have chosen D=15
    MatrixX<TypeParam> X = MatrixX<TypeParam>::Random(10, 1);
    // y is vector of size Dx1
    VectorXi y(1);
    for (int i=0; i<1; i++) {
        y(i) = rand() % (int)5; 
    }

    TypeParam L = 1;
    MultinomialLogisticRegression<TypeParam> mlr(X, y, L);

    // grad is the gradient according to the equation
    // implemented in MultinomialLogisticRegression.cpp 
    // gradient function
    // grad is of same size as eta, which is KxC
    MatrixX<TypeParam> grad(10, 5);
    
    // Calculate the gradients
    mlr.gradient(eta, grad);
    
    // Grad's approximation
    TypeParam grad_hat;
    TypeParam t = 1e-4;

    for (int i=0; i < eta.rows(); i++) {
        for (int j=0; j < eta.cols(); j++) {
            eta(i, j) += t;
            TypeParam ll1 = mlr.value(eta);
            eta(i, j) -= 2*t;
            TypeParam ll2 = mlr.value(eta);

            // Compute gradients approximation
            grad_hat = (ll1 - ll2) / (2 * t);

            auto absolute_error = std::abs(grad(i, j) - grad_hat);
            if (grad_hat != 0) {
                auto relative_error = absolute_error / std::abs(grad_hat);
                EXPECT_TRUE(
                    relative_error < 1e-4 ||
                    absolute_error < 1e-5
                ) << relative_error << " " << absolute_error;
            }
            else {
                EXPECT_LT(absolute_error, 1e-5);
            }
        }
    }
}

/**
  * In this test we check whether the minimizer works as expected on the Fisher
  * Iris dataset. We have isolated three classed from Fisher Iris and just two
  * features, from the total four.
  * We compare the results from our Minimizer with the corresponding results from
  * LogisticRegression's implementation of SKlearn with the same initial parameters.
  */
TYPED_TEST(TestMultinomialLogisticRegression, Minimizer) {
    
    // X contains the two features from Fisher Iris 
    MatrixX<TypeParam> Xin(150, 2);
    Xin << 5.1, 3.5, 4.9, 3., 4.7, 3.2, 4.6, 3.1, 5., 3.6, 5.4, 3.9, 4.6, 3.4, 5., 3.4,
    4.4, 2.9, 4.9, 3.1, 5.4, 3.7, 4.8, 3.4, 4.8, 3., 4.3, 3., 5.8, 4., 5.7, 4.4,
    5.4, 3.9, 5.1, 3.5, 5.7, 3.8, 5.1, 3.8, 5.4, 3.4, 5.1, 3.7, 4.6, 3.6, 5.1, 3.3,
    4.8, 3.4, 5., 3., 5., 3.4, 5.2, 3.5, 5.2, 3.4, 4.7, 3.2, 4.8, 3.1, 5.4, 3.4,
    5.2, 4.1, 5.5, 4.2, 4.9, 3.1, 5., 3.2, 5.5, 3.5, 4.9, 3.1, 4.4, 3., 5.1, 3.4,
    5., 3.5, 4.5, 2.3, 4.4, 3.2, 5., 3.5, 5.1, 3.8, 4.8, 3., 5.1, 3.8, 4.6, 3.2,
    5.3, 3.7, 5., 3.3, 7., 3.2, 6.4, 3.2, 6.9, 3.1, 5.5, 2.3, 6.5, 2.8, 5.7, 2.8,
    6.3, 3.3, 4.9, 2.4, 6.6, 2.9, 5.2, 2.7, 5., 2., 5.9, 3., 6., 2.2, 6.1, 2.9,
    5.6, 2.9, 6.7, 3.1, 5.6, 3., 5.8, 2.7, 6.2, 2.2, 5.6, 2.5, 5.9, 3.2, 6.1, 2.8,
    6.3, 2.5, 6.1, 2.8, 6.4, 2.9, 6.6, 3., 6.8, 2.8, 6.7, 3., 6., 2.9, 5.7, 2.6,
    5.5, 2.4, 5.5, 2.4, 5.8, 2.7, 6., 2.7, 5.4, 3., 6., 3.4, 6.7, 3.1, 6.3, 2.3,
    5.6, 3., 5.5, 2.5, 5.5, 2.6, 6.1, 3., 5.8, 2.6, 5., 2.3, 5.6, 2.7, 5.7, 3.,
    5.7, 2.9, 6.2, 2.9, 5.1, 2.5, 5.7, 2.8, 6.3, 3.3, 5.8, 2.7, 7.1, 3., 6.3, 2.9,
    6.5, 3., 7.6, 3., 4.9, 2.5, 7.3, 2.9, 6.7, 2.5, 7.2, 3.6, 6.5, 3.2, 6.4, 2.7,
    6.8, 3., 5.7, 2.5, 5.8, 2.8, 6.4, 3.2, 6.5, 3., 7.7, 3.8, 7.7, 2.6, 6., 2.2,
    6.9, 3.2, 5.6, 2.8, 7.7, 2.8, 6.3, 2.7, 6.7, 3.3, 7.2, 3.2, 6.2, 2.8, 6.1, 3.,
    6.4, 2.8, 7.2, 3., 7.4, 2.8, 7.9, 3.8, 6.4, 2.8, 6.3, 2.8, 6.1, 2.6, 7.7, 3.,
    6.3, 3.4, 6.4, 3.1, 6., 3., 6.9, 3.1, 6.7, 3.1, 6.9, 3.1, 5.8, 2.7, 6.8, 3.2,
    6.7, 3.3, 6.7, 3., 6.3, 2.5, 6.5, 3., 6.2, 3.4, 5.9, 3.;

    MatrixX<TypeParam> X(2, 150);
    X = Xin.transpose();

    // y is a vector of size Dx1, which contains class labels
    VectorXi y(150);
    y << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2;
    
    TypeParam regularization_penalty = 1e-2;
    MultinomialLogisticRegression<TypeParam> mlr(
        X,
        y,
        regularization_penalty
    );
    
    MatrixX<TypeParam> eta = MatrixX<TypeParam>::Zero(2, 3);
    
    GradientDescent<MultinomialLogisticRegression<TypeParam>, MatrixX<TypeParam>> minimizer(
        std::make_shared<ArmijoLineSearch<MultinomialLogisticRegression<TypeParam>, MatrixX<TypeParam>> >(),
        [](TypeParam value, TypeParam gradNorm, size_t iterations) {
            return iterations < 500;
        }
    );
    minimizer.minimize(mlr, eta);
    
    MatrixX<TypeParam> lbfgs_eta(2, 3);
    lbfgs_eta << -4.57679568, 2.03822995, 2.53856573, 8.12944469, -3.53562038, -4.59382431;

    VectorX<TypeParam> cosine_similarities = (lbfgs_eta.transpose() * eta).diagonal();
    cosine_similarities.array() /= eta.colwise().norm().array();
    cosine_similarities.array() /= lbfgs_eta.colwise().norm().array();

    const double pi = std::acos(-1);
    for (int i=0; i<cosine_similarities.rows(); i++) {
        EXPECT_LT(std::cos(2*pi/180), cosine_similarities[i]);
    }
}


TYPED_TEST(TestMultinomialLogisticRegression, MinimizerOverfitSmall) {
    MatrixX<TypeParam> X(2, 10);
    VectorXi y(10);

    X << 0.6097662 ,  0.53395565,  0.9499446 ,  0.67289898,  0.94173948,
         0.56675891,  0.80363783,  0.85303565,  0.15903886,  0.99518533,
         0.41655682,  0.29256121,  0.36103228,  0.29899503,  0.4957268 ,
         -0.04277318, -0.28038614, -0.12334621, -0.17497722,  0.1492248;
    y << 0, 0, 0, 0, 0, 1, 1, 1, 1, 1;

    MultinomialLogisticRegression<TypeParam> mlr(X, y, 0);
    MatrixX<TypeParam> eta = MatrixX<TypeParam>::Zero(2, 3);

    GradientDescent<MultinomialLogisticRegression<TypeParam>, MatrixX<TypeParam>> minimizer(
        std::make_shared<ArmijoLineSearch<MultinomialLogisticRegression<TypeParam>, MatrixX<TypeParam>> >(),
        [](TypeParam value, TypeParam gradNorm, size_t iterations) {
            return iterations < 5000;
        }
    );
    minimizer.minimize(mlr, eta);

    EXPECT_GT(1e-2, mlr.value(eta));
}
