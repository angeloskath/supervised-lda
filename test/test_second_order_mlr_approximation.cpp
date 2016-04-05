#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "test/utils.hpp"

#include "MultinomialLogisticRegression.hpp"
#include "GradientDescent.hpp"
#include "SecondOrderLogisticRegressionApproximation.hpp"

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
    std::vector<MatrixX<TypeParam> > X_var = {MatrixX<TypeParam>::Random(10, 10).array().abs()};

    TypeParam L = 1;
    SecondOrderLogisticRegressionApproximation<TypeParam> mlr(X, X_var, y, L);

    // grad is the gradient according to the equation
    // implemented in MultinomialLogisticRegression.cpp 
    // gradient function
    // grad is of same size as eta, which is KxC
    MatrixX<TypeParam> grad(10, 5);

    // Calculate the gradients
    mlr.gradient(eta, grad);

    // Grad's approximation
    TypeParam grad_hat;
    TypeParam t = 1e-6;

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


TYPED_TEST(TestMultinomialLogisticRegression, MinimizerOverfitSmall) {
    MatrixX<TypeParam> X(2, 10);
    VectorXi y(10);

    X << 0.6097662 ,  0.53395565,  0.9499446 ,  0.67289898,  0.94173948,
         0.56675891,  0.80363783,  0.85303565,  0.15903886,  0.99518533,
         0.41655682,  0.29256121,  0.36103228,  0.29899503,  0.4957268 ,
         -0.04277318, -0.28038614, -0.12334621, -0.17497722,  0.1492248;
    y << 0, 0, 0, 0, 0, 1, 1, 1, 1, 1;
    std::vector<MatrixX<TypeParam> > X_var;
    for (int i=0; i<10; i++) {
        //X_var.push_back(MatrixX<TypeParam>::Random(2, 2).array().abs() * 0.01);
        //X_var.push_back(MatrixX<TypeParam>::Zero(2, 2));
        VectorX<TypeParam> a = VectorX<TypeParam>::Random(2).array() * 0.01;
        X_var.push_back(
            a * a.transpose()
        );
    }

    SecondOrderLogisticRegressionApproximation<TypeParam> mlr(X, X_var, y, 0);
    MatrixX<TypeParam> eta = MatrixX<TypeParam>::Zero(2, 3);

    GradientDescent<SecondOrderLogisticRegressionApproximation<TypeParam>, MatrixX<TypeParam>> minimizer(
        std::make_shared<
            ArmijoLineSearch<
                SecondOrderLogisticRegressionApproximation<TypeParam>,
                MatrixX<TypeParam>
            >
        >(),
        [](TypeParam value, TypeParam gradNorm, size_t iterations) {
            return iterations < 5000;
        }
    );
    minimizer.minimize(mlr, eta);

    EXPECT_GT(0.1, mlr.value(eta));
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

