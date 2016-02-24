#include <cmath>
#include <stdlib.h>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "test/utils.hpp"

#include "MultinomialLogisticRegression.hpp"

using namespace Eigen;

template <typename T>
class TestMultinomialLogisticRegression : public ParameterizedTest<T> {};

TYPED_TEST_CASE(TestMultinomialLogisticRegression, ForFloatAndDouble);

TYPED_TEST(TestMultinomialLogisticRegression, gradient) {
    
    // eta is typically of size KxC, where K is the number of topics and C the
    // number of different classes.
    // Here we choose randomly for conviency K=10 and C=5
    MatrixX<TypeParam> eta = MatrixX<TypeParam>::Random(10, 5);
    // X is of size KxD, where D is the total number of documents.
    // In our case we have chosen D=15
    MatrixX<TypeParam> X = MatrixX<TypeParam>::Random(10, 10);
    // y is vector of size Dx1
    VectorXi y(10);
    for (int i=0; i<10; i++) {
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
                    relative_error < 1e-3 ||
                    absolute_error < 1e-4
                ) << relative_error << " " << absolute_error;
            }
            else {
                EXPECT_LT(absolute_error, 1e-5);
            }
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

