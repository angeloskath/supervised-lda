#include "MultinomialLogisticRegression.hpp"

template <typename Scalar>
MultinomialLogisticRegression<Scalar>::MultinomialLogisticRegression(
    const MatrixX &X,
    const VectorXi &y
) : X_(X), y_(y) {}

template <typename Scalar>
Scalar MultinomialLogisticRegression<Scalar>::value(const MatrixX &eta) {
    Scalar likelihood = 0;
    VectorX t(eta.cols());

    for (int d=0; d<y_.rows(); d++) {
        t = eta.transpose() * X_.col(d);
        likelihood += t[y_[d]] - std::log(t.array().exp().sum());
    }

    return likelihood;
}

template <typename Scalar>
void MultinomialLogisticRegression<Scalar>::gradient(const MatrixX &eta, MatrixX &grad) {
    grad.fill(0);
    VectorX t(eta.cols());

    for (int d=0; d<y_.rows(); d++) {
        grad.col(y_[d]) += X_.col(d);

        t = (eta.transpose() * X_.col(d)).array().exp();
        grad -= (X_.col(d) * t.transpose()) / t.sum();
    }
}

// template instatiation
template class MultinomialLogisticRegression<float>;
template class MultinomialLogisticRegression<double>;
