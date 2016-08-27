#include <utility>

#include "MultinomialLogisticRegression.hpp"

namespace ldaplusplus {


template <typename Scalar>
MultinomialLogisticRegression<Scalar>::MultinomialLogisticRegression(
    const MatrixX &X,
    const VectorXi &y,
    Scalar L
) : X_(X), y_(y), L_(L) {
    
    // Total number of classes
    int C = y_.maxCoeff() + 1;
    // Allocate suitable memory
    Cy_ = VectorX::Zero(C);

    // Compute the total number of documents for every class
    for (int d=0; d< y_.rows(); d++) {
        Cy_(y_[d]) += 1;
    }

    Cy_ = y_.rows() / (Cy_.array() * C).array();
}

template <typename Scalar>
MultinomialLogisticRegression<Scalar>::MultinomialLogisticRegression(
    const MatrixX &X,
    const VectorXi &y,
    VectorX Cy,
    Scalar L
) : X_(X), y_(y), L_(L), Cy_(std::move(Cy))
{}


template <typename Scalar>
Scalar MultinomialLogisticRegression<Scalar>::value(const MatrixX &eta) const {
    Scalar likelihood = 0;
    VectorX t(eta.cols());

    // \eta_T E_q[Z]y - log(\sum_{y=1}^C exp(\eta^T E_q[Z]y))
    for (int d=0; d<y_.rows(); d++) {
        t = eta.transpose() * X_.col(d);
        likelihood += Cy_[y_[d]] * t[y_[d]];
        likelihood -= Cy_[y_[d]] * std::log(t.array().exp().sum());
        
        // This will be used in case we have overflow issues
        //Scalar a = t.maxCoeff();
        //likelihood -= a + std::log((t.array() - a).exp().sum());
    }
    
    // Add suitable normalization to the final likelihood
    auto norm = L_ * eta.squaredNorm() / 2;
    
    // we need to return the negative for maximization instead of minimization
    return - likelihood + norm;
}


template <typename Scalar>
void MultinomialLogisticRegression<Scalar>::gradient(const MatrixX &eta, Ref<MatrixX> grad) const {
    grad.fill(0);
    VectorX t(eta.cols());

    for (int d=0; d<y_.rows(); d++) {
        grad.col(y_[d]) -= Cy_[y_[d]] * X_.col(d);

        t = (eta.transpose() * X_.col(d)).array().exp();
        grad += Cy_[y_[d]] * (X_.col(d) * t.transpose()) / t.sum();
    }

    // Add suitable normalization for the gradient
    grad.array() += eta.array() * L_;
}


// template instatiation
template class MultinomialLogisticRegression<float>;
template class MultinomialLogisticRegression<double>;

}
