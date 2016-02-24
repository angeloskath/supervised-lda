#include "MultinomialLogisticRegression.hpp"


template <typename Scalar>
MultinomialLogisticRegression<Scalar>::MultinomialLogisticRegression(
    const MatrixX &X,
    const VectorXi &y,
    Scalar L
) : X_(X), y_(y), L_(L) {}


template <typename Scalar>
Scalar MultinomialLogisticRegression<Scalar>::value(const MatrixX &eta) const {
    Scalar likelihood = 0;
    VectorX t(eta.cols());

    // \eta_T E_q[Z]y - log(\sum_{y=1}^C exp(\eta^T E_q[Z]y))
    for (int d=0; d<y_.rows(); d++) {
        t = eta.transpose() * X_.col(d);
        likelihood += t[y_[d]];
        likelihood -= std::log(t.array().exp().sum());
        
        // This will be used in case we have overflow issues
        //Scalar a = t.maxCoeff();
        //likelihood -= a + std::log((t.array() - a).exp().sum());
    }
    
    // Add suitable normalization to the final likelihood
    auto norm = L_ * eta.squaredNorm() / 2;

    // we need to return the negative for maximization instead of minimization
    return - likelihood - norm;
}


template <typename Scalar>
void MultinomialLogisticRegression<Scalar>::gradient(const MatrixX &eta, MatrixX &grad) const {
    grad.fill(0);
    VectorX t(eta.cols());

    for (int d=0; d<y_.rows(); d++) {
        grad.col(y_[d]) -= X_.col(d);

        t = (eta.transpose() * X_.col(d)).array().exp();
        grad += (X_.col(d) * t.transpose()) / t.sum();
    }

    // Add suitable normalization for the gradient
    grad.array() -= eta.array() * L_;
}


// template instatiation
template class MultinomialLogisticRegression<float>;
template class MultinomialLogisticRegression<double>;
