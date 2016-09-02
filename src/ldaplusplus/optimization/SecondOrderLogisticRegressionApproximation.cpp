#include <utility>

#include "ldaplusplus/optimization/SecondOrderLogisticRegressionApproximation.hpp"

namespace ldaplusplus {
namespace optimization {


template <typename Scalar>
SecondOrderLogisticRegressionApproximation<Scalar>::SecondOrderLogisticRegressionApproximation(
    const MatrixX &X,
    const std::vector<MatrixX> &X_var,
    const Eigen::VectorXi &y,
    Scalar L
) : X_(X), X_var_(X_var), y_(y), L_(L) {
    
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
SecondOrderLogisticRegressionApproximation<Scalar>::SecondOrderLogisticRegressionApproximation(
    const MatrixX &X,
    const std::vector<MatrixX> &X_var,
    const Eigen::VectorXi &y,
    VectorX Cy,
    Scalar L
) : X_(X), X_var_(X_var), y_(y), L_(L), Cy_(std::move(Cy))
{}


template <typename Scalar>
Scalar SecondOrderLogisticRegressionApproximation<Scalar>::value(const MatrixX &eta) const {
    Scalar likelihood = 0;
    VectorX t(eta.cols());

    // \eta_T E_q[Z]y - log(\sum_{y=1}^C exp(\eta^T E_q[Z]y))(1 + \frac{1}{2} \eta^T V_q[z] \eta)
    for (int d=0; d<y_.rows(); d++) {
        t = eta.transpose() * X_.col(d);
        likelihood += Cy_[y_[d]] * t[y_[d]];
        likelihood -= Cy_[y_[d]] * std::log(
            (t.array().exp() * (1. + 0.5 * (eta.transpose() * X_var_[d] * eta).diagonal().array())).sum()
        );
    }
    
    // Add suitable normalization to the final likelihood
    auto norm = L_ * eta.squaredNorm() / 2;
    
    // we need to return the negative for maximization instead of minimization
    return - likelihood + norm;
}


template <typename Scalar>
void SecondOrderLogisticRegressionApproximation<Scalar>::gradient(const MatrixX &eta, Eigen::Ref<MatrixX> grad) const {
    grad.fill(0);
    VectorX eta_Ez(eta.cols());
    VectorX eta_Vz_eta(eta.cols());
    MatrixX eta_Vz(eta.rows(), eta.cols());
    MatrixX grad_d(eta.rows(), eta.cols());
    Scalar normalizer = 0;

    for (int d=0; d<y_.rows(); d++) {
        grad.col(y_[d]) -= Cy_[y_[d]] * X_.col(d);

        // compute everything needed to assemble the gradients
        eta_Ez = (eta.transpose() * X_.col(d)).array().exp();
        eta_Vz = eta.transpose() * X_var_[d];
        eta_Vz_eta = (eta_Vz * eta).diagonal();
        eta_Vz += eta.transpose() * X_var_[d].transpose();
        eta_Vz /= 2;
        normalizer = eta_Ez.sum() + 0.5 * (eta_Ez.transpose() * eta_Vz_eta).value();

        // exp(\eta_y^T E_q[z]) E_q[z] (1 + \frac{1}{2} \eta_y^T V_q[z] \eta_y)
        grad_d = X_.col(d) * (eta_Ez.array() * (1 + 0.5 * eta_Vz_eta.array())).matrix().transpose();

        // exp(\eta_y^T E_q[z]) ( \eta_y^T (V_q[z] + V_q[z]^T) )
        grad_d.array() += eta_Vz.transpose().array().rowwise() * eta_Ez.array().transpose();

        grad += Cy_[y_[d]] * (1./normalizer) * grad_d;
    }

    // Add suitable normalization for the gradient
    grad.array() += eta.array() * L_;
}


// template instatiation
template class SecondOrderLogisticRegressionApproximation<float>;
template class SecondOrderLogisticRegressionApproximation<double>;

}  // namespace optimization
}  // namespace ldaplusplus
