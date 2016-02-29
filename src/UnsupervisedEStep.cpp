#include "UnsupervisedEStep.hpp"
#include "utils.hpp"

template <typename Scalar>
UnsupervisedEStep<Scalar>::UnsupervisedEStep(
    size_t e_step_iterations,
    size_t fixed_point_iterations,
    Scalar e_step_tolerance
) {
    e_step_iterations_ = e_step_iterations;
    fixed_point_iterations_ = fixed_point_iterations;
    e_step_tolerance_ = e_step_tolerance;
}

template <typename Scalar>
Scalar UnsupervisedEStep<Scalar>::doc_e_step(
    const VectorXi &X,
    int y,
    const VectorX &alpha,
    const MatrixX &beta,
    const MatrixX &eta,
    Ref<MatrixX> phi,
    Ref<VectorX> gamma
) {
    Scalar new_loglikelihood = 0.0;

    return new_loglikelihood;
}

template <typename Scalar>
Scalar UnsupervisedEStep<Scalar>::compute_likelihood(
    const VectorXi &X,
    const VectorX &alpha,
    const MatrixX &beta,
    const MatrixX &phi,
    const VectorX &gamma
) {
    auto cwise_digamma = CwiseDigamma<Scalar>();
    auto cwise_lgamma = CwiseLgamma<Scalar>();

    Scalar likelihood = 0;

    // \Psi(\gamma) - \Psi(\sum_j \gamma)
    VectorX t1 = gamma.unaryExpr(cwise_digamma).array() - digamma(gamma.sum());

    // E_q[log p(\theta | \alpha)]
    likelihood += ((alpha.array() - 1.0).matrix().transpose() * t1).value();
    likelihood += std::lgamma(alpha.sum()) - alpha.unaryExpr(cwise_lgamma).sum();

    // E_q[log p(z | \theta)]
    likelihood += (phi.transpose() * t1).sum();

    // E_q[log p(w | z, \beta)]
    auto phi_scaled = phi.array().rowwise() * X.cast<Scalar>().transpose().array();
    likelihood += (phi_scaled * beta.array().log()).sum();

    // H(q)
    likelihood += -((gamma.array() - 1).matrix().transpose() * t1).value();
    likelihood += -std::lgamma(gamma.sum()) + gamma.unaryExpr(cwise_lgamma).sum();
    likelihood += -(phi.array() * (phi.array() + 1e-44).log()).sum();
    
    return likelihood;
}

// Template instantiation
template class UnsupervisedEStep<float>;
template class UnsupervisedEStep<double>;

