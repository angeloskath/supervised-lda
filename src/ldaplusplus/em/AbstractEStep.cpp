#include "ldaplusplus/em/AbstractEStep.hpp"

namespace ldaplusplus {
namespace em {


template <typename Scalar>
AbstractEStep<Scalar>::AbstractEStep(int random_state)
    : random_(random_state)
{}

template <typename Scalar>
bool AbstractEStep<Scalar>::converged(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> & gamma_old,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> & gamma,
    Scalar tolerance
) {
    Scalar mean_change = (gamma_old - gamma).array().abs().sum() / gamma.rows();

    return mean_change < tolerance;
}

// Template instantiation
template class AbstractEStep<float>;
template class AbstractEStep<double>;


}  // namespace em
}  // namespace ldaplusplus
