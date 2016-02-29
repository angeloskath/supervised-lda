#include "UnsupervisedMStep.hpp"

template <typename Scalar>
Scalar UnsupervisedMStep<Scalar>::m_step(
    const MatrixX &expected_z_bar,
    const MatrixX &b,
    const VectorXi &y,
    Ref<MatrixX> beta,
    Ref<MatrixX> eta
) {
    // we maximized w.r.t \beta during each doc_m_step
    beta = b;
    beta = beta.array().colwise() / beta.array().rowwise().sum();
}

template <typename Scalar>
void UnsupervisedMStep<Scalar>::doc_m_step(
   const VectorXi &X,
   const MatrixX &phi,
   Ref<MatrixX> b,
   Ref<VectorX> expected_z_bar
) {
    auto t1 = X.cast<Scalar>().transpose().array();
    auto t2 = phi.array().rowwise() * t1;

    b.array() += t2;
    expected_z_bar = t2.rowwise().sum() / X.sum();
}


template <typename Scalar>
int UnsupervisedMStep<Scalar>::get_id() {
    return IMStep<Scalar>::BatchUnsupervised;
}
template <typename Scalar>
std::vector<Scalar> UnsupervisedMStep<Scalar>::get_parameters() {
    return {};
}
template <typename Scalar>
void UnsupervisedMStep<Scalar>::set_parameters(std::vector<Scalar> parameters) {
}
