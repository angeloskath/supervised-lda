#include "UnsupervisedMStep.hpp"

template <typename Scalar>
void UnsupervisedMStep<Scalar>::m_step(
    std::shared_ptr<Parameters> model_parameters
) {
    // we maximized w.r.t \beta during each doc_m_step
    beta = b;
    beta = beta.array().colwise() / beta.array().rowwise().sum();

    return 0;
}

template <typename Scalar>
void UnsupervisedMStep<Scalar>::doc_m_step(
    const std::shared_ptr<Document> doc,
    const std::shared_ptr<Parameters> variational_parameters
    std::shared_ptr<Parameters> model_parameters
) {
    auto t1 = X.cast<Scalar>().transpose().array();
    auto t2 = phi.array().rowwise() * t1;

    b_.array() += t2;
    //expected_z_bar = t2.rowwise().sum() / X.sum();
}

// Template instantiation
template class UnsupervisedMStep<float>;
template class UnsupervisedMStep<double>;

