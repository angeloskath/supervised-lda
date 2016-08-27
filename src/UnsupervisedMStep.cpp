#include "UnsupervisedMStep.hpp"

namespace ldaplusplus {


template <typename Scalar>
void UnsupervisedMStep<Scalar>::m_step(
    std::shared_ptr<Parameters> parameters
) {
    // we maximized w.r.t \beta during each doc_m_step
    std::static_pointer_cast<ModelParameters<Scalar> >(parameters)->beta = 
        b_.array().colwise() / b_.array().rowwise().sum();

    b_.fill(0);
}

template <typename Scalar>
void UnsupervisedMStep<Scalar>::doc_m_step(
    const std::shared_ptr<Document> doc,
    const std::shared_ptr<Parameters> v_parameters,
    std::shared_ptr<Parameters> m_parameters
) {
    // Words form Document doc
    const VectorXi &X = doc->get_words();
    auto t1 = X.cast<Scalar>().transpose().array();
    
    // Cast Parameters to VariationalParameters in order to have access to phi
    const MatrixX &phi = std::static_pointer_cast<VariationalParameters<Scalar> >(v_parameters)->phi;
    auto t2 = phi.array().rowwise() * t1;
    
    // Check if b_ is accessed and allocate suitable amound of memory
    if (b_.rows() == 0)
        b_ = MatrixX::Zero(phi.rows(), phi.cols());

    b_.array() += t2;
}

// Template instantiation
template class UnsupervisedMStep<float>;
template class UnsupervisedMStep<double>;


}
