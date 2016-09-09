#include "ldaplusplus/em/SemiSupervisedMStep.hpp"

namespace ldaplusplus {
namespace em {


template <typename Scalar>
void SemiSupervisedMStep<Scalar>::doc_m_step(
    const std::shared_ptr<corpus::Document> doc,
    const std::shared_ptr<parameters::Parameters> v_parameters,
    std::shared_ptr<parameters::Parameters> m_parameters
) {
    if (std::static_pointer_cast<corpus::ClassificationDocument>(doc)->get_class() < 0) {
        UnsupervisedMStep<Scalar>::doc_m_step(doc, v_parameters, m_parameters);
    } else {
        FastSupervisedMStep<Scalar>::doc_m_step(doc, v_parameters, m_parameters);
    }
}


// template instantiation
template class SemiSupervisedMStep<float>;
template class SemiSupervisedMStep<double>;


}  // namespace em
}  // namespace ldaplusplus
