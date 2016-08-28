#include "ldaplusplus/em/SemiSupervisedMStep.hpp"

namespace ldaplusplus {

using em::SemiSupervisedMStep;


template <typename Scalar>
void SemiSupervisedMStep<Scalar>::doc_m_step(
    const std::shared_ptr<corpus::Document> doc,
    const std::shared_ptr<Parameters> v_parameters,
    std::shared_ptr<Parameters> m_parameters
) {
    if (std::static_pointer_cast<corpus::ClassificationDocument>(doc)->get_class() < 0) {
        UnsupervisedMStep<Scalar>::doc_m_step(doc, v_parameters, m_parameters);
    } else {
        SupervisedMStep<Scalar>::doc_m_step(doc, v_parameters, m_parameters);
    }
}


// template instantiation
template class SemiSupervisedMStep<float>;
template class SemiSupervisedMStep<double>;

}
