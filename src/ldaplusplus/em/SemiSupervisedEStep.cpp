#include "ldaplusplus/em/SemiSupervisedEStep.hpp"

namespace ldaplusplus {

using em::SemiSupervisedEStep;

template <typename Scalar>
SemiSupervisedEStep<Scalar>::SemiSupervisedEStep(
    std::shared_ptr<EStepInterface<Scalar> > supervised_step,
    std::shared_ptr<EStepInterface<Scalar> > unsupervised_step
) : supervised_step_(supervised_step),
    unsupervised_step_(unsupervised_step)
{
    event_forwarder_ = supervised_step_->get_event_dispatcher()->add_listener(
        [this](std::shared_ptr<events::Event> event) {
            this->get_event_dispatcher()->dispatch(event);
        }
    );
    unsupervised_step_->get_event_dispatcher()->add_listener(event_forwarder_);
}


template <typename Scalar>
SemiSupervisedEStep<Scalar>::~SemiSupervisedEStep() {
    supervised_step_->get_event_dispatcher()->remove_listener(event_forwarder_);
    unsupervised_step_->get_event_dispatcher()->remove_listener(event_forwarder_);
}


template <typename Scalar>
std::shared_ptr<parameters::Parameters> SemiSupervisedEStep<Scalar>::doc_e_step(
    const std::shared_ptr<corpus::Document> doc,
    const std::shared_ptr<parameters::Parameters> parameters
) {
    if (std::static_pointer_cast<corpus::ClassificationDocument>(doc)->get_class() < 0) {
        return unsupervised_step_->doc_e_step(doc, parameters);
    } else {
        return supervised_step_->doc_e_step(doc, parameters);
    }
}


template <typename Scalar>
void SemiSupervisedEStep<Scalar>::e_step() {
    supervised_step_->e_step();
    unsupervised_step_->e_step();
}


// template instantiation
template class SemiSupervisedEStep<float>;
template class SemiSupervisedEStep<double>;

}
