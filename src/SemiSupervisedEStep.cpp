#include "SemiSupervisedEStep.hpp"

template <typename Scalar>
SemiSupervisedEStep<Scalar>::SemiSupervisedEStep(
    std::shared_ptr<IEStep<Scalar> > supervised_step,
    std::shared_ptr<IEStep<Scalar> > unsupervised_step
) : supervised_step_(supervised_step),
    unsupervised_step_(unsupervised_step)
{
    event_forwarder_ = supervised_step_->get_event_dispatcher()->add_listener(
        [this](std::shared_ptr<Event> event) {
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
std::shared_ptr<Parameters> SemiSupervisedEStep<Scalar>::doc_e_step(
    const std::shared_ptr<Document> doc,
    const std::shared_ptr<Parameters> parameters
) {
    if (std::static_pointer_cast<ClassificationDocument>(doc)->get_class() < 0) {
        return unsupervised_step_->doc_e_step(doc, parameters);
    } else {
        return supervised_step_->doc_e_step(doc, parameters);
    }
}


// template instantiation
template class SemiSupervisedEStep<float>;
template class SemiSupervisedEStep<double>;
