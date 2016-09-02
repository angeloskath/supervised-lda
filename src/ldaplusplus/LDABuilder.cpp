#include "ldaplusplus/LDABuilder.hpp"

namespace ldaplusplus {

template <typename Scalar>
LDABuilder<Scalar>::LDABuilder()
    : iterations_(20),
      workers_(std::thread::hardware_concurrency()),
      e_step_(std::make_shared<em::UnsupervisedEStep<Scalar> >()),
      m_step_(std::make_shared<em::UnsupervisedMStep<Scalar> >()),
      model_parameters_(
        std::make_shared<parameters::SupervisedModelParameters<Scalar> >()
      )
{}

template <typename Scalar>
LDABuilder<Scalar> & LDABuilder<Scalar>::set_iterations(size_t iterations) {
    iterations_ = iterations;

    return *this;
}
template <typename Scalar>
LDABuilder<Scalar> & LDABuilder<Scalar>::set_workers(size_t workers) {
    workers_ = workers;

    return *this;
}

template <typename Scalar>
std::shared_ptr<em::IEStep<Scalar> > LDABuilder<Scalar>::get_classic_e_step(
    size_t e_step_iterations,
    Scalar e_step_tolerance
) {
    return std::make_shared<em::UnsupervisedEStep<Scalar> >(
        e_step_iterations,
        e_step_tolerance
    );
}

template <typename Scalar>
std::shared_ptr<em::IEStep<Scalar> > LDABuilder<Scalar>::get_fast_classic_e_step(
    size_t e_step_iterations,
    Scalar e_step_tolerance
) {
    return std::make_shared<em::FastUnsupervisedEStep<Scalar> >(
        e_step_iterations,
        e_step_tolerance
    );
}

template <typename Scalar>
std::shared_ptr<em::IEStep<Scalar> > LDABuilder<Scalar>::get_supervised_e_step(
    size_t e_step_iterations,
    Scalar e_step_tolerance,
    size_t fixed_point_iterations
) {
    return std::make_shared<em::SupervisedEStep<Scalar> >(
        e_step_iterations,
        e_step_tolerance,
        fixed_point_iterations
    );
}

template <typename Scalar>
std::shared_ptr<em::IEStep<Scalar> > LDABuilder<Scalar>::get_fast_supervised_e_step(
    size_t e_step_iterations,
    Scalar e_step_tolerance,
    Scalar C,
    typename em::ApproximatedSupervisedEStep<Scalar>::CWeightType weight_type,
    bool compute_likelihood
) {
    return std::make_shared<em::ApproximatedSupervisedEStep<Scalar> >(
        e_step_iterations,
        e_step_tolerance,
        C,
        weight_type,
        compute_likelihood
    );
}

template <typename Scalar>
std::shared_ptr<em::IEStep<Scalar> > LDABuilder<Scalar>::get_semi_supervised_e_step(
    std::shared_ptr<em::IEStep<Scalar> > supervised_step,
    std::shared_ptr<em::IEStep<Scalar> > unsupervised_step
) {
    if (supervised_step == nullptr) {
        supervised_step = get_fast_supervised_e_step();
    }
    if (unsupervised_step == nullptr) {
        unsupervised_step = get_fast_classic_e_step();
    }

    return std::make_shared<em::SemiSupervisedEStep<Scalar> >(
        supervised_step,
        unsupervised_step
    );
}

template <typename Scalar>
std::shared_ptr<em::IEStep<Scalar> > LDABuilder<Scalar>::get_multinomial_supervised_e_step(
    size_t e_step_iterations,
    Scalar e_step_tolerance,
    Scalar mu,
    Scalar eta_weight
) {
    return std::make_shared<em::MultinomialSupervisedEStep<Scalar> >(
        e_step_iterations,
        e_step_tolerance,
        mu,
        eta_weight
    );
}

template <typename Scalar>
std::shared_ptr<em::IEStep<Scalar> > LDABuilder<Scalar>::get_correspondence_supervised_e_step(
    size_t e_step_iterations,
    Scalar e_step_tolerance,
    Scalar mu
) {
    return std::make_shared<em::CorrespondenceSupervisedEStep<Scalar> >(
        e_step_iterations,
        e_step_tolerance,
        mu
    );
}

template <typename Scalar>
std::shared_ptr<em::IMStep<Scalar> > LDABuilder<Scalar>::get_classic_m_step() {
    return std::make_shared<em::UnsupervisedMStep<Scalar> >();
}

template <typename Scalar>
std::shared_ptr<em::IMStep<Scalar> > LDABuilder<Scalar>::get_supervised_m_step(
    size_t m_step_iterations,
    Scalar m_step_tolerance,
    Scalar regularization_penalty
) {
    return std::make_shared<em::SupervisedMStep<Scalar> >(
        m_step_iterations,
        m_step_tolerance,
        regularization_penalty
    );
}

template <typename Scalar>
std::shared_ptr<em::IMStep<Scalar> > LDABuilder<Scalar>::get_second_order_supervised_m_step(
    size_t m_step_iterations,
    Scalar m_step_tolerance,
    Scalar regularization_penalty
) {
    return std::make_shared<em::SecondOrderSupervisedMStep<Scalar> >(
        m_step_iterations,
        m_step_tolerance,
        regularization_penalty
    );
}

template <typename Scalar>
std::shared_ptr<em::IMStep<Scalar> > LDABuilder<Scalar>::get_supervised_online_m_step(
    size_t num_classes,
    Scalar regularization_penalty,
    size_t minibatch_size,
    Scalar eta_momentum,
    Scalar eta_learning_rate,
    Scalar beta_weight
) {
    return std::make_shared<em::OnlineSupervisedMStep<Scalar> >(
        num_classes,
        regularization_penalty,
        minibatch_size,
        eta_momentum,
        eta_learning_rate,
        beta_weight
    );
}

template <typename Scalar>
std::shared_ptr<em::IMStep<Scalar> > LDABuilder<Scalar>::get_supervised_online_m_step(
    std::vector<Scalar> class_weights,
    Scalar regularization_penalty,
    size_t minibatch_size,
    Scalar eta_momentum,
    Scalar eta_learning_rate,
    Scalar beta_weight
) {
    // Construct an Eigen Matrix and copy the weights
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weights(class_weights.size());
    for (size_t i=0; i<class_weights.size(); i++) {
        weights[i] = class_weights[i];
    }

    return get_supervised_online_m_step(
        weights,
        regularization_penalty,
        minibatch_size,
        eta_momentum,
        eta_learning_rate,
        beta_weight
    );
}

template <typename Scalar>
std::shared_ptr<em::IMStep<Scalar> > LDABuilder<Scalar>::get_supervised_online_m_step(
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> class_weights,
    Scalar regularization_penalty,
    size_t minibatch_size,
    Scalar eta_momentum,
    Scalar eta_learning_rate,
    Scalar beta_weight
) {
    return std::make_shared<em::OnlineSupervisedMStep<Scalar> >(
        class_weights,
        regularization_penalty,
        minibatch_size,
        eta_momentum,
        eta_learning_rate,
        beta_weight
    );
}

template <typename Scalar>
std::shared_ptr<em::IMStep<Scalar> > LDABuilder<Scalar>::get_semi_supervised_m_step(
    size_t m_step_iterations,
    Scalar m_step_tolerance,
    Scalar regularization_penalty
) {
    return std::make_shared<em::SemiSupervisedMStep<Scalar> >(
        m_step_iterations,
        m_step_tolerance,
        regularization_penalty
    );
}

template <typename Scalar>
std::shared_ptr<em::IMStep<Scalar> > LDABuilder<Scalar>::get_multinomial_supervised_m_step(
    Scalar mu
) {
    return std::make_shared<em::MultinomialSupervisedMStep<Scalar> >(mu);
}

template <typename Scalar>
std::shared_ptr<em::IMStep<Scalar> > LDABuilder<Scalar>::get_correspondence_supervised_m_step(
    Scalar mu
) {
    return std::make_shared<em::CorrespondenceSupervisedMStep<Scalar> >(mu);
}

// Just the template instantiations all the rest is defined in the headers.
template class LDABuilder<float>;
template class LDABuilder<double>;

}
