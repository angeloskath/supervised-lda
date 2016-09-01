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

// Just the template instantiations all the rest is defined in the headers.
template class LDABuilder<float>;
template class LDABuilder<double>;

}
