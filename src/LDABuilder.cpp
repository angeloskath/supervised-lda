#include "LDABuilder.hpp"

template <typename Scalar>
LDABuilder<Scalar>::LDABuilder() :
    LDABuilder(std::make_shared<InternalsFactory<Scalar> >())
{}

template <typename Scalar>
LDABuilder<Scalar>::LDABuilder(
    std::shared_ptr<IInternalsFactory<Scalar> > factory
) : factory_(factory),
    iterations_(20),
    initialization_type_(IInitialization<Scalar>::Seeded),
    initialization_parameters_({100}),
    unsupervised_e_step_iterations_(10),
    unsupervised_e_step_tolerance_(1e-2),
    unsupervised_m_step_iterations_(10),
    unsupervised_m_step_tolerance_(1e-2),
    e_step_type_(IEStep<Scalar>::BatchUnsupervised),
    e_step_parameters_({10, 1e-2}),
    m_step_type_(IMStep<Scalar>::BatchUnsupervised),
    m_step_parameters_({10, 1e-2})
{}


template <typename Scalar>
LDABuilder<Scalar>::operator LDA<Scalar>() const {
    return LDA<Scalar>(
        // the initialization strategy
        factory_->create_initialization(
            initialization_type_,
            initialization_parameters_
        ),

        // the unsupervised e step used for transform(X)
        factory_->create_e_step(
            IEStep<Scalar>::BatchUnsupervised,
            std::vector<Scalar>{
                static_cast<Scalar>(unsupervised_e_step_iterations_),
                unsupervised_e_step_tolerance_
            }
        ),

        // the unsupervised m step used for transform(X)
        factory_->create_m_step(
            IMStep<Scalar>::BatchUnsupervised,
            std::vector<Scalar>{
                static_cast<Scalar>(unsupervised_m_step_iterations_),
                unsupervised_m_step_tolerance_
            }
        ),

        // the actual e step
        factory_->create_e_step(
            e_step_type_,
            e_step_parameters_
        ),

        // the actual m step
        factory_->create_m_step(
            m_step_type_,
            m_step_parameters_
        ),

        // and the rest of the parameters to create an LDA model
        iterations_
    );
}


template <typename Scalar>
LDABuilder<Scalar> & LDABuilder<Scalar>::set_iterations(size_t iterations) {
    iterations_ = iterations;

    return *this;
}


template <typename Scalar>
LDABuilder<Scalar> & LDABuilder<Scalar>::set_initialization(
    typename IInitialization<Scalar>::Type initialization_type,
    std::vector<Scalar> parameters
) {
    initialization_type_ = initialization_type;
    initialization_parameters_ = parameters;

    return *this;
}


template <typename Scalar>
LDABuilder<Scalar> & LDABuilder<Scalar>::set_e_step(
    typename IEStep<Scalar>::Type e_step_type,
    std::vector<Scalar> parameters
) {
    e_step_type_ = e_step_type;
    e_step_parameters_ = parameters;

    return *this;
}


template <typename Scalar>
LDABuilder<Scalar> & LDABuilder<Scalar>::set_m_step(
    typename IMStep<Scalar>::Type m_step_type,
    std::vector<Scalar> parameters
) {
    m_step_type_ = m_step_type;
    m_step_parameters_ = parameters;

    return *this;
}


template <typename Scalar>
LDABuilder<Scalar> & LDABuilder<Scalar>::set_unsupervised_e_step(
    size_t iterations,
    Scalar tolerance
) {
    unsupervised_e_step_iterations_ = iterations;
    unsupervised_e_step_tolerance_ = tolerance;

    return *this;
}


template <typename Scalar>
LDABuilder<Scalar> & LDABuilder<Scalar>::set_unsupervised_m_step(
    size_t iterations,
    Scalar tolerance
) {
    unsupervised_m_step_iterations_ = iterations;
    unsupervised_m_step_tolerance_ = tolerance;

    return *this;
}


// template instantiation
template class LDABuilder<float>;
template class LDABuilder<double>;
