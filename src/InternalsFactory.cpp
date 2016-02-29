
#include "SeededInitialization.hpp"
#include "SupervisedEStep.hpp"
#include "SupervisedMStep.hpp"
#include "UnsupervisedEStep.hpp"
#include "UnsupervisedMStep.hpp"

#include "InternalsFactory.hpp"


template <typename Scalar>
std::shared_ptr<IInitialization<Scalar> > InternalsFactory<Scalar>::create_initialization(
    int id,
    std::vector<Scalar> parameters
) {
    std::shared_ptr<IInitialization<Scalar> > init = nullptr;

    switch (id) {
        case IInitialization<Scalar>::Seeded:
            init = std::make_shared<SeededInitialization<Scalar> >();
            break;
        case IInitialization<Scalar>::Random:
            break;
        default:
            break;
    }

    if (init != nullptr)
        init->set_parameters(parameters);

    return init;
}


template <typename Scalar>
std::shared_ptr<IEStep<Scalar> > InternalsFactory<Scalar>::create_e_step(
    int id,
    std::vector<Scalar> parameters
) {
    std::shared_ptr<IEStep<Scalar> > e_step = nullptr;

    switch (id) {
        case IEStep<Scalar>::BatchUnsupervised:
            e_step = std::make_shared<UnsupervisedEStep<Scalar> >();
            break;
        case IEStep<Scalar>::BatchSupervised:
            e_step = std::make_shared<SupervisedEStep<Scalar> >();
            break;
        case IEStep<Scalar>::OnlineUnsupervised:
            break;
        case IEStep<Scalar>::OnlineSupervised:
            break;
        default:
            break;
    }

    if (e_step != nullptr)
        e_step->set_parameters(parameters);

    return e_step;
}


template <typename Scalar>
std::shared_ptr<IMStep<Scalar> > InternalsFactory<Scalar>::create_m_step(
    int id,
    std::vector<Scalar> parameters
) {
    std::shared_ptr<IMStep<Scalar> > m_step = nullptr;

    switch (id) {
        case IMStep<Scalar>::BatchUnsupervised:
            m_step = std::make_shared<UnsupervisedMStep<Scalar> >();
            break;
        case IMStep<Scalar>::BatchSupervised:
            m_step = std::make_shared<SupervisedMStep<Scalar> >();
            break;
        case IMStep<Scalar>::OnlineUnsupervised:
            break;
        case IMStep<Scalar>::OnlineSupervised:
            break;
        default:
            break;
    }

    if (m_step != nullptr)
        m_step->set_parameters(parameters);

    return m_step;
}


// template instantiation
template class InternalsFactory<float>;
template class InternalsFactory<double>;
