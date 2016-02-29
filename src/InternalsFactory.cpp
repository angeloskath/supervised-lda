
#include "InternalsFactory.hpp"


template <typename Scalar>
std::shared_ptr<IInitialization<Scalar> > InternalsFactory<Scalar>::create_initialization(
    int id,
    std::vector<Scalar> parameters
) {
    switch (id) {
        case IInitialization<Scalar>::Seeded:
            return nullptr;
        case IInitialization<Scalar>::Random:
            return nullptr;
        default:
            return nullptr;
    }
}


template <typename Scalar>
std::shared_ptr<IEStep<Scalar> > InternalsFactory<Scalar>::create_e_step(
    int id,
    std::vector<Scalar> parameters
) {
    switch (id) {
        case IEStep<Scalar>::BatchUnsupervised:
            return nullptr;
        case IEStep<Scalar>::BatchSupervised:
            return nullptr;
        case IEStep<Scalar>::OnlineUnsupervised:
            return nullptr;
        case IEStep<Scalar>::OnlineSupervised:
            return nullptr;
        default:
            return nullptr;
    }
}


template <typename Scalar>
std::shared_ptr<IMStep<Scalar> > InternalsFactory<Scalar>::create_m_step(
    int id,
    std::vector<Scalar> parameters
) {
    switch (id) {
        case IMStep<Scalar>::BatchUnsupervised:
            return nullptr;
        case IMStep<Scalar>::BatchSupervised:
            return nullptr;
        case IMStep<Scalar>::OnlineUnsupervised:
            return nullptr;
        case IMStep<Scalar>::OnlineSupervised:
            return nullptr;
        default:
            return nullptr;
    }
}


// template instantiation
template class InternalsFactory<float>;
template class InternalsFactory<double>;
