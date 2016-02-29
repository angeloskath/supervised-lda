#ifndef _INTERNALS_FACTORY_HPP_
#define _INTERNALS_FACTORY_HPP_


#include <memory>


#include "IInitialization.hpp"
#include "IEStep.hpp"
#include "IMStep.hpp"


/**
 * A factory interface for constructing internal modules implementing
 * initialization strategies and E-M pairs using the serialized information.
 */
template <typename Scalar>
class IInternalsFactory
{
    public:
        virtual std::shared_ptr<IInitialization<Scalar> > create_initialization(
            int id,
            std::vector<Scalar> parameters
        )=0;

        virtual std::shared_ptr<IEStep<Scalar> > create_e_step(
            int id,
            std::vector<Scalar> parameters
        )=0;

        virtual std::shared_ptr<IMStep<Scalar> > create_m_step(
            int id,
            std::vector<Scalar> parameters
        )=0;
};


template <typename Scalar>
class InternalsFactory : public IInternalsFactory<Scalar>
{
    public:
        std::shared_ptr<IInitialization<Scalar> > create_initialization(
            int id,
            std::vector<Scalar> parameters
        ) override;

        std::shared_ptr<IEStep<Scalar> > create_e_step(
            int id,
            std::vector<Scalar> parameters
        ) override;

        std::shared_ptr<IMStep<Scalar> > create_m_step(
            int id,
            std::vector<Scalar> parameters
        ) override;
};


#endif  // _INTERNALS_FACTORY_HPP_
