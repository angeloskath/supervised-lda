#ifndef _LDA_BUILDER_HPP_
#define _LDA_BUILDER_HPP_


#include <memory>
#include <vector>

#include "IEStep.hpp"
#include "IInitialization.hpp"
#include "IMStep.hpp"
#include "InternalsFactory.hpp"
#include "LDA.hpp"


template <typename Scalar>
class ILDABuilder
{
    public:
        virtual operator LDA<Scalar>() const = 0;
};


/**
 * The LDABuilder provides a simple interface to build an LDA.
 *
 * Examples:
 *
 * LDA<double> lda = LDABuilder<double>();
 * LDA<double> lda = LDABuilder<double>().
 *                      set_topics(100).
 *                      set_iterations(20).
 *                      set_initialization(IInitialization<double>::Seeded, {}).
 *                      set_e_step(IEStep<double>::BatchSupervised, 10, 1e-2, 20).
 *                      set_m_step(IMStep<double>::BatchSupervised, 10, 1e-2);
 */
template <typename Scalar>
class LDABuilder : public ILDABuilder<Scalar>
{
    public:
        LDABuilder();
        LDABuilder(std::shared_ptr<IInternalsFactory<Scalar> > factory);

        // set generic parameters
        LDABuilder & set_topics(size_t topics);
        LDABuilder & set_iterations(size_t topics);

        // set the initialization
        template <typename... P>
        LDABuilder & set_initialization(
            typename IInitialization<Scalar>::Type initialization_type,
            P... parameters
        ) {
            return set_initialization(
                initialization_type,
                std::vector<Scalar>{static_cast<Scalar>(parameters)...}
            );
        }
        LDABuilder & set_initialization(
            typename IInitialization<Scalar>::Type initialization_type,
            std::vector<Scalar> parameters
        );

        // set the unsupervised e step
        LDABuilder & set_unsupervised_e_step(size_t iterations, Scalar tolerance);
        LDABuilder & set_unsupervised_m_step(size_t iterations, Scalar tolerance);

        // set the actual e step
        template <typename... P>
        LDABuilder & set_e_step(
            typename IEStep<Scalar>::Type e_step_type,
            P... parameters
        ) {
            return set_e_step(
                e_step_type,
                std::vector<Scalar>{static_cast<Scalar>(parameters)...}
            );
        }
        LDABuilder & set_e_step(
            typename IEStep<Scalar>::Type e_step_type,
            std::vector<Scalar> parameters
        );

        // set the actual m step
        template <typename... P>
        LDABuilder & set_m_step(
            typename IMStep<Scalar>::Type m_step_type,
            P... parameters
        ) {
            return set_m_step(
                m_step_type,
                std::vector<Scalar>{static_cast<Scalar>(parameters)...}
            );
        }
        LDABuilder & set_m_step(
            typename IMStep<Scalar>::Type m_step_type,
            std::vector<Scalar> parameters
        );

        virtual operator LDA<Scalar>() const;

    private:
        // the factory used to create the internal modules
        std::shared_ptr<IInternalsFactory<Scalar> > factory_;

        // generic lda parameters
        size_t topics_;
        size_t iterations_;

        // initialization
        typename IInitialization<Scalar>::Type initialization_type_;
        std::vector<Scalar> initialization_parameters_;

        // unsupervised e step
        size_t unsupervised_e_step_iterations_;
        Scalar unsupervised_e_step_tolerance_;
        size_t unsupervised_m_step_iterations_;
        Scalar unsupervised_m_step_tolerance_;

        // actual e step
        typename IEStep<Scalar>::Type e_step_type_;
        std::vector<Scalar> e_step_parameters_;

        // actual m step
        typename IMStep<Scalar>::Type m_step_type_;
        std::vector<Scalar> m_step_parameters_;
};


#endif  //_LDA_BUILDER_HPP_
