#ifndef _MULTINOMIAL_SUPERVISED_M_STEP_HPP_
#define _MULTINOMIAL_SUPERVISED_M_STEP_HPP_

#include "IMStep.hpp"

template <typename Scalar>
class MultinomialSupervisedMStep : public IMStep<Scalar>
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;
    
    public:
        MultinomialSupervisedMStep(int num_classes, Scalar mu=2)
            : num_classes_(num_classes), mu_(mu)
        {}

        /**
         * Maximize the ELBO w.r.t to \beta.
         *
         * @param parameters       Model parameters, after being updated in m_step
         */
        virtual void m_step(
            std::shared_ptr<Parameters> parameters
        ) override;

        /**
         * This function calculates all necessary parameters, that
         * will be used for the maximazation step.
         *
         * @param doc              A single document
         * @param v_parameters     The variational parameters used in m-step
         *                         in order to maximize model parameters
         * @param m_parameters     Model parameters, used as output in case of 
         *                         online methods
         */
        virtual void doc_m_step(
            const std::shared_ptr<Document> doc,
            const std::shared_ptr<Parameters> v_parameters,
            std::shared_ptr<Parameters> m_parameters
        ) override;

    private:
        MatrixX b_;
        MatrixX h_;
        int num_classes_;
        Scalar mu_;
};
#endif  // _MULTINOMIAL_SUPERVISED_M_STEP_HPP_
