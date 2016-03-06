#ifndef _UNSUPERVISEDMSTEP_HPP_
#define _UNSUPERVISEDMSTEP_HPP_

#include "IMStep.hpp"

template <typename Scalar>
class UnsupervisedMStep : public IMStep<Scalar>
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;
    
    public:
        UnsupervisedMStep() {}
        
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
};
#endif  // _UNSUPERVISEDMSTEP_HPP_ 
