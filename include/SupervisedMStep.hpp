#ifndef _SUPERVISEDMSTEP_HPP
#define _SUPERVISEDMSTEP_HPP

#include "UnsupervisedMStep.hpp"

template <typename Scalar>
class SupervisedMStep : public UnsupervisedMStep<Scalar>
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;
    
    public:
        SupervisedMStep(
            size_t m_step_iterations = 10,
            Scalar m_step_tolerance = 1e-2,
            Scalar regularization_penalty = 1e-2
        ) : docs_(0),
            m_step_iterations_(m_step_iterations),
            m_step_tolerance_(m_step_tolerance),
            regularization_penalty_(regularization_penalty) {};
        
        /**
         * Maximize the ELBO w.r.t to \beta and \eta.
         *
         * @param parameters           Model parameters, after being updated in m_step
         */
        void m_step(
            std::shared_ptr<Parameters> parameters
        );
        
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
        void doc_m_step(
            const std::shared_ptr<Document> doc,
            const std::shared_ptr<Parameters> v_parameters,
            std::shared_ptr<Parameters> m_parameters
        );
        
    private:
        // The maximum number of iterations in M-step
        size_t m_step_iterations_;
        // The convergence tolerance for the maximazation of the ELBO w.r.t.
        // eta in M-step
        Scalar m_step_tolerance_;
        // The regularization penalty for the multinomial logistic regression
        Scalar regularization_penalty_;
        
        // Number of documents processed so far
        int docs_;
        MatrixX expected_z_bar_;
        VectorXi y_;
};
#endif  // _SUPERVISEDMSTEP_HPP
