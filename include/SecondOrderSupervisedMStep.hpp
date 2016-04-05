#ifndef _SECOND_ORDER_SUPERVISED_M_STEP_HPP_
#define _SECOND_ORDER_SUPERVISED_M_STEP_HPP_

#include <vector>

#include "UnsupervisedMStep.hpp"

template <typename Scalar>
class SecondOrderSupervisedMStep : public UnsupervisedMStep<Scalar>
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;
    
    public:
        SecondOrderSupervisedMStep(
            size_t m_step_iterations = 10,
            Scalar m_step_tolerance = 1e-2,
            Scalar regularization_penalty = 1e-2
        ) : m_step_iterations_(m_step_iterations),
            m_step_tolerance_(m_step_tolerance),
            regularization_penalty_(regularization_penalty),
            docs_(0)
        {}
        
        /**
         * Maximize the ELBO w.r.t to \beta and \eta.
         *
         * @param parameters           Model parameters, after being updated in m_step
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
        // The maximum number of iterations in M-step
        size_t m_step_iterations_;
        // The convergence tolerance for the maximazation of the ELBO w.r.t.
        // eta in M-step
        Scalar m_step_tolerance_;
        // The regularization penalty for the multinomial logistic regression
        Scalar regularization_penalty_;
        
        // Number of documents processed so far
        int docs_;
        MatrixX phi_scaled;
        MatrixX expected_z_bar_;
        std::vector<MatrixX> variance_z_bar_;
        VectorXi y_;
};
#endif  // _SECOND_ORDER_SUPERVISED_M_STEP_HPP_
