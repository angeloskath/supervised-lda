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
        ) : m_step_iterations_(m_step_iterations),
            m_step_tolerance_(m_step_tolerance),
            regularization_penalty_(regularization_penalty) {};
        
        /**
         * Maximize the ELBO w.r.t to \beta and \eta.
         *
         * @param expected_Z_bar Is the expected values of Z_bar for every
         *                       document
         * @param b              The unnormalized new betas
         * @param y              The class indexes for every document
         * @param beta           The topic word distributions
         * @param eta            The classification parameters
         * @return               The likelihood of the Multinomial logistic
         *                       regression
         */
        Scalar m_step(
            const MatrixX &expected_z_bar,
            const MatrixX &b,
            const VectorXi &y,
            Ref<MatrixX> beta,
            Ref<MatrixX> eta
        );
    
    private:
        // The maximum number of iterations in M-step
        size_t m_step_iterations_;
        // The convergence tolerance for the maximazation of the ELBO w.r.t.
        // eta in M-step
        Scalar m_step_tolerance_;
        // The regularization penalty for the multinomial logistic regression
        Scalar regularization_penalty_;
};
#endif  // _SUPERVISEDMSTEP_HPP
