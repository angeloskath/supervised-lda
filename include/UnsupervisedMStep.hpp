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
         * @param expected_Z_bar Is the expected values of Z_bar for every
         *                       document
         * @param b              The unnormalized new betas
         * @param y              The class indexes for every document
         * @param beta           The topic word distributions
         * @param eta            The classification parameters
         * @return               The likelihood of the Multinomial logistic
         *                       regression
         */
        virtual Scalar m_step(
            const MatrixX &expected_z_bar,
            const MatrixX &b,
            const VectorXi &y,
            Ref<MatrixX> beta,
            Ref<MatrixX> eta
        ) override;

        /**
         * This function calculates all necessary parameters, that
         * will be used for the maximazation step.
         *
         * @param X              The word counts in column-major order for a single 
         *                       document
         * @param phi            The Multinomial parameters
         * @param b              The unnormalized new betas
         * @param expected_Z_bar Is the expected values of Z_bar for every
         *                       document
         */
        void doc_m_step(
           const VectorXi &X,
           const MatrixX &phi,
           Ref<MatrixX> b,
           Ref<VectorX> expected_z_bar
        ) override;

};
#endif  // _UNSUPERVISEDMSTEP_HPP_ 
