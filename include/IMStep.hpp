#ifndef _IMSTEP_HPP_
#define _IMSTEP_HPP_

#include <Eigen/Core>

#include "Events.hpp"
#include "ISerializable.hpp"

using namespace Eigen;

template <typename Scalar>
class IMStep : public ISerializable<Scalar>, public EventDispatcherComposition
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;
    
    public:
        enum Type
        {
            BatchUnsupervised = 0,
            BatchSupervised,
            OnlineUnsupervised,
            OnlineSupervised
        };

        /**
         * Maximize the ELBO.
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
        )=0;

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
        virtual void doc_m_step(
           const VectorXi &X,
           const MatrixX &phi,
           Ref<MatrixX> b,
           Ref<VectorX> expected_z_bar
        )=0;
};

#endif  // _IMSTEP_HPP_
