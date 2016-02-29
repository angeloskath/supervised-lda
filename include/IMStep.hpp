#ifndef _IEMSTEP_HPP_
#define _IEMSTEP_HPP_

#include <Eigen/Core>

using namespace Eigen;

template <typename Scalar>
class IMStep
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;
    
    public:
        virtual Scalar m_step(
            const MatrixX &expected_z_bar,
            const MatrixX &b,
            const VectorXi &y,
            Ref<MatrixX> beta,
            Ref<MatrixX> eta,
        )=0;

        /**
         * This function calculates all necessary parameters, that
         * will be used for the maximazation step.
         */
        virtual void doc_m_step(
           const VectorXi &X,
           const MatrixX &phi,
           Ref<MatrixX> b,
           Ref<VectorX> expected_z_bar
        )=0;
};

#endif //  _IEMSTEP_HPP_
