#ifndef _IESTEP_HPP_
#define _IESTEP_HPP_

#include <Eigen/Core>

#include "ISerializable.hpp"

using namespace Eigen;

/**
  * Interface that implements an e-step iteration for a single document
  */
template <typename Scalar>
class IEStep : public ISerializable<Scalar>
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
          * Maximize the ELBO
          *
          * @param X        The word counts in column-major order for a single 
          *                 document
          * @param y        The class label as integer for the current document
          * @param alpha    The Dirichlet priors
          * @param beta     The over word topic distributiosn
          * @param eta      The classification parameters
          * @param phi      The Multinomial parameters
          * @param gamma    The Dirichlet parameters
          * @return         The likelihood so far
          */
        virtual Scalar doc_e_step(
            const VectorXi &X,
            int y,
            const VectorX &alpha,
            const MatrixX &beta,
            const MatrixX &eta,
            Ref<MatrixX> phi,
            Ref<VectorX> gamma
        )=0;
};
#endif //  _IESTEP_HPP_
