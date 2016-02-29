#ifndef _IINITIALIZATION_HPP
#define _IINITIALIZATION_HPP

#include <Eigen/Core>

#include "Events.hpp"
#include "ISerializable.hpp"

using namespace Eigen;

template <typename Scalar>
class IInitialization : public ISerializable<Scalar>, public EventDispatcherComposition
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;
    
    public:
        enum Type
        {
            Seeded = 0,
            Random
        };

        /**
          * Function used for the initialization of model parameters, namely
          * alpha, beta, eta
          *
          * @param X      The word counts in column-major order for a single 
          *               document
          * @param y      The class label as integer for the current document
          * @param alpha  The Dirichlet priors
          * @param beta   The over word topic distributiosn
          * @param eta    The classification parameters
          */
        virtual void initialize_model_parameters(
            const MatrixXi &X,
            const VectorXi &y,
            VectorX &alpha,
            MatrixX &beta,
            MatrixX &eta
        )=0;
};

#endif  // _IINITIALIZATION_HPP
