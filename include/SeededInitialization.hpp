#ifndef _SEEDEDINITIALIZATION_HPP_
#define _SEEDEDINITIALIZATION_HPP_

#include "IInitialization.hpp"

template <typename Scalar>
class SeededInitialization : public IInitialization<Scalar>
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;
    
    public:
        SeededInitialization(size_t topics) : topics_(topics) {};
        
        /**
          * Function used for the initialization of model parameters, namely
          * alpha, beta, eta in a seeded way.
          *
          * @param X      The word counts in column-major order for a single 
          *               document
          * @param y      The class label as integer for the current document
          * @param alpha  The Dirichlet priors
          * @param beta   The over word topic distributiosn
          * @param eta    The classification parameters
          */
        void initialize_model_parameters(
            const MatrixXi &X,
            const VectorXi &y,
            Ref<VectorX> alpha,
            Ref<MatrixX> beta,
            Ref<MatrixX> eta
        );

    private:
        // The total number of topics used
        size_t topics_; 
};
#endif  // _SEEDEDINITIALIZATION_HPP_
