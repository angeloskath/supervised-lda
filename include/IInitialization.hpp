#ifndef _IINITIALIZATION_HPP
#define _IINITIALIZATION_HPP

#include <Eigen/Core>

using namespace Eigen;

template <typename Scalar>
class IInitialization
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;
    
    public:
        virtual void initialize_model_parameters(
            const MatrixXi &X,
            const VectorXi &y,
            Ref<VectorX> alpha,
            Ref<MatrixX> beta,
            Ref<MatrixX> eta
        )=0;
};

#endif //  _IINITIALIZATION_HPP
