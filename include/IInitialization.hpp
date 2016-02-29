#ifndef _IINITIALIZATION_HPP
#define _IINITIALIZATION_HPP

#include <Eigen/Core>

#include "ISerializable.hpp"

using namespace Eigen;

template <typename Scalar>
class IInitialization : public ISerializable<Scalar>
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;
    
    public:
        enum Type
        {
            Seeded = 0,
            Random
        };

        virtual void initialize_model_parameters(
            const MatrixXi &X,
            const VectorXi &y,
            Ref<VectorX> alpha,
            Ref<MatrixX> beta,
            Ref<MatrixX> eta
        )=0;
};

#endif  // _IINITIALIZATION_HPP
