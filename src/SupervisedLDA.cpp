
#include "SupervisedLDA.hpp"

template <typename Scalar>
void SupervisedLDA<Scalar>::fit(MatrixXi X, VectorXi y) {
}

template <typename Scalar>
void SupervisedLDA<Scalar>::partial_fit(MatrixXi X, VectorXi y) {
}

// Template instantiation
template class SupervisedLDA<float>;
template class SupervisedLDA<double>;
