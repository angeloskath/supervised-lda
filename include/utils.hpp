#ifndef UTILS_H
#define UTILS_H

#include <cmath>

#include <Eigen/Core>


using namespace Eigen;


/**
 * @brief This function is used for the calculation of the digamma function,
 * which is the logarithmic derivative of the Gamma Function. The digamma
 * function is computable via Taylor approximations (Abramowitz and Stegun,
 * 1970)
**/
template <typename Scalar>
Scalar digamma(Scalar x) {
    Scalar result = 0, xx, xx2, xx4;

    for (; x < 7; ++x) {
        result -= 1/x;
    }
    x -= 1.0/2.0;
    xx = 1.0/x;
    xx2 = xx*xx;
    xx4 = xx2*xx2;
    result += std::log(x)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4;

    return result;
}


template <typename Scalar>
struct CwiseDigamma
{
    const Scalar operator()(const Scalar &x) const {
        return digamma(x);
    }
};


template <typename Scalar>
struct CwiseLgamma
{
    const Scalar operator()(const Scalar &x) const {
        return std::lgamma(x);
    }
};


/**
 * Reshape a matrix into a vector by copying the matrix into the vector in a
 * way to avoid errors by Eigen expressions.
 */
template <typename Scalar>
void reshape_into(
    const Matrix<Scalar, Dynamic, Dynamic> &src,
    Matrix<Scalar, Dynamic, 1> &dst
) {
    size_t srcR = src.rows();
    size_t srcC = src.cols();

    for (int c=0; c<srcC; c++) {
        dst.segment(c*srcR, srcR) = src.col(c);
    }
}


/**
 * Reshape a vector into a matrix by copying the vector into the matrix in a
 * way to avoid errors by Eigen expressions.
 */
template <typename Scalar>
void reshape_into(
    const Matrix<Scalar, Dynamic, 1> &src,
    Matrix<Scalar, Dynamic, Dynamic> &dst
) {
    size_t dstR = dst.rows();
    size_t dstC = dst.cols();

    for (int c=0; c<dstC; c++) {
        dst.col(c) = src.segment(c*dstR, dstR);
    }
}


#endif // UTILS_H
