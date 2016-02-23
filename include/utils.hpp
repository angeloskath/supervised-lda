#ifndef UTILS_H
#define UTILS_H

#include <cmath>

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

#endif // UTILS_H
