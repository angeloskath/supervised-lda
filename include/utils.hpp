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
    Scalar p;
    p = 1/(x*x);
    p = (((0.004166666666667*p-0.003968253986254)*p+0.008333333333333)*p-0.083333333333333)*p;
    return p+std::log(x)-0.5/x-1/(x-1)-1/(x-2)-1/(x-3)-1/(x-4)-1/(x-5)-1/(x-6);
}


#endif // UTILS_H
