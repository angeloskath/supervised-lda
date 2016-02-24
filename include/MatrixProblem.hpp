#ifndef _MINIMIZER_HPP_
#define _MINIMIZER_HPP_


#include <Eigen/Core>
#include <cppoptlib/problem.h>

#include "utils.hpp"


using namespace Eigen;


/**
 * Purpose of this class is to wrap a problem implementing gradient() and
 * value() but which accepts Matrices as arguments instead of Vector and
 * transform it to a cppoptlib problem.
 */
template <class ProblemType, typename Scalar>
class MatrixProblem : public cppoptlib::Problem<Scalar>
{
    public:
        MatrixProblem(
            const ProblemType &problem,
            size_t rows,
            size_t cols
        ) : problem_(problem),
            x_(rows, cols),
            g_(rows, cols)
        {}

        Scalar value(const Matrix<Scalar, Dynamic, 1> &x) {
            reshape_into(x, x_);

            return problem_.value(x_);
        }

        void gradient(
            const Matrix<Scalar, Dynamic, 1> &x,
            const Matrix<Scalar, Dynamic, 1> &g
        ) {
            reshape_into(x, x_);

            problem_.gradient(x_, g_);

            reshape_into(g_, g);
        }

    private:
        const ProblemType &problem_;
        Matrix<Scalar, Dynamic, Dynamic> x_;
        Matrix<Scalar, Dynamic, Dynamic> g_;
};


#endif  // _MINIMIZER_HPP_
