#ifndef _SECOND_ORDER_LOGISTIC_REGRESSION_APPROXIMATION_HPP_
#define _SECOND_ORDER_LOGISTIC_REGRESSION_APPROXIMATION_HPP_

#include <cmath>

#include <Eigen/Core>

using namespace Eigen;


template <typename Scalar>
class SecondOrderLogisticRegressionApproximation
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;

    public:
        SecondOrderLogisticRegressionApproximation(
            const MatrixX &X,
            const std::vector<MatrixX> &X_var,
            const VectorXi &y,
            VectorX Cy,
            Scalar L
        );
        SecondOrderLogisticRegressionApproximation(
            const MatrixX &X,
            const std::vector<MatrixX> &X_var,
            const VectorXi &y,
            Scalar L
        );
        
        /**
         * The value of the objective function to be solved.
         *
         * In our case the objective function for each document is
         * \eta^T E_q[Z]y - log(\sum_{y=1}^C exp(\eta^T E_q[Z]y))
         */
        Scalar value(const MatrixX &eta) const;
        
        /**
         * The gradient of the objective function implemented above.
         *
         * The function to be implemented for each document is
         * E_q[Z]y - \frac{\sum_{y=1}^C E_q[Z]y exp(\eta^T E_q[Z]y}{\sum_{y=1}^C exp(\eta^T E_q[Z]y)}
         * 
         */
        void gradient(const MatrixX &eta, Ref<MatrixX> grad) const;

    private:
        const MatrixX &X_;
        const std::vector<MatrixX> &X_var_;
        const VectorXi &y_;
        Scalar L_;
        VectorX Cy_;
};


#endif // _SECOND_ORDER_LOGISTIC_REGRESSION_APPROXIMATION_HPP_
