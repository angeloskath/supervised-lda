#ifndef _MULTINOMIAL_LOGISTIC_REGRESSION
#define _MULTINOMIAL_LOGISTIC_REGRESSION

#include <cmath>

#include <Eigen/Core>

using namespace Eigen;


template <typename Scalar>
class MultinomialLogisticRegression
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;

    public:
        MultinomialLogisticRegression(const MatrixX &X, const VectorXi &y);
        
        /**
         * The value of the objective function to be solved.
         *
         * In our case the objective function is
         * \eta^T E_q[Z]y - log(\sum_{y=1}^C exp(\eta^T E_q[Z]y))
         */
        Scalar value(const MatrixX &eta);
        
        void gradient(const MatrixX &eta, MatrixX &grad);
    private:
        const MatrixX &X_;
        const VectorXi &y_;
};


#endif // _MULTINOMIAL_LOGISTIC_REGRESSION
