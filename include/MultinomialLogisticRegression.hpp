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
        MultinomialLogisticRegression(const MatrixX &X, const VectorXi &y, Scalar L);
        
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
        const VectorXi &y_;
        Scalar L_;
};


#endif // _MULTINOMIAL_LOGISTIC_REGRESSION
