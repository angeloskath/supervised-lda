#ifndef _LDAPLUSPLUS_OPTIMIZATION_MULTINOMIAL_LOGISTIC_REGRESSION
#define _LDAPLUSPLUS_OPTIMIZATION_MULTINOMIAL_LOGISTIC_REGRESSION

#include <cmath>

#include <Eigen/Core>

namespace ldaplusplus {
namespace optimization {


/**
 * MultinomialLogisticRegression is an implementation of the multinomial
 * logistic loss function (without bias unit).
 *
 * It follows the protocol used by GradientDescent. For the specific function
 * implementations see value() and gradient().
 */
template <typename Scalar>
class MultinomialLogisticRegression
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;

    public:
        /**
         * @param X  The documents defining the minimization problem (\f$X \in
         *           \mathbb{R}^{D \times N}\f$)
         * @param y  The class indexes for each document (\f$y \in
         *           \mathbb{N}^N\f$)
         * @param Cy A different weight for each class in the optimization
         *           problem
         * @param L  The L2 regularization penalty for the weights
         */
        MultinomialLogisticRegression(const MatrixX &X, const Eigen::VectorXi &y, VectorX Cy, Scalar L);
        /**
         * @param X  The documents defining the minimization problem (\f$X \in
         *           \mathbb{R}^{D \times N}\f$)
         * @param y  The class indexes for each document (\f$y \in
         *           \mathbb{N}^N\f$)
         * @param L  The L2 regularization penalty for the weights
         */
        MultinomialLogisticRegression(const MatrixX &X, const Eigen::VectorXi &y, Scalar L);

        /**
         * The value of the objective function to be minimized.
         *
         * \f$N\f$ is the number of documents (different vectors), \f$X_n \in
         * \mathbb{R}^D\f$ is the nth document, \f$\eta_y \in \mathbb{R}^D\f$
         * is the weights vector for the class \f$y\f$ defining the hyperplane
         * that separates class \f$y\f$ from all the other, finally \f$y_n\f$
         * is the class of the nth document.
         *
         * \f[
         *     J = -\sum_{n=1}^N C_{y_n}\left(\eta_{y_n}^T X_n - \log\left(
         *         \sum_{\hat{y}=1}^Y \exp\left( \eta_{\hat{y}}^T X_n \right)
         *         \right)\right) +
         *         \frac{L}{2} \left\| \eta \right\|_F^2
         * \f]
         *
         * @param eta The weights of the linear model (\f$\eta \in
         *            \mathbb{R}^{D \times Y}\f$)
         */
        Scalar value(const MatrixX &eta) const;

        /**
         * The gradient of the objective function implemented in value().
         *
         * We use \f$I(y) \in \mathbb{R}^Y\f$ as the indicator vector of
         * \f$y\f$ (a vector with all the values 0 except at the yth position).
         *
         * \f[
         *     \nabla_{\eta} J = -\sum_{n=1}^N C_{y_n} \left(
         *         X_n I(y_n)^T -
         *         \frac{\sum_{\hat{y}=1}^Y X_n I(\hat{y})^T \exp(\eta_{\hat{y}}^T X_n)}
         *              {\sum_{\hat{y}=1}^Y \exp(\eta_{\hat{y}}^T X_n)}
         *         \right) +
         *         L \eta
         * \f]
         * 
         * @param eta  The weights of the linear model (\f$\eta \in
         *             \mathbb{R}^{D \times Y}\f$)
         * @param grad A matrix of dimensions equal to \f$\eta\f$ that will
         *             hold the result
         */
        void gradient(const MatrixX &eta, Eigen::Ref<MatrixX> grad) const;

    private:
        const MatrixX &X_;
        const Eigen::VectorXi &y_;
        Scalar L_;
        VectorX Cy_;
};


}  // namespace optimization
}  // namespace ldaplusplus
#endif // _LDAPLUSPLUS_OPTIMIZATION_MULTINOMIAL_LOGISTIC_REGRESSION
