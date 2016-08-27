#ifndef _SECOND_ORDER_LOGISTIC_REGRESSION_APPROXIMATION_HPP_
#define _SECOND_ORDER_LOGISTIC_REGRESSION_APPROXIMATION_HPP_

#include <cmath>

#include <Eigen/Core>

using namespace Eigen;

namespace ldaplusplus {
namespace optimization {


/**
 * SecondOrderLogisticRegressionApproximation is a second order taylor
 * approximation to the expectation of the logistic loss function of a random
 * variable.
 *
 * We use this class to approximate the following equation in the lower bound
 * of the likelihood of an LDA model. \f$q\f$ is the variational distribution
 * used and \f$\bar{z}\f$ is a random variable (the mean of the topic
 * assignments). The equation below is for a single document.
 *
 * \f[
 *     \mathbb{E}_q\left[
 *         \eta_{y_n}^T \bar{z} -
 *         \log\left( \sum_{\hat{y}=1}^Y \exp(\eta_{\hat{y}}^T \bar{z}) \right)
 *         \right] \approx
 *         \eta_{y_n}^T \mathbb{E}_q[\bar{z}] -
 *         \log \sum_{\hat{y}=1}^Y \exp(\eta_{\hat{y}}^T \mathbb{E}_q[\bar{z})\left(
 *         1 + \frac{1}{2} \eta_{\hat{y}}^T \mathbb{V}_q[\bar{z}] \eta_{\hat{y}}
 *         \right)
 * \f]
 */
template <typename Scalar>
class SecondOrderLogisticRegressionApproximation
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;

    public:
        /**
         * @param X     The documents defining the minimization problem (\f$X
         *              \in \mathbb{R}^{D \times N}\f$)
         * @param X_var A vector containing the variance matrix for each
         *              document (\f$X_{\text{var}} \in \mathbb{R}^{N \times D
         *              \times D}\f$)
         * @param y     The class indexes for each document (\f$y \in
         *              \mathbb{N}^N\f$)
         * @param Cy    A different weight for each class in the optimization
         *              problem
         * @param L     The L2 regularization penalty for the weights
         */
        SecondOrderLogisticRegressionApproximation(
            const MatrixX &X,
            const std::vector<MatrixX> &X_var,
            const VectorXi &y,
            VectorX Cy,
            Scalar L
        );

        /**
         * @param X     The documents defining the minimization problem (\f$X
         *              \in \mathbb{R}^{D \times N}\f$)
         * @param X_var A vector containing the variance matrix for each
         *              document (\f$X_{\text{var}} \in \mathbb{R}^{N \times D
         *              \times D}\f$)
         * @param y     The class indexes for each document (\f$y \in
         *              \mathbb{N}^N\f$)
         * @param L     The L2 regularization penalty for the weights
         */
        SecondOrderLogisticRegressionApproximation(
            const MatrixX &X,
            const std::vector<MatrixX> &X_var,
            const VectorXi &y,
            Scalar L
        );

        /**
         * The value of the objective function to be minimized.
         *
         * \f$N\f$ is the number of documents (different vectors), \f$X_n \in
         * \mathbb{R}^D\f$ is the nth document, \f$\eta_y \in \mathbb{R}^D\f$
         * is the weights vector for the class \f$y\f$ defining the hyperplane
         * that separates class \f$y\f$ from all the other, \f$y_n\f$
         * is the class of the nth document and finally \f$X_n^{\text{var}} \in
         * \mathbb{R}^{D \times D}\f$ is the variance of the nth document (see
         * the class description).
         *
         * \f[
         *     J = - \sum_{n=1}^N C_{y_n} \left(
         *         \eta_{y_n}^T X_n -
         *         \log \sum_{\hat{y}=1}^Y \exp(\eta_{\hat{y}}^T X_n) \left(
         *         1 +
         *         \frac{1}{2} \eta_{\hat{y}}^T X_n^{\text{var}} \eta_{\hat{y}}
         *         \right)
         *         \right) +
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
         *     \nabla_{\eta} J = - \sum_{n=1}^N C_{y_n} \left(
         *         X_n I(y_n)^T -
         *         \frac{
         *              \sum_{\hat{y}=1}^Y \left(\left(
         *              X_n \exp(\eta_{\hat{y}}^T X_n) \left(
         *              1 +
         *              \frac{1}{2} \eta_{\hat{y}}^T X_n^{\text{var}} \eta_{\hat{y}}
         *              \right)\right) + \left(
         *              \frac{1}{2}
         *              \exp(\eta_{\hat{y}}^T X_n) \eta_{\hat{y}}^T \left(
         *              X_n^{\text{var}} + \left(X_n^{\text{var}}\right)^T
         *              \right)\right)
         *              \right) I(\hat{y})^T
         *              }
         *              {
         *              \sum_{\hat{y}=1}^Y \exp(\eta_{\hat{y}}^T X_n) \left(
         *              1 +
         *              \frac{1}{2} \eta_{\hat{y}}^T X_n^{\text{var}} \eta_{\hat{y}}
         *              \right)
         *              }
         *         \right) +
         *         L \eta
         * \f]
         *
         * @param eta  The weights of the linear model (\f$\eta \in
         *             \mathbb{R}^{D \times Y}\f$)
         * @param grad A matrix of dimensions equal to \f$\eta\f$ that will
         *             hold the result
         */
        void gradient(const MatrixX &eta, Ref<MatrixX> grad) const;

    private:
        const MatrixX &X_;
        const std::vector<MatrixX> &X_var_;
        const VectorXi &y_;
        Scalar L_;
        VectorX Cy_;
};


}  // namespace optimization
}  // namespace ldaplusplus

#endif // _SECOND_ORDER_LOGISTIC_REGRESSION_APPROXIMATION_HPP_
