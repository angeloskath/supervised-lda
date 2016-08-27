#ifndef _SECOND_ORDER_SUPERVISED_M_STEP_HPP_
#define _SECOND_ORDER_SUPERVISED_M_STEP_HPP_

#include <vector>

#include "ldaplusplus/em/UnsupervisedMStep.hpp"

namespace ldaplusplus {
namespace em {


/**
 * SecondOrderSupervisedMStep implements the M step for the categorical
 * supervised LDA.
 *
 * As in SupervisedMStep we delegate the maximization with respect to
 * \f$\beta\f$ to UnsupervisedMStep and then maximize the the lower bound of
 * the log likelihood with respect to \f$\eta\f$ using gradient descent.
 *
 * The difference of SecondOrderSupervisedMStep compared to simple
 * SupervisedMStep is that this class uses the second order taylor
 * approximation (instead of the first) to approximate \f$\mathbb{E}_q[\log p(y
 * \mid z, \eta)]\f$.
 *
 * \f[
 *     \mathcal{L}_{\eta} = \sum_{d=1}^D \eta_{y_d}^T \mathbb{E}_q[\bar{z_d}] -
 *         \sum_{d=1}^D \log \sum_{\hat{y}=1}^C \exp(\eta_{\hat{y}}^T \mathbb{E}_q[\bar{z_d}])
 *         \left(
 *             1 + \frac{1}{2} \eta_{\hat{y}}^T \mathbb{V}_q[\bar{z_d}] \eta_{\hat{y}}
 *         \right)
 * \f]
 *
 * This approximation has been used in [1] but it is slower and requires huge
 * amounts of memory for even moderately large document collections.
 *
 * [1] Chong, Wang, David Blei, and Fei-Fei Li. "Simultaneous image
 *     classification and annotation." Computer Vision and Pattern Recognition,
 *     2009\\. CVPR 2009. IEEE Conference on. IEEE, 2009.
 */
template <typename Scalar>
class SecondOrderSupervisedMStep : public UnsupervisedMStep<Scalar>
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;

    public:
        /**
         * @param m_step_iterations      The maximum number of gradient descent
         *                               iterations
         * @param m_step_tolerance       The minimum relative improvement between
         *                               consecutive gradient descent iterations
         * @param regularization_penalty The L2 penalty for logistic regression
         */
        SecondOrderSupervisedMStep(
            size_t m_step_iterations = 10,
            Scalar m_step_tolerance = 1e-2,
            Scalar regularization_penalty = 1e-2
        ) : m_step_iterations_(m_step_iterations),
            m_step_tolerance_(m_step_tolerance),
            regularization_penalty_(regularization_penalty),
            docs_(0)
        {}

        /**
         * Maximize the ELBO w.r.t. to \f$\beta\f$ and \f$\eta\f$.
         *
         * Delegate the maximization regarding \f$\beta\f$ to UnsupervisedMStep
         * and maximize \f$\mathcal{L}_{\eta}\f$ using gradient descent.
         *
         * @param parameters Model parameters (changed by this method)
         */
        virtual void m_step(
            std::shared_ptr<Parameters> parameters
        ) override;

        /**
         * Delegate the collection of some sufficient statistics to
         * UnsupervisedMStep and keep in memory \f$\mathbb{E}_q[\bar z_d]\f$
         * and \f$\mathbb{V}_q[\bar z_d]\f$ for use in m_step().
         *
         * @param doc          A single document
         * @param v_parameters The variational parameters used in m-step
         *                     in order to maximize model parameters
         * @param m_parameters Model parameters, used as output in case of 
         *                     online methods
         */
        virtual void doc_m_step(
            const std::shared_ptr<Document> doc,
            const std::shared_ptr<Parameters> v_parameters,
            std::shared_ptr<Parameters> m_parameters
        ) override;

    private:
        // The maximum number of iterations in M-step
        size_t m_step_iterations_;
        // The convergence tolerance for the maximazation of the ELBO w.r.t.
        // eta in M-step
        Scalar m_step_tolerance_;
        // The regularization penalty for the multinomial logistic regression
        Scalar regularization_penalty_;

        // Number of documents processed so far
        int docs_;
        MatrixX phi_scaled;
        MatrixX expected_z_bar_;
        std::vector<MatrixX> variance_z_bar_;
        VectorXi y_;
};

}  // namespace em
}  // namespace ldaplusplus

#endif  // _SECOND_ORDER_SUPERVISED_M_STEP_HPP_
