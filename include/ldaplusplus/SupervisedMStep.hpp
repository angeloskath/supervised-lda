#ifndef _SUPERVISEDMSTEP_HPP
#define _SUPERVISEDMSTEP_HPP

#include "ldaplusplus/UnsupervisedMStep.hpp"

namespace ldaplusplus {


/**
 * Implement the M step for the categorical supervised LDA.
 *
 * Similarly to the UnsupervisedMStep the purpose is to maximize the lower
 * bound of the log likelihood \f$\mathcal{L}\f$. The same notation as in
 * UnsupervisedMStep is used.
 *
 * \f[
 *     \log p(w, y \mid \alpha, \beta, \eta) \geq
 *         \mathcal{L}(\gamma, \phi \mid \alpha, \beta, \eta) =
 *         \mathbb{E}_q[\log p(\theta \mid \alpha)] + \mathbb{E}_q[\log p(z \mid \theta)] +
 *         \mathbb{E}_q[\log p(w \mid z, \beta)] +
 *         H(q) + \mathbb{E}_q[\log p(y \mid z, \eta)]
 * \f]
 *
 * We observe that with respect to the parameter \f$\beta\f$ nothing changes
 * thus SupervisedMStep extends UnsupervisedMStep to delegate part of the
 * maximization to it. Decoration or another type of composition may be a more
 * appropriate form of code reuse in this case.
 *
 * To maximize with respect to \f$\eta\f$ we use the following equation which
 * amounts to simple logistic regression. The reasons for this approximation
 * are explained in our ACM MM '16 paper (to be linked when published).
 *
 * \f[
 *     \mathcal{L}_{\eta} = \sum_d^D \eta_{y_d}^T \left(\frac{1}{N} \sum_n^{N_d} \phi_{dn}\right) -
 *         \sum_d^D \log \sum_{\hat y}^C \text{exp}\left(
 *             \eta_{\hat y}^T \left(\frac{1}{N} \sum_n^{N_d} \phi_{dn}\right)
 *         \right)
 * \f]
 *
 * This implementation maximizes the above equation using batch gradient
 * descent with ArmijoLineSearch.
 */
template <typename Scalar>
class SupervisedMStep : public UnsupervisedMStep<Scalar>
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
        SupervisedMStep(
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
         * UnsupervisedMStep and keep in memory \f$\mathbb{E}_q[\bar z_d] =
         * \frac{1}{N} \sum_n^{N_d} \phi_{dn}\f$ for use in m_step().
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
        // The convergence tolerance for the maximization of the ELBO w.r.t.
        // eta in M-step
        Scalar m_step_tolerance_;
        // The regularization penalty for the multinomial logistic regression
        Scalar regularization_penalty_;

        // Number of documents processed so far
        int docs_;
        MatrixX expected_z_bar_;
        VectorXi y_;
};
}
#endif  // _SUPERVISEDMSTEP_HPP
