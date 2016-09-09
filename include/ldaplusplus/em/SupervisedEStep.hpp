#ifndef _SUPERVISEDESTEP_HPP_
#define _SUPERVISEDESTEP_HPP_

#include "ldaplusplus/em/AbstractEStep.hpp"

namespace ldaplusplus {
namespace em {


/**
 * SupervisedEStep implements the categorical supervised LDA expectation step.
 *
 * As in UnsupervisedEStep the parameters of a distribution (\f$\gamma\f$ and
 * \f$\phi\f$) are computed such that the likelihood of generating each
 * document is maximized. For all supervised LDA variants except for generating
 * the document the likelihood of generating the label is also maximized.
 *
 * The solution implemented in SupervisedEStep is the one presented in paper
 * [2].
 *
 * [1] Mcauliffe, Jon D., and David M. Blei. "Supervised topic models."
 *     Advances in neural information processing systems. 2008
 *
 * [2] Chong, Wang, David Blei, and Fei-Fei Li. "Simultaneous image
 *     classification and annotation." Computer Vision and Pattern Recognition,
 *     2009 CVPR 2009 IEEE Conference on. IEEE, 2009
 */
template <typename Scalar>
class SupervisedEStep : public AbstractEStep<Scalar>
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;

    public:
        /**
         * @param e_step_iterations      The max number of times to alternate
         *                               between maximizing for \f$\gamma\f$
         *                               and for \f$\phi\f$.
         * @param e_step_tolerance       The minimum relative change in the
         *                               variational parameter \f$\gamma\f$.
         * @param fixed_point_iterations The number of fixed point iterations
         *                               used in the maximization for
         *                               \f$\phi\f$.
         * @param compute_likelihood     The percentage of documents to compute
         *                               likelihood for (1.0 means compute for
         *                               every document)
         * @param random_state           An initial seed value for any random
         *                               numbers needed
         */
        SupervisedEStep(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-2,
            size_t fixed_point_iterations = 20,
            Scalar compute_likelihood = 1.0,
            int random_state = 0
        );

        /**
         *
         * Maximize the ELBO w.r.t. to \f$\phi\f$ and \f$\gamma\f$.
         *
         * The following steps are the mathematics that are implemented where
         * \f$\beta\f$ are the topics, \f$\eta\f$ are the logistic regression
         * parameters, \f$i\f$ is the topic subscript, \f$n\f$ is the word
         * subscript, \f$\hat{y}\f$ is the class subscript, \f$y\f$ is the
         * document's class, \f$w_n\f$ is n-th word vocabulary index,
         * \f$\alpha\f$ is the Dirichlet prior and finally \f$\Psi(\cdot)\f$ is
         * the first derivative of the \f$\log \Gamma\f$ function.
         *
         * 1. Repeat until convergence
         *    1. Repeat for `fixed_point_iterations`
         *    2. Compute \f$h\f$ such that \f$h^T \phi_n =
         *       \sum_{\hat y}^C \prod_n^N \left(
         *       \sum_i^K \phi_{ni} \exp(\frac{1}{N}\eta_{\hat{y}i})
         *       \right)\f$ and \f$h\f$ doesn't contain \f$\phi_n\f$
         *    3. \f$\phi_{ni} \propto \beta_{iw_n} \exp\left(
         *       \Psi(\gamma_i) +
         *       \frac{1}{N} \eta_{yi} +
         *       \frac{h_i}{h^T\phi_n^{\text{old}}}
         *       \right)\f$
         * 3. \f$\gamma_i = \alpha_i + \sum_n^N \phi_{ni} \f$
         *
         * @param doc        A single document
         * @param parameters An instance of class Parameters, which
         *                   contains all necessary model parameters 
         *                   for e-step's implementation
         * @return           The variational parameters for the current
         *                   model, after e-step is completed
         */
        std::shared_ptr<parameters::Parameters> doc_e_step(
            const std::shared_ptr<corpus::Document> doc,
            const std::shared_ptr<parameters::Parameters> parameters
        ) override;

    private:
        // The maximum number of iterations in E-step.
        size_t e_step_iterations_;
        // The maximum number of iterations while maximizing phi in E-step.
        size_t fixed_point_iterations_;
        // The convergence tolerance for the maximazation of the ELBO w.r.t.
        // phi and gamma in E-step.
        Scalar e_step_tolerance_;
        // Compute the likelihood of that many documents (pecentile)
        Scalar compute_likelihood_;
};

}  // namespace em
}  // namespace ldaplusplus

#endif  // _SUPERVISEDESTEP_HPP_
