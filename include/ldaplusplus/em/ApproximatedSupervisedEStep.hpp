#ifndef _APPROXIMATEDSUPERVISEDESTEP_HPP_
#define _APPROXIMATEDSUPERVISEDESTEP_HPP_

#include "ldaplusplus/em/UnsupervisedEStep.hpp"

namespace ldaplusplus {
namespace em {


/**
 * ApproximatedSupervisedEStep implements the expectation step of fsLDA, as it
 * is explained in our ACM MM '16 paper (to be linked when published).
 * 
 * Similarly to all expectation steps (e.g. SupervisedEStep, UnsupervisedEStep)
 * we compute the values of variational parameters \f$\gamma\f$ and \f$\phi\f$
 * such that the likelihood of generating each document is maximized.
 */
template<typename Scalar>
class ApproximatedSupervisedEStep : public UnsupervisedEStep<Scalar>
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;

    public:
        /**
          * CWeightType defines the methods of updating hyperparameter
          * \f$\mathcal{C}\f$ between expectation steps.
          */
        enum CWeightType
        {
            /**
             * The hyperparameter \f$\mathcal{C}\f$ remains unchanged, between
             * the consecutive expectation steps.
             */ 
            Constant = 1,
            /**
             * The hyperparameter \f$\mathcal{C}\f$ is reduced exponentially,
             * namely in the ith e step we have \f$ C_i = C^i\f$, where \f$ C<1\f$.
             */ 
            ExponentialDecay
        };

        /**
         * @param e_step_iterations  The max number of times to alternate
         *                           between maximizing for \f$\gamma\f$
         *                           and for \f$\phi\f$.
         * @param e_step_tolerance   The minimum relative change in the
         *                           likelihood of generating the document.
         * @param C                  A hyperparameter used for the weighting
         *                           of the supervised component in the 
         *                           update rule of \f$\phi\f$ 
         *                           (see doc_e_step()).
         * @param weight_type        The method used to update the
         *                           hyperparameter C.
         * @param compute_likelihood A parameter used to indicate whether the
         *                           supervised likelihood will be computed
         *                           at the end of each expectation step
         *                           (in order to be reported).
         */
        ApproximatedSupervisedEStep(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-2,
            Scalar C = 1,
            CWeightType weight_type = CWeightType::Constant,
            bool compute_likelihood = true
        );

        /**
         *
         * Maximize the ELBO w.r.t. \f$\phi\f$ and \f$\gamma\f$.
         *
         * The following steps are the mathematics that are implemented where
         * \f$\beta\f$ are the over words topics distributions, \f$\eta\f$ are
         * the logistic regression parameters, \f$i\f$ is the topic subscript,
         * \f$n\f$ is the word subscript, \f$\hat{y}\f$ is the class subscript,
         * \f$y\f$ is the document's class, \f$w_n\f$ is nth word vocabulary
         * index, \f$\alpha\f$ is the Dirichlet prior and finally
         * \f$\Psi(\cdot)\f$ is the first derivative of the \f$\log \Gamma\f$
         * function.
         *
         * 1. Repeat until convergence of \f$\gamma\f$.
         * 2. Compute \f$ s = softmax\left( \frac{1}{N} \sum_{n=1}^N \phi_n,
         * \eta \right)\f$
         * 3. Compute \f$\phi_{ni} \propto \beta_{iw_n} \exp\left(
         *    \Psi(\gamma_i) +
         *    \frac{C}{max(\eta)} \left(\eta_{yi} - \sum_{\hat{y}=1}^C
         *    s_{\hat{y}}\eta_{yi} \right)
         *    \right)\f$
         * 4. Compute \f$\gamma_i = \alpha_i + \sum_n^N \phi_{ni} \f$
         *
         * @param doc        A single document.
         * @param parameters An instance of class Parameters, which
         *                   contains all necessary model parameters 
         *                   for e-step's implementation.
         * @return           The variational parameters for the current
         *                   model, after one expecation step is completed.
         */
        std::shared_ptr<parameters::Parameters> doc_e_step(
            const std::shared_ptr<corpus::Document> doc,
            const std::shared_ptr<parameters::Parameters> parameters
        ) override;

        /**
         * Count how many epochs have already passed, in order to suitably
         * adjust the value of \f$\mathcal{C}\f$ hyperparameter, when
         * CWeightType::ExponentialDecay method is selected.
         */
        void e_step() override;

    private:
        /**
         * Check for convergence based on the change of the variational
         * parameter \f$\gamma\f$.
         *
         * @param gamma_old The gamma of the previous iteration.
         * @param gamma     The gamma of this iteration.
         * @return          Whether the change is small enough to indicate
         *                  convergence.
         */
        bool converged(const VectorX & gamma_old, const VectorX & gamma);

        /**
         * Define the weighting parameter for the supervised part of
         * expectation step according to the selected CWeightType method.
         *
         * @return The value of the weighting parameter.
         */
        Scalar get_weight();

        // The maximum number of iterations in expecation step.
        size_t e_step_iterations_;
        // The convergence tolerance for the maximazation of the ELBO w.r.t.
        // \f$\phi\f$ and \f$\gamma\f$ in expecation step.
        Scalar e_step_tolerance_;
        // A parameter weighting the supervised component in the variational
        // distribution.
        Scalar C_;
        // A parameter that is used to indicate whether to compute or not the
        // supervised likelihood at the end of each expectation step.
        bool compute_likelihood_;
        // The method used to update parameter C between consecutive expectation
        // steps.
        CWeightType weight_type_;
        // The epochs seen so far.
        int epochs_;
};

}  // namespace em
}  // namespace ldaplusplus

#endif   // _APPROXIMATEDSUPERVISEDESTEP_HPP_
