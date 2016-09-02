#ifndef _MULTINOMIALSUPERVISEDESTEP_HPP_
#define _MULTINOMIALSUPERVISEDESTEP_HPP__

#include "ldaplusplus/em/UnsupervisedEStep.hpp"

namespace ldaplusplus {
namespace em {


template<typename Scalar>
class MultinomialSupervisedEStep: public UnsupervisedEStep<Scalar>
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;

    public:
        /**
         * @param e_step_iterations The max number of times to alternate
         *                          between maximizing for \f$\gamma\f$
         *                          and for \f$\phi\f$.
         * @param e_step_tolerance  The minimum relative change in the
         *                          likelihood of generating the document.
         * @param mu                The uniform Dirichlet prior of \f$\eta\f$,
         *                          practically is a smoothing parameter 
         *                          during the maximization of \f$\eta\f$.
         * @param eta_weight        A weighting parameter that either increases
         *                          or decreases the influence of the supervised part.
         */
        MultinomialSupervisedEStep(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-2,
            Scalar mu = 2,
            Scalar eta_weight = 1
        );

        /** Maximize the ELBO w.r.t. \f$\phi\f$ and \f$\gamma\f$.
         *
         * The following steps are the mathematics that are implemented where
         * \f$\beta\f$ are the over words topics distributions, \f$\alpha\f$ is
         * the Dirichlet prior, \f$\eta\f$ are the logistic regression
         * parameters, \f$i\f$ is the topic subscript, \f$n\f$ is the word
         * subscript, \f$\hat{y}\f$ is the class subscript, \f$y\f$ is the
         * document's class, \f$w_n\f$ is n-th word vocabulary index, \f$m \f$
         * is a weighting parameter used to adjust the influence of the
         * supervised part and finally \f$\Psi(\cdot)\f$ is the first
         * derivative of the \f$\log \Gamma\f$ function.
         *
         * 1. Repeat until convergence of \f$\gamma\f$.
         * 2. Compute \f$\phi_{ni} \propto \beta_{iw_n}\eta_{yi}^m\exp\left(
         *    \Psi(\gamma_i)\right)\f$
         * 3. Compute \f$\gamma_i = \alpha_i + \sum_n^N \phi_{ni} \f$
         *
         * @param doc        A single document.
         * @param parameters An instance of class Parameters, which
         *                   contains all necessary model parameters 
         *                   for e-step's implementation.
         * @return           The variational parameters for the current
         *                   model, after e-step is completed.
         */
        std::shared_ptr<parameters::Parameters> doc_e_step(
            const std::shared_ptr<corpus::Document> doc,
            const std::shared_ptr<parameters::Parameters> parameters
        ) override;

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

        // The maximum number of iterations in E-step.
        size_t e_step_iterations_;
        // The convergence tolerance for the maximazation of the ELBO w.r.t.
        // phi and gamma in E-step.
        Scalar e_step_tolerance_;
        // The Dirichlet prior for the class predicting parameters.
        Scalar mu_;
        // A weighting parameter that either increases or decreases the
        // influence of the supervised part.
        Scalar eta_weight_;
};

}  // namespace em
}  // namespace ldaplusplus

#endif   //  _MULTINOMIALSUPERVISEDESTEP_HPP_
