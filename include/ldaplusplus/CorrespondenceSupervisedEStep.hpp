#ifndef _CORRESPONDENCESUPERVISEDESTEP_HPP_
#define _CORRESPONDENCESUPERVISEDESTEP_HPP__

#include "ldaplusplus/UnsupervisedEStep.hpp"

namespace ldaplusplus {


/** 
 * CorrespondenceSupervisedEStep implements the expectation step of a variant
 * of the correspondence LDA model as it was introduced in [1]. Iinstead of
 * trying to generate labels it tries to generate the class of the document.
 *
 * The generative process according to this model is summarized below:
 *
 * 1. Given \f$K\f$ \f$V\f$-dimensional multinomial distributions as the
 * topics (\f$ \beta\f$) and \f$K\f$ \f$C\f$-dimensional multinomial
 * distributions (\f$ \eta\f$) to sample the class labels from.
 * 2. For each of the D documents:
 *    1. Sample from a Dirichlet distribution with Dirichlet prior \f$ \alpha
 *    \f$ and create the topic distribution for document d, \f$ \theta_d \sim
 *    Dir\left(\alpha\right)\f$.
 *    2. For each of the \f$N\f$ words \f$w_n\f$:
 *       1. Sample a topic \f$ z_n \sim Mult\left( \theta \right)\f$
 *       2. From that topic sample a word using the \f$k\f$th
 *       \f$V\f$-dimensional multinomial distribution, namely \f$w_n \sim p(w
 *       \mid z_n, \beta)\f$
 *    3. From a uniform distribution sample a number \f$n\f$ between \f$1\f$
 *    and \f$N\f$, namely \f$ \lambda_d \sim Unif\left(1...N\right)\f$.
 *    4. Sample a class label for the \f$d \f$ document from a multinomial
 *    distribution, namely \f$ y_d \sim p\left( y \mid \lambda, z, \eta
 *    \right)\f$.
 *
 * Exact probabilistic inference for this model is intractable, thus we use
 * variational inference methods. The factorized distribution on the latent
 * variables is the following.
 *
 * \f$ q\left(\theta, z, \lambda \right) = q\left(\theta \mid
 * \gamma\right)\left( \prod_{n=1}^N q\left(z_n \mid \phi_n
 * \right)\right)q\left(\lambda \mid \tau \right)\f$
 *
 * [1] Blei, D.M. and Jordan, M.I., 2003, July. Modeling annotated data. In
 * Proceedings of the 26th annual international ACM SIGIR conference on
 * Research and development in informaion retrieval (pp. 127-134). ACM.
 */
template<typename Scalar>
class CorrespondenceSupervisedEStep: public UnsupervisedEStep<Scalar>
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;

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
         */
        CorrespondenceSupervisedEStep(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-2,
            Scalar mu = 2.
        );

        /** Maximize the ELBO w.r.t. \f$\phi\f$ and \f$\gamma\f$.
         *
         * The following steps are the mathematics that are implemented where
         * \f$\beta\f$ are the over words topics distributions, \f$\alpha\f$ is
         * the Dirichlet prior, \f$\eta\f$ are the logistic regression
         * parameters, \f$\tau\f$ are the \f$N\f$-dimensional mutltinomial
         * parameters, \f$i\f$ is the topic subscript, \f$n\f$ is the word
         * subscript, \f$\hat{y}\f$ is the class subscript, \f$y\f$ is the
         * document's class, \f$w_n\f$ is n-th word vocabulary index, and
         * finally \f$\Psi(\cdot)\f$ is the first derivative of the \f$\log
         * \Gamma\f$ function.
         *
         * 1. Repeat until convergence of \f$\gamma\f$.
         * 2. Compute \f$\phi_{ni} \propto \beta_{iw_n} \exp\left(
         *    \Psi(\gamma_i) + \tau_{ni} * \log\left( \eta_{yi} \right)
         *    \right)\f$
         * 3. Compute \f$\tau_{ni} \propto \exp\left( \sum_{i=1}^K \phi_{ni} *
         *    \log\left( \eta_{yi}\right)\right)\f$
         * 4. Compute \f$\gamma_i = \alpha_i + \sum_n^N \phi_{ni} \f$
         *
         * @param doc        A single document.
         * @param parameters An instance of class Parameters, which
         *                   contains all necessary model parameters 
         *                   for e-step's implementation.
         * @return           The variational parameters for the current
         *                   model, after e-step is completed.
         */
        std::shared_ptr<Parameters> doc_e_step(
            const std::shared_ptr<Document> doc,
            const std::shared_ptr<Parameters> parameters
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
};

}
#endif   //  _CORRESPONDENCESUPERVISEDESTEP_HPP_
