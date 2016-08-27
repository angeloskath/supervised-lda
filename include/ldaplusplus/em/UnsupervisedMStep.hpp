#ifndef _UNSUPERVISEDMSTEP_HPP_
#define _UNSUPERVISEDMSTEP_HPP_

#include "ldaplusplus/em/IMStep.hpp"

namespace ldaplusplus {
namespace em {


/**
 * Implement the M step for the traditional unsupervised LDA.
 *
 * The following equations are used to maximize the lower bound of the log
 * likelihood (\f$ \mathcal{L} \f$). \f$D\f$ is the number of documents,
 * \f$N_d\f$ is the number of words in the \f$d\f$-th document and \f$K\f$ is
 * the number of topics.
 *
 * \f{eqnarray*}{
 *     \log p(w \mid \alpha, \beta) \geq
 *         \mathcal{L}(\gamma, \phi \mid \alpha, \beta) &=&
 *         \mathbb{E}_q[\log p(\theta \mid \alpha)] +
 *         \mathbb{E}_q[\log p(z \mid \theta)] +
 *         \mathbb{E}_q[\log p(w \mid z, \beta)] +
 *         H(q) \\
 *     \mathcal{L}_{\beta} &=& \mathbb{E}_q[\log p(w \mid \beta)] =
 *         \sum_d^D \sum_n^{N_d} \sum_i^K \phi_{dni} \log \beta_{iw_n} \\
 *     \beta_{ij} &\propto& \sum_d^D \sum_n^{N_d} \begin{cases}
 *             \phi_{dni} & w_n = j \\
 *             0          & \text{otherwise}
 *         \end{cases}
 * \f}
 *
 * Since we are using the bag of words count vector in the implementation, the
 * exact equation implemented is the following if \f$X_{dj}\f$ is the number of
 * occurences of the vocabulary word \f$j\f$ in the document \f$d\f$.
 *
 * \f[
 *     \beta_{ij} \propto \sum_d^D \phi_{dji} X_{dj}
 * \f]
 */
template <typename Scalar>
class UnsupervisedMStep : public IMStep<Scalar>
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;

    public:
        UnsupervisedMStep() {}

        /**
         * Normalize the temporary variable aggregated in doc_m_step() and set
         * it to the model parameters.
         *
         * @param parameters Model parameters (changed after this method)
         */
        virtual void m_step(
            std::shared_ptr<Parameters> parameters
        ) override;

        /**
         * Compute the \f$ \sum_{d=1}^{\hat{d}}\phi_{dji} X_{dj}\f$, where \f$
         * \hat{d}\f$ is this document and save its value to a temporary
         * variable.
         *
         * @param doc              A single document
         * @param v_parameters     The variational parameters used in m-step
         *                         in order to maximize model parameters
         * @param m_parameters     Model parameters, used as output in case of 
         *                         online methods
         */
        virtual void doc_m_step(
            const std::shared_ptr<Document> doc,
            const std::shared_ptr<Parameters> v_parameters,
            std::shared_ptr<Parameters> m_parameters
        ) override;

    private:
        MatrixX b_;
};

}  // namespace em
}  // namespace ldaplusplus

#endif  // _UNSUPERVISEDMSTEP_HPP_ 
