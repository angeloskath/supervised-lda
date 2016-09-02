#ifndef _CORRESPONDENCE_SUPERVISED_M_STEP_HPP_
#define _CORRESPONDENCE_SUPERVISED_M_STEP_HPP_

#include "ldaplusplus/em/IMStep.hpp"

namespace ldaplusplus {
namespace em {


/**
 * CorrespondenceSupervisedMStep implements the maximization step of a variant
 * of the correspondence LDA model.
 *
 * This model implements the correspondence LDA but instead of labels it tries
 * to also generate the class of the document. The generative procedure is the
 * following:
 *
 * 1. Given \f$K\f$ \f$V\f$-dimensional multinomials as the topics and \f$K\f$
 *    \f$C\f$-dimensional multinomials to sample class labels from
 * 2. Sample from a Dirichlet the topic mixture from a document
 * 3. For N times
 *    1. Sample a topic \f$k\f$
 *    2. From that topic sample a word using the \f$k\f$-th \f$V\f$-dimensional
 *       multinomial
 * 4. Sample from a uniform distribution a number \f$n\f$ between 1 and N
 * 5. Using the previously sampled \f$n\f$-th topic sample a class label from
 *    the multinomials
 */
template <typename Scalar>
class CorrespondenceSupervisedMStep : public IMStep<Scalar>
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;
    
    public:
        CorrespondenceSupervisedMStep(Scalar mu = 2.)
            : mu_(mu)
        {}

        /**
         * Maximize the ELBO w.r.t. to \f$\beta\f$ and \f$\eta\f$ (the topics
         * and class generating multinomials).
         *
         * @param parameters Model parameters (maybe changed after call)
         */
        virtual void m_step(
            std::shared_ptr<parameters::Parameters> parameters
        ) override;

        /**
         * Count the occurences of every word and class to implement maximum
         * likelihood estimation in the m_step()
         *
         * @param doc          A single document
         * @param v_parameters The variational parameters used in m-step
         *                     in order to maximize model parameters
         * @param m_parameters Model parameters, used as output in case of 
         *                     online methods
         */
        virtual void doc_m_step(
            const std::shared_ptr<corpus::Document> doc,
            const std::shared_ptr<parameters::Parameters> v_parameters,
            std::shared_ptr<parameters::Parameters> m_parameters
        ) override;

    private:
        MatrixX phi_scaled_;
        VectorX phi_scaled_sum_;
        MatrixX b_;
        MatrixX h_;
        Scalar mu_;

        Scalar log_py_;
};

}  // namespace em
}  // namespace ldaplusplus

#endif  // _CORRESPONDENCE_SUPERVISED_M_STEP_HPP_
