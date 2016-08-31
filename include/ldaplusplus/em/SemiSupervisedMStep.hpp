#ifndef _SEMISUPERVISEDMSTEP_HPP_
#define _SEMISUPERVISEDMSTEP_HPP_


#include "ldaplusplus/em/SupervisedMStep.hpp"

namespace ldaplusplus {
namespace em {


/**
 * SemiSupervisedMStep passes the documents to either
 * SupervisedMStep::doc_m_step or UnsupervisedMStep::doc_m_step depending on
 * whether the document's class is a non negative integer.
 *
 * The bad choice of inheritance over composition is also evident in this
 * implementation although it does result in the minimum code written to
 * implement SemiSupervisedMStep.
 */
template <typename Scalar>
class SemiSupervisedMStep : public SupervisedMStep<Scalar>
{
    public:
        /**
         * @param m_step_iterations      The maximum number of gradient descent
         *                               iterations
         * @param m_step_tolerance       The minimum relative improvement between
         *                               consecutive gradient descent iterations
         * @param regularization_penalty The L2 penalty for logistic regression
         */
        SemiSupervisedMStep(
            size_t m_step_iterations = 10,
            Scalar m_step_tolerance = 1e-2,
            Scalar regularization_penalty = 1e-2
        ) : SupervisedMStep<Scalar>(
                m_step_iterations,
                m_step_tolerance,
                regularization_penalty
            )
        {}

        /**
         * Delegate to either SupervisedMStep or UnsupervisedMStep based on
         * whether the document has a class.
         *
         * @param doc              A single document
         * @param v_parameters     The variational parameters used in m-step
         *                         in order to maximize model parameters
         * @param m_parameters     Model parameters, used as output in case of 
         *                         online methods
         */
        virtual void doc_m_step(
            const std::shared_ptr<corpus::Document> doc,
            const std::shared_ptr<parameters::Parameters> v_parameters,
            std::shared_ptr<parameters::Parameters> m_parameters
        ) override;
};

}  // namespace em
}  // namespace ldaplusplus

#endif  // _SEMISUPERVISEDMSTEP_HPP_
