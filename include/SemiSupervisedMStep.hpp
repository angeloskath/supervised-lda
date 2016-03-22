#ifndef _SEMISUPERVISEDMSTEP_HPP_
#define _SEMISUPERVISEDMSTEP_HPP_


#include "SupervisedMStep.hpp"


/**
 * SemiSupervisedMStep passes the documents to either supervised doc_m_step or
 * unsupervised doc_m_step depending on wether the document's class is a non
 * negative integer.
 *
 * TODO: The fact that to do this easily we extend SupervisedMStep is an ugly
 *       choice we probably need to fix ASAP.
 */
template <typename Scalar>
class SemiSupervisedMStep : public SupervisedMStep<Scalar>
{
    public:
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

        virtual void doc_m_step(
            const std::shared_ptr<Document> doc,
            const std::shared_ptr<Parameters> v_parameters,
            std::shared_ptr<Parameters> m_parameters
        ) override;
};

#endif  // _SEMISUPERVISEDMSTEP_HPP_
