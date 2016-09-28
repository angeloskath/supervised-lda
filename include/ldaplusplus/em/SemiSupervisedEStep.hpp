#ifndef _LDAPLUSPLUS_EM_SEMISUPERVISEDESTEP_HPP_
#define _LDAPLUSPLUS_EM_SEMISUPERVISEDESTEP_HPP_


#include <memory>

#include "ldaplusplus/em/EStepInterface.hpp"

namespace ldaplusplus {
namespace em {


/**
 * SemiSupervisedEStep passes a document to either a supervised step or an
 * unsupervised step based on whether there exists class information for a
 * given document.
 *
 * SemiSupervisedEStep emits all the events emitted from the supervised or
 * unsupervised expectation steps so one has to subscribe only to the
 * SemiSupervisedEStep and not the steps passed in as constructor parameters.
 */
template <typename Scalar>
class SemiSupervisedEStep : public EStepInterface<Scalar>
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;

    public:
        /**
         * @param supervised_step   A pointer to a supervised expectation step
         * @param unsupervised_step A pointer to an unsupervised expectation
         *                          step
         */
        SemiSupervisedEStep(
            std::shared_ptr<EStepInterface<Scalar> > supervised_step,
            std::shared_ptr<EStepInterface<Scalar> > unsupervised_step
        );
        virtual ~SemiSupervisedEStep();

        /**
         * If the class from the document is less than 0 then the pass the
         * document to the unsupervised step otherwise pass it to the
         * supervised.
         */
        std::shared_ptr<parameters::Parameters> doc_e_step(
            const std::shared_ptr<corpus::Document> doc,
            const std::shared_ptr<parameters::Parameters> parameters
        ) override;

        /**
         * Inform both the sub e steps that an epoch has finished.
         */
        void e_step() override;

    private:
        std::shared_ptr<EStepInterface<Scalar> > supervised_step_;
        std::shared_ptr<EStepInterface<Scalar> > unsupervised_step_;

        std::shared_ptr<events::EventListenerInterface> event_forwarder_;
};

}  // namespace em
}  // namespace ldaplusplus

#endif  // _LDAPLUSPLUS_EM_SEMISUPERVISEDESTEP_HPP_
