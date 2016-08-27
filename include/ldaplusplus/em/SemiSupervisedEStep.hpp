#ifndef _SEMISUPERVISEDESTEP_HPP_
#define _SEMISUPERVISEDESTEP_HPP_


#include <memory>

#include "ldaplusplus/em/IEStep.hpp"

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
class SemiSupervisedEStep : public IEStep<Scalar>
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;

    public:
        /**
         * @param supervised_step   A pointer to a supervised expectation step
         * @param unsupervised_step A pointer to an unsupervised expectation
         *                          step
         */
        SemiSupervisedEStep(
            std::shared_ptr<IEStep<Scalar> > supervised_step,
            std::shared_ptr<IEStep<Scalar> > unsupervised_step
        );
        virtual ~SemiSupervisedEStep();

        /**
         * If the class from the document is less than 0 then the pass the
         * document to the unsupervised step otherwise pass it to the
         * supervised.
         */
        std::shared_ptr<Parameters> doc_e_step(
            const std::shared_ptr<Document> doc,
            const std::shared_ptr<Parameters> parameters
        ) override;

        /**
         * Inform both the sub e steps that an epoch has finished.
         */
        void e_step() override;

    private:
        std::shared_ptr<IEStep<Scalar> > supervised_step_;
        std::shared_ptr<IEStep<Scalar> > unsupervised_step_;

        std::shared_ptr<IEventListener> event_forwarder_;
};

}  // namespace em
}  // namespace ldaplusplus

#endif  // _SEMISUPERVISEDESTEP_HPP_
