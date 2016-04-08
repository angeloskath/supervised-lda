#ifndef _SEMISUPERVISEDESTEP_HPP_
#define _SEMISUPERVISEDESTEP_HPP_


#include <memory>

#include "IEStep.hpp"

/**
 * SemiSupervisedEStep passes a document to either a supervised step or an
 * unsupervised step based on whether there exists class information for a
 * given document.
 */
template <typename Scalar>
class SemiSupervisedEStep : public IEStep<Scalar>
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;

    public:
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

#endif  // _SEMISUPERVISEDESTEP_HPP_
