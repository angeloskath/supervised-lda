#ifndef _LDA_HPP_
#define _LDA_HPP_


#include <memory>
#include <vector>

#include "IInitialization.hpp"
#include "IEStep.hpp"
#include "IMStep.hpp"
#include "Events.hpp"


using namespace Eigen;


/**
 * LDA computes a multitude of lda models.
 */
template <typename Scalar>
class LDA
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;

    public:
        struct LDAState
        {
            int ids[5];
            std::vector<Scalar> parameters[5];
            VectorX * alpha;
            MatrixX * beta;
            MatrixX * eta;
        };

        /**
         * Create an LDA from a set of internal EM steps an initialization
         * strategy and an unsupervised expectation step to transform unseen
         * data.
         */
        LDA(
            std::shared_ptr<IInitialization<Scalar> > initialization,
            std::shared_ptr<IEStep<Scalar> > unsupervised_e_step,
            std::shared_ptr<IMStep<Scalar> > unsupervised_m_step,
            std::shared_ptr<IEStep<Scalar> > e_step,
            std::shared_ptr<IMStep<Scalar> > m_step,
            size_t iterations = 20
        );

        /**
         * Rebuild an LDA instance from an LDAState.
         */
        LDA(
            LDAState lda_state,
            size_t iterations = 20
        );

        /**
         * Compute a supervised topic model for word counts X and classes y.
         *
         * Perform as many em iterations as configured and stop when reaching
         * max_iter_ or any other stopping criterion.
         *
         * @param X The word counts in column-major order
         * @param y The classes as integers
         */
        void fit(const MatrixXi &X, const VectorXi &y);

        /**
         * Perform a single em iteration.
         *
         * @param X The word counts in column-major order
         * @param y The classes as integers
         */
        void partial_fit(const MatrixXi &X, const VectorXi &y);

        /**
         * Transform the word counts X into topic proportions and return a new
         * matrix.
         */
        MatrixX transform(const MatrixXi &X);

        /**
         * Return the class scores according to the model parameters.
         */
        MatrixX decision_function(const MatrixXi &X);

        /**
         * Use the model to predict the class indexes for the word counts X.
         */
        VectorXi predict(const MatrixXi &X);

        /**
         * Get the progress visitor for this lda instance.
         */
        std::shared_ptr<IEventDispatcher> get_event_dispatcher() {
            return event_dispatcher_;
        }

        LDAState get_state() {
            LDAState s;

            s.ids[0] = initialization_->get_id();
            s.parameters[0] = initialization_->get_parameters();
            s.ids[1] = unsupervised_e_step_->get_id();
            s.parameters[1] = unsupervised_e_step_->get_parameters();
            s.ids[2] = unsupervised_m_step_->get_id();
            s.parameters[2] = unsupervised_m_step_->get_parameters();
            s.ids[3] = e_step_->get_id();
            s.parameters[3] = e_step_->get_parameters();
            s.ids[4] = m_step_->get_id();
            s.parameters[4] = m_step_->get_parameters();

            s.alpha = &alpha_;
            s.beta = &beta_;
            s.eta = &eta_;

            return s;
        }

    private:
        /**
         * Pass the event dispatcher down to the implementations so that they
         * can communicate with the outside world.
         */
        void set_up_event_dispatcher();

        // The internal modules used for the implementation
        std::shared_ptr<IInitialization<Scalar> > initialization_;
        std::shared_ptr<IEStep<Scalar> > unsupervised_e_step_;
        std::shared_ptr<IMStep<Scalar> > unsupervised_m_step_;
        std::shared_ptr<IEStep<Scalar> > e_step_;
        std::shared_ptr<IMStep<Scalar> > m_step_;

        // The model parameters
        VectorX alpha_;
        MatrixX beta_;
        MatrixX eta_;

        // Member variables that affect the behaviour of fit
        size_t iterations_;

        // An event dispatcher that we will use to communicate with the
        // external components
        std::shared_ptr<IEventDispatcher> event_dispatcher_;
};


#endif  // _LDA_HPP_
