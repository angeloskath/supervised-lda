#ifndef _LDA_HPP_
#define _LDA_HPP_


#include <condition_variable>
#include <list>
#include <memory>
#include <mutex>
#include <vector>
#include <thread>
#include <tuple>

#include <Eigen/Core>

#include "Events.hpp"
#include "IEStep.hpp"
#include "IMStep.hpp"
#include "Parameters.hpp"

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
        /**
         * Create an LDA from a set of internal EM steps an initialization
         * strategy and an unsupervised expectation step to transform unseen
         * data.
         */
        LDA(
            std::shared_ptr<Parameters> model_parameters,
            std::shared_ptr<IEStep<Scalar> > e_step,
            std::shared_ptr<IMStep<Scalar> > m_step,
            size_t iterations = 20,
            size_t workers = 1
        );

        /**
         * Create a move constructor that doesn't try to copy or move the
         * mutexes.
         */
        LDA(LDA &&lda);

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
         * Perform a single em iteration.
         *
         * @param corpus The implementation of Corpus that contains the
         *               observed variables.
         */
        void partial_fit(std::shared_ptr<Corpus> corpus);

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
         * Return both the class predictions and the transformed data using a
         * single LDA expectation step.
         */
        std::tuple<MatrixX, VectorXi> transform_predict(const MatrixXi &X);

        /**
         * Get the progress visitor for this lda instance.
         */
        std::shared_ptr<IEventDispatcher> get_event_dispatcher() {
            return event_dispatcher_;
        }

        void process_worker_events() {
            std::static_pointer_cast<ThreadSafeEventDispatcher>(
                event_dispatcher_
            )->process_events();
        }

        /**
         * Get a constant reference to the model's parameters.
         */
        const std::shared_ptr<Parameters> model_parameters() {
            return model_parameters_;
        }

    protected:
        /**
         * Generate a Corpus from a pair of X, y matrices
         */
        std::shared_ptr<Corpus> get_corpus(
            const MatrixXi &X,
            const VectorXi &y
        );

        /**
         * Generate a Corpus from just the word count matrix.
         */
        std::shared_ptr<Corpus> get_corpus(const MatrixXi &X);

        /**
         * Create a worker thread pool.
         */
        void create_worker_pool();

        /**
         * Destroy the worker thread pool
         */
        void destroy_worker_pool();

        /**
         * Extract the variational parameters and the document index.
         */
        std::tuple<std::shared_ptr<Parameters>, size_t> extract_vp_from_queue();

        /**
         * A doc_e_step worker thread
         */
        void doc_e_step_worker();

        /**
         * Implement the decision function using already transformed data.
         * Topic representations instead of BOW.
         */
        MatrixX decision_function(const MatrixX &X);

        /**
         * Transform the decision function to class predictions.
         */
        VectorXi predict(const MatrixX &scores);


    private:
        /**
         * Pass the event dispatcher down to the implementations so that they
         * can communicate with the outside world.
         */
        void set_up_event_dispatcher();

        // The model parameters
        std::shared_ptr<Parameters> model_parameters_;

        // The LDA implementation
        std::shared_ptr<IEStep<Scalar> > e_step_;
        std::shared_ptr<IMStep<Scalar> > m_step_;

        // Member variables that affect the behaviour of fit
        size_t iterations_;

        // The thread related member variables
        std::vector<std::thread> workers_;
        std::mutex queue_in_mutex_;
        std::list<std::tuple<std::shared_ptr<Corpus>, size_t> > queue_in_;
        std::mutex queue_out_mutex_;
        std::condition_variable queue_out_cv_;
        std::list<std::tuple<std::shared_ptr<Parameters>, size_t> > queue_out_;

        // An event dispatcher that we will use to communicate with the
        // external components
        std::shared_ptr<IEventDispatcher> event_dispatcher_;
};


#endif  // _LDA_HPP_
