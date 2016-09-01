#ifndef _LDA_BUILDER_HPP_
#define _LDA_BUILDER_HPP_


#include <memory>
#include <stdexcept>
#include <thread>

#include <Eigen/Core>

#include "ldaplusplus/initialize.hpp"
#include "ldaplusplus/em/ApproximatedSupervisedEStep.hpp"
#include "ldaplusplus/em/CorrespondenceSupervisedEStep.hpp"
#include "ldaplusplus/em/CorrespondenceSupervisedMStep.hpp"
#include "ldaplusplus/em/FastUnsupervisedEStep.hpp"
#include "ldaplusplus/em/MultinomialSupervisedEStep.hpp"
#include "ldaplusplus/em/MultinomialSupervisedMStep.hpp"
#include "ldaplusplus/em/IEStep.hpp"
#include "ldaplusplus/em/IMStep.hpp"
#include "ldaplusplus/em/OnlineSupervisedMStep.hpp"
#include "ldaplusplus/em/SecondOrderSupervisedMStep.hpp"
#include "ldaplusplus/em/SemiSupervisedEStep.hpp"
#include "ldaplusplus/em/SemiSupervisedMStep.hpp"
#include "ldaplusplus/em/SupervisedEStep.hpp"
#include "ldaplusplus/em/SupervisedMStep.hpp"
#include "ldaplusplus/em/UnsupervisedEStep.hpp"
#include "ldaplusplus/em/UnsupervisedMStep.hpp"
#include "ldaplusplus/LDA.hpp"

namespace ldaplusplus {


/**
 * An ILDABuilder is an interface for any class that can be cast into an LDA
 * instance.
 */
template <typename Scalar>
class ILDABuilder
{
    public:
        virtual operator LDA<Scalar>() const = 0;
};


/**
 * The LDABuilder provides a simpler interface to build an LDA.
 *
 * The builder has the following three main responsibilities:
 *
 * 1. Create an expectation step
 * 2. Create a maximization step
 * 3. Create & Initialize the model parameters
 *
 * Examples:
 *
 * LDA<double> lda = LDABuilder<double>().
 *                      initialize_topics("random", X, 100);
 *
 * LDA<double> lda = LDABuilder<double>().
 *                      set_iterations(20).
 *                      set_e_step("classic").
 *                      set_m_step("supervised-batch").
 *                      initialize_topics("seeded", X, 100).
 *                      initialize_eta("zeros", X, y);
 *
 * LDA<double> lda = LDABuilder<double>().
 *                      set_e_step("classic").
 *                      set_m_step("supervised-batch").
 *                      initialize_topics_from_model(model).
 *                      initialize_eta_from_model(model);
 */
template <typename Scalar>
class LDABuilder : public ILDABuilder<Scalar>
{
    public:
        /**
         * Create a default builder that will create a simple unsupervised LDA.
         *
         * The default builder uses unsupervised expectation and maximization
         * steps with 20 iterations and as many workers as there are cpus
         * available.
         *
         * Before being usable the model parameters must be initialized.
         */
        LDABuilder()
            : iterations_(20),
              workers_(std::thread::hardware_concurrency()),
              e_step_(std::make_shared<em::UnsupervisedEStep<Scalar> >()),
              m_step_(std::make_shared<em::UnsupervisedMStep<Scalar> >()),
              model_parameters_(
                std::make_shared<parameters::SupervisedModelParameters<Scalar> >()
              )
        {}

        /** Choose a number of iterations see LDA::fit */
        LDABuilder & set_iterations(size_t iterations) {
            iterations_ = iterations;

            return *this;
        }
        /** Choose a number of workers for the expectation step */
        LDABuilder & set_workers(size_t workers) {
            workers_ = workers;

            return *this;
        }

        /** Get the classic unsupervised LDA expectation step
         * (UnsupervisedEStep) */
        template <typename ...Args>
        std::shared_ptr<em::IEStep<Scalar> > get_classic_e_step(Args... args) {
            return std::make_shared<em::UnsupervisedEStep<Scalar> >(args...);
        }
        /** Get the classic unsupervised LDA expectation step that doesn't
         * report log likelihood (FastUnsupervisedEStep) */
        template <typename ...Args>
        std::shared_ptr<em::IEStep<Scalar> > get_fast_classic_e_step(Args... args) {
            return std::make_shared<em::FastUnsupervisedEStep<Scalar> >(args...);
        }
        /** Get the supervised LDA (sLDA) expectation step (SupervisedEStep) */
        template <typename ...Args>
        std::shared_ptr<em::IEStep<Scalar> > get_supervised_e_step(Args... args) {
            return std::make_shared<em::SupervisedEStep<Scalar> >(args...);
        }
        /** Get the fast approximate supervised LDA (fsLDA) expectation step
         * (ApproximatedSupervisedEStep) */
        template <typename ...Args>
        std::shared_ptr<em::IEStep<Scalar> > get_fast_supervised_e_step(Args... args) {
            return std::make_shared<em::ApproximatedSupervisedEStep<Scalar> >(args...);
        }
        /** Get the SemiSupervisedEStep */
        template <typename ...Args>
        std::shared_ptr<em::IEStep<Scalar> > get_semi_supervised_e_step(Args... args) {
            return std::make_shared<em::SemiSupervisedEStep<Scalar> >(args...);
        }
        /** Get the MultinomialSupervisedEStep */
        template <typename ...Args>
        std::shared_ptr<em::IEStep<Scalar> > get_supervised_multinomial_e_step(Args... args) {
            return std::make_shared<em::MultinomialSupervisedEStep<Scalar> >(args...);
        }
        /** Get the CorrespondenceSupervisedEStep */
        template <typename ...Args>
        std::shared_ptr<em::IEStep<Scalar> > get_supervised_correspondence_e_step(Args... args) {
            return std::make_shared<em::CorrespondenceSupervisedEStep<Scalar> >(args...);
        }
        /**
         * Set an expectation step.
         *
         * This is meant to be used with all the methods get_*_e_step() as in
         * the following example.
         *
         *     auto builder = LDABuilder<double>();
         *     builder.set_e(builder.get_fast_classic_e_step());
         */
        LDABuilder & set_e(std::shared_ptr<em::IEStep<Scalar> > e_step) {
            e_step_ = e_step;
            return *this;
        }

        /** Set the maximization step to the classic unsupervised LDA M step
         * (UnsupervisedMStep) */
        template <typename ...Args>
        LDABuilder & set_batch_m_step(Args... args) {
            m_step_ = std::make_shared<em::UnsupervisedMStep<Scalar> >(args...);

            return *this;
        }
        /** Set the maximization step to the first order approximation
         * supervised (SupervisedMStep) */
        template <typename ...Args>
        LDABuilder & set_supervised_batch_m_step(Args... args) {
            m_step_ = std::make_shared<em::SupervisedMStep<Scalar> >(args...);

            return *this;
        }
        /** Set the maximization step to the second order approximation
         * supervised (SupervisedMStep) */
        template <typename ...Args>
        LDABuilder & set_second_order_supervised_batch_m_step(Args... args) {
            m_step_ = std::make_shared<em::SecondOrderSupervisedMStep<Scalar> >(args...);

            return *this;
        }
        /** Set the maximization step to an online variant that changes the
         * parameters many times in a single epoch (OnlineSupervisedMStep) */
        template <typename ...Args>
        LDABuilder & set_supervised_online_m_step(Args... args) {
            m_step_ = std::make_shared<em::OnlineSupervisedMStep<Scalar> >(args...);

            return *this;
        }
        /**
         * Set the maximization step to semi supervised (SemiSupervisedMStep)
         */
        template <typename ...Args>
        LDABuilder & set_semi_supervised_batch_m_step(Args... args) {
            m_step_ = std::make_shared<em::SemiSupervisedMStep<Scalar> >(args...);

            return *this;
        }
        /** Set the maximization step to MultinomialSupervisedMStep */
        template <typename ...Args>
        LDABuilder & set_supervised_multinomial_m_step(Args... args) {
            m_step_ = std::make_shared<em::MultinomialSupervisedMStep<Scalar> >(args...);

            return *this;
        }
        /** Set the maximization step to CorrespondenceSupervisedMStep */
        template <typename ...Args>
        LDABuilder & set_supervised_correspondence_m_step(Args... args) {
            m_step_ = std::make_shared<em::CorrespondenceSupervisedMStep<Scalar> >(args...);

            return *this;
        }

        /**
         * Initialize the topic over words distributions, for each topic choose
         * a distribution over the words.
         *
         * The available methods are 'seeded' and 'random' and the extra
         * parameters are the number of topics as a size_t and an integer as
         * random state.
         *
         * TODO: This should change as the set_*_m_step() methods.
         *
         * @param type A string that defines the initialization method.
         * @param X    The word counts for each document in column-major order
         * @param args The extra arguments needed for the initialization
         *             functions
         */
        template <typename ...Args>
        LDABuilder & initialize_topics(
            const std::string &type,
            const MatrixXi &X,
            Args... args
        ) {
            auto corpus = std::make_shared<corpus::EigenCorpus>(X);

            if (type == "seeded") {
                initialize_topics_seeded<Scalar>(model_parameters_, corpus, args...);
            }
            else if (type == "random") {
                initialize_topics_random<Scalar>(model_parameters_, corpus, args...);
            }
            else {
                throw std::invalid_argument(type + " is an unknown topic initialization method");
            }

            return *this;
        }

        /**
         * Initialize the supervised model parameters which generate the class
         * label (in the generative model).
         *
         * The initialization methods are 'zeros' and 'multinomial'. The extra
         * parameters needed are the number of topics as a size_t.
         *
         * TODO: This should change as the set_*_m_step() methods.
         *
         * @param type A string that defines the initialization method
         * @param X    The word counts for each document in column-major order
         * @param y    The index of the class that each document belongs to
         * @param args The extra arguments needed for the initialization
         *             functions
         */
        template <typename ...Args>
        LDABuilder & initialize_eta(
            const std::string &type,
            const MatrixXi &X,
            const VectorXi &y,
            Args... args
        ) {
            auto corpus = std::make_shared<corpus::EigenClassificationCorpus>(X, y);

            if (type == "zeros") {
                initialize_eta_zeros<Scalar>(model_parameters_, corpus, args...);
            }
            else if (type == "multinomial") {
                initialize_eta_multinomial<Scalar>(model_parameters_, corpus, args...);
            }
            else {
                throw std::invalid_argument(type + " is an unknown eta initialization method");
            }

            return *this;
        }

        /**
         * Initialize the topic distributions from another model.
         *
         * In practice copy \f$\beta\f$ and \f$\alpha\f$ from the passed in
         * model to a local copy.
         */
        LDABuilder & initialize_topics_from_model(
            std::shared_ptr<parameters::ModelParameters<Scalar> > model
        ) {
            model_parameters_->alpha = model->alpha;
            model_parameters_->beta = model->beta;

            return *this;
        }

        /**
         * Initialize the supervised model parameters from another model.
         *
         * In practice copy \f$\eta\f$ from the passed in model to a local
         * copy.
         */
        LDABuilder & initialize_eta_from_model(
            std::shared_ptr<parameters::SupervisedModelParameters<Scalar> > model
        ) {
            model_parameters_->eta = model->eta;

            return *this;
        }

        /**
         * Build a brand new LDA instance from the configuration of the
         * builder.
         *
         * Before returning it also checks a few things that would result in an
         * unusable LDA instance and throws a runtime_error. 
         */
        virtual operator LDA<Scalar>() const override {
            if (model_parameters_->beta.rows() == 0) {
                throw std::runtime_error("You need to call initialize_topics before "
                                         "creating an LDA from the builder.");
            }

            if (
                model_parameters_->beta.rows() != model_parameters_->eta.rows() &&
                model_parameters_->eta.rows() > 0
            ) {
                throw std::runtime_error("\\eta and \\beta should be "
                                         "initialized with the same number of "
                                         "topics");
            }

            return LDA<Scalar>(
                model_parameters_,
                e_step_,
                m_step_,
                iterations_,
                workers_
            );
        };

    private:
        // generic lda parameters
        size_t iterations_;
        size_t workers_;

        // implementations
        std::shared_ptr<em::IEStep<Scalar> > e_step_;
        std::shared_ptr<em::IMStep<Scalar> > m_step_;

        // the model parameters
        std::shared_ptr<parameters::SupervisedModelParameters<Scalar> > model_parameters_;
};


}
#endif  //_LDA_BUILDER_HPP_
