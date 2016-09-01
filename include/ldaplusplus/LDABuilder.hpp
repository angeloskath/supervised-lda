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
         * Before being usable the model parameters must be initialized to set
         * the number of topics etc.
         */
        LDABuilder();

        /** Choose a number of iterations see LDA::fit */
        LDABuilder & set_iterations(size_t iterations);

        /** Choose a number of parallel workers for the expectation step */
        LDABuilder & set_workers(size_t workers);


        /**
         * Create an UnsupervisedEStep.
         *
         * You can also see a description of the parameters at
         * UnsupervisedEStep::UnsupervisedEStep
         *
         * @param e_step_iterations The maximum iterations for each document's
         *                          expectation step
         * @param e_step_tolerance  The minimum relative change in the ELBO
         *                          (less than that and we stop iterating)
         */
        std::shared_ptr<em::IEStep<Scalar> > get_classic_e_step(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-4
        );
        /**
         * See the corresponding get_*_e_step() method.
         */
        LDABuilder & set_classic_e_step(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-4
        ) {
            return set_e(get_classic_e_step(
                e_step_iterations,
                e_step_tolerance
            ));
        }

        /**
         * Create an FastUnsupervisedEStep.
         *
         * You can also see a description of the parameters at
         * FastUnsupervisedEStep::FastUnsupervisedEStep
         *
         * @param e_step_iterations The maximum iterations for each document's
         *                          expectation step
         * @param e_step_tolerance  The minimum relative change in the
         *                          variational parameters (less than that and
         *                          we stop iterating)
         */
        std::shared_ptr<em::IEStep<Scalar> > get_fast_classic_e_step(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-4
        );
        /**
         * See the corresponding get_*_e_step() method.
         */
        LDABuilder & set_fast_classic_e_step(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-4
        ) {
            return set_e(get_fast_classic_e_step(
                e_step_iterations,
                e_step_tolerance
            ));
        }

        /**
         * Create a SupervisedEStep.
         *
         * You can also see a description of the parameters at
         * SupervisedEStep::SupervisedEStep.
         *
         * @param e_step_iterations      The maximum iterations for each
         *                               document's expectation step
         * @param e_step_tolerance       The minimum relative change in the
         *                               ELBO (less than that and we stop
         *                               iterating)
         * @param fixed_point_iterations The number of fixed point iterations
         *                               see SupervisedEStep
         */
        std::shared_ptr<em::IEStep<Scalar> > get_supervised_e_step(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-2,
            size_t fixed_point_iterations = 10
        );
        /**
         * See the corresponding get_*_e_step() method.
         */
        LDABuilder & set_supervised_e_step(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-2,
            size_t fixed_point_iterations = 10
        ) {
            return set_e(get_supervised_e_step(
                e_step_iterations,
                e_step_tolerance,
                fixed_point_iterations
            ));
        }

        /**
         * Create an ApproximatedSupervisedEStep.
         *
         * You can also see a description of the parameters at
         * ApproximatedSupervisedEStep::ApproximatedSupervisedEStep.
         *
         * @param e_step_iterations  The maximum iterations for each
         *                           document's expectation step
         * @param e_step_tolerance   The minimum relative change in the
         *                           ELBO (less than that and we stop
         *                           iterating)
         * @param C                  Weight of the supervised part in the
         *                           inference (default: 1)
         * @param weight_type        How the weight will change between
         *                           iterations (default: constant)
         * @param compute_likelihood Compute the likelihood at the end of each
         *                           expectation step (in order to be reported)
         */
        std::shared_ptr<em::IEStep<Scalar> > get_fast_supervised_e_step(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-2,
            Scalar C = 1,
            typename em::ApproximatedSupervisedEStep<Scalar>::CWeightType weight_type =
                em::ApproximatedSupervisedEStep<Scalar>::CWeightType::Constant,
            bool compute_likelihood = true
        );
        /**
         * See the corresponding get_*_e_step() method.
         */
        LDABuilder & set_fast_supervised_e_step(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-2,
            Scalar C = 1,
            typename em::ApproximatedSupervisedEStep<Scalar>::CWeightType weight_type =
                em::ApproximatedSupervisedEStep<Scalar>::CWeightType::Constant,
            bool compute_likelihood = true
        ) {
            return set_e(get_fast_supervised_e_step(
                e_step_iterations,
                e_step_tolerance,
                C,
                weight_type,
                compute_likelihood
            ));
        }

        /**
         * Create a SemiSupervisedEStep.
         *
         * You can also see a description of the parameters at
         * SemiSupervisedEStep::SemiSupervisedEStep.
         *
         * @param supervised_step   The supervised step to use (when nullptr is
         *                          provided it defaults to
         *                          ApproximatedSupervisedEStep with default
         *                          parameters)
         * @param unsupervised_step The unsupervised step to use (when nullptr is
         *                          provided it defaults to
         *                          FastUnsupervisedEStep with default
         *                          parameters)
         */
        std::shared_ptr<em::IEStep<Scalar> > get_semi_supervised_e_step(
            std::shared_ptr<em::IEStep<Scalar> > supervised_step = nullptr,
            std::shared_ptr<em::IEStep<Scalar> > unsupervised_step = nullptr
        );
        /**
         * See the corresponding get_*_e_step() method.
         */
        LDABuilder & set_semi_supervised_e_step(
            std::shared_ptr<em::IEStep<Scalar> > supervised_step = nullptr,
            std::shared_ptr<em::IEStep<Scalar> > unsupervised_step = nullptr
        ) {
            return set_e(get_semi_supervised_e_step(
                supervised_step,
                unsupervised_step
            ));
        }

        /**
         * Create an MultinomialSupervisedEStep.
         *
         * You can also see a description of the parameters at
         * MultinomialSupervisedEStep::MultinomialSupervisedEStep.
         *
         * @param e_step_iterations The maximum iterations for each
         *                          document's expectation step
         * @param e_step_tolerance  The minimum relative change in the
         *                          ELBO (less than that and we stop
         *                          iterating)
         * @param mu                A uniform Dirichlet prior for the
         *                          supervised parameters (default: 2)
         * @param eta_weight        A weight parameter that decreases or
         *                          increases the influence of the supervised
         *                          part (default: 1).
         */
        std::shared_ptr<em::IEStep<Scalar> > get_multinomial_supervised_e_step(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-2,
            Scalar mu = 2,
            Scalar eta_weight = 1
        );
        /**
         * See the corresponding get_*_e_step() method.
         */
        LDABuilder & set_multinomial_supervised_e_step(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-2,
            Scalar mu = 2,
            Scalar eta_weight = 1
        ) {
            return set_e(get_multinomial_supervised_e_step(
                e_step_iterations,
                e_step_tolerance,
                mu,
                eta_weight
            ));
        }

        /**
         * Create an CorrespondenceSupervisedEStep.
         *
         * You can also see a description of the parameters at
         * CorrespondenceSupervisedEStep::CorrespondenceSupervisedEStep.
         *
         * @param e_step_iterations The maximum iterations for each
         *                          document's expectation step
         * @param e_step_tolerance  The minimum relative change in the
         *                          ELBO (less than that and we stop
         *                          iterating)
         * @param mu                A uniform Dirichlet prior for the
         *                          supervised parameters (default: 2)
         */
        std::shared_ptr<em::IEStep<Scalar> > get_correspondence_supervised_e_step(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-2,
            Scalar mu = 2
        );
        /**
         * See the corresponding get_*_e_step() method.
         */
        LDABuilder & set_correspondence_supervised_e_step(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-2,
            Scalar mu = 2
        ) {
            return set_e(get_correspondence_supervised_e_step(
                e_step_iterations,
                e_step_tolerance,
                mu
            ));
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
            const Eigen::MatrixXi &X,
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
            const Eigen::MatrixXi &X,
            const Eigen::VectorXi &y,
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
