#ifndef _LDAPLUSPLUS_LDABUILDER_HPP_
#define _LDAPLUSPLUS_LDABUILDER_HPP_

#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include <Eigen/Core>

#include "ldaplusplus/Document.hpp"
#include "ldaplusplus/em/FastSupervisedEStep.hpp"
#include "ldaplusplus/em/EStepInterface.hpp"
#include "ldaplusplus/em/MStepInterface.hpp"
#include "ldaplusplus/LDA.hpp"

namespace ldaplusplus {


/**
 * An LDABuilderInterface is an interface for any class that can be cast into an LDA
 * instance.
 */
template <typename Scalar>
class LDABuilderInterface
{
    public:
        virtual operator LDA<Scalar>() const = 0;

    virtual ~LDABuilderInterface(){};
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
 *                      initialize_topics_seeded(X, 100);
 *
 * LDA<double> lda = LDABuilder<double>().
 *                      set_iterations(20).
 *                      set_classic_e_step().
 *                      set_supervised_m_step().
 *                      initialize_topics_seeded(X, 100).
 *                      initialize_eta_zeros(y.maxCoeff() + 1);
 *
 * LDA<double> lda = LDABuilder<double>().
 *                      set_classic_e_step(50, 1e-2).
 *                      set_supervised_m_step().
 *                      initialize_topics_from_model(model).
 *                      initialize_eta_from_model(model);
 */
template <typename Scalar>
class LDABuilder : public LDABuilderInterface<Scalar>
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;

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
         * @param e_step_iterations  The max number of times to alternate
         *                           between maximizing for \f$\gamma\f$ and
         *                           for \f$\phi\f$.
         * @param e_step_tolerance   The minimum relative change in the
         *                           variational parameter \f$\gamma\f$.
         * @param compute_likelihood The percentage of documents to compute
         *                           likelihood for (1.0 means compute for
         *                           every document)
         * @param random_state       An initial seed value for any random
         *                           numbers needed
         */
        std::shared_ptr<em::EStepInterface<Scalar> > get_classic_e_step(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-2,
            Scalar compute_likelihood = 1.0,
            int random_state = 0
        );
        /**
         * See the corresponding get_*_e_step() method.
         */
        LDABuilder & set_classic_e_step(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-2,
            Scalar compute_likelihood = 1.0,
            int random_state = 0
        ) {
            e_requires_eta_ = false;
            return set_e(get_classic_e_step(
                e_step_iterations,
                e_step_tolerance,
                compute_likelihood,
                random_state
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
        std::shared_ptr<em::EStepInterface<Scalar> > get_supervised_e_step(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-2,
            size_t fixed_point_iterations = 10,
            Scalar compute_likelihood = 1.0,
            int random_state = 0
        );
        /**
         * See the corresponding get_*_e_step() method.
         */
        LDABuilder & set_supervised_e_step(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-2,
            size_t fixed_point_iterations = 10,
            Scalar compute_likelihood = 1.0,
            int random_state = 0
        ) {
            set_e(get_supervised_e_step(
                e_step_iterations,
                e_step_tolerance,
                fixed_point_iterations,
                compute_likelihood,
                random_state
            ));
            e_requires_eta_ = true;
            return *this;
        }

        /**
         * Create a FastSupervisedEStep.
         *
         * You can also see a description of the parameters at
         * FastSupervisedEStep::FastSupervisedEStep.
         *
         * @param e_step_iterations  The maximum iterations for each
         *                           document's expectation step
         * @param e_step_tolerance   The minimum relative change in the
         *                           ELBO (less than that and we stop
         *                           iterating)
         * @param C                  Weight of the supervised part in the
         *                           inference (default: 1)
         * @param compute_likelihood The percentage of documents to compute
         *                           likelihood for (1.0 means compute for
         *                           every document)
         * @param random_state       An initial seed value for any random
         *                           numbers needed
         */
        std::shared_ptr<em::EStepInterface<Scalar> > get_fast_supervised_e_step(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-2,
            Scalar C = 1,
            Scalar compute_likelihood = 1.0,
            int random_state = 0
        );
        /**
         * See the corresponding get_*_e_step() method.
         */
        LDABuilder & set_fast_supervised_e_step(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-2,
            Scalar C = 1,
            Scalar compute_likelihood = 1.0,
            int random_state = 0
        ) {
            set_e(get_fast_supervised_e_step(
                e_step_iterations,
                e_step_tolerance,
                C,
                compute_likelihood,
                random_state
            ));
            e_requires_eta_ = true;
            return *this;
        }

        /**
         * Create a SemiSupervisedEStep.
         *
         * You can also see a description of the parameters at
         * SemiSupervisedEStep::SemiSupervisedEStep.
         *
         * @param supervised_step   The supervised step to use (when nullptr is
         *                          provided it defaults to
         *                          FastSupervisedEStep with default
         *                          parameters)
         * @param unsupervised_step The unsupervised step to use (when nullptr is
         *                          provided it defaults to
         *                          FastUnsupervisedEStep with default
         *                          parameters)
         */
        std::shared_ptr<em::EStepInterface<Scalar> > get_semi_supervised_e_step(
            std::shared_ptr<em::EStepInterface<Scalar> > supervised_step = nullptr,
            std::shared_ptr<em::EStepInterface<Scalar> > unsupervised_step = nullptr
        );
        /**
         * See the corresponding get_*_e_step() method.
         */
        LDABuilder & set_semi_supervised_e_step(
            std::shared_ptr<em::EStepInterface<Scalar> > supervised_step = nullptr,
            std::shared_ptr<em::EStepInterface<Scalar> > unsupervised_step = nullptr
        ) {
            set_e(get_semi_supervised_e_step(
                supervised_step,
                unsupervised_step
            ));
            e_requires_eta_ = true;
            return *this;
        }

        /**
         * Create an MultinomialSupervisedEStep.
         *
         * You can also see a description of the parameters at
         * MultinomialSupervisedEStep::MultinomialSupervisedEStep.
         *
         * @param e_step_iterations  The max number of times to alternate
         *                           between maximizing for \f$\gamma\f$ and
         *                           for \f$\phi\f$.
         * @param e_step_tolerance   The minimum relative change in the
         *                           variational parameter \f$\gamma\f$.
         * @param mu                 The uniform Dirichlet prior of \f$\eta\f$,
         *                           practically is a smoothing parameter 
         *                           during the maximization of \f$\eta\f$.
         * @param eta_weight         A weighting parameter that either
         *                           increases or decreases the influence of
         *                           the supervised part.
         * @param compute_likelihood The percentage of documents to compute
         *                           likelihood for (1.0 means compute for
         *                           every document)
         * @param random_state       An initial seed value for any random
         *                           numbers needed
         */
        std::shared_ptr<em::EStepInterface<Scalar> > get_multinomial_supervised_e_step(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-2,
            Scalar mu = 2,
            Scalar eta_weight = 1,
            Scalar compute_likelihood = 1.0,
            int random_state = 0
        );
        /**
         * See the corresponding get_*_e_step() method.
         */
        LDABuilder & set_multinomial_supervised_e_step(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-2,
            Scalar mu = 2,
            Scalar eta_weight = 1,
            Scalar compute_likelihood = 1.0,
            int random_state = 0
        ) {
            set_e(get_multinomial_supervised_e_step(
                e_step_iterations,
                e_step_tolerance,
                mu,
                eta_weight,
                compute_likelihood,
                random_state
            ));
            e_requires_eta_ = true;
            return *this;
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
        std::shared_ptr<em::EStepInterface<Scalar> > get_correspondence_supervised_e_step(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-2,
            Scalar mu = 2,
            Scalar compute_likelihood = 1.0,
            int random_state = 0
        );
        /**
         * See the corresponding get_*_e_step() method.
         */
        LDABuilder & set_correspondence_supervised_e_step(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-2,
            Scalar mu = 2,
            Scalar compute_likelihood = 1.0,
            int random_state = 0
        ) {
            set_e(get_correspondence_supervised_e_step(
                e_step_iterations,
                e_step_tolerance,
                mu,
                compute_likelihood,
                random_state
            ));
            e_requires_eta_ = true;
            return *this;
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
        LDABuilder & set_e(std::shared_ptr<em::EStepInterface<Scalar> > e_step) {
            e_requires_eta_ = false; // clear require eta because we do not know
                                     // this e_step
            e_step_ = e_step;
            return *this;
        }

        /**
         * Create an UnsupervisedMStep.
         */
        std::shared_ptr<em::MStepInterface<Scalar> > get_classic_m_step();
        /**
         * See the corresponding get_*_m_step() method.
         */
        LDABuilder & set_classic_m_step() {
            set_m(get_classic_m_step());
            m_requires_eta_ = false;
            return *this;
        }

        /**
         * Create a SupervisedMStep.
         *
         * You can also see a description of the parameters at
         * SupervisedMStep::SupervisedMStep.
         *
         * @param m_step_iterations      The maximum number of gradient descent
         *                               iterations
         * @param m_step_tolerance       The minimum relative improvement in
         *                               the log likelihood between consecutive
         *                               gradient descent iterations
         * @param regularization_penalty The L2 penalty for logistic regression
         */
        std::shared_ptr<em::MStepInterface<Scalar> > get_fast_supervised_m_step(
            size_t m_step_iterations = 10,
            Scalar m_step_tolerance = 1e-2,
            Scalar regularization_penalty = 1e-2
        );
        /**
         * See the corresponding get_*_m_step() method.
         */
        LDABuilder & set_fast_supervised_m_step(
            size_t m_step_iterations = 10,
            Scalar m_step_tolerance = 1e-2,
            Scalar regularization_penalty = 1e-2
        ) {
            set_m(get_fast_supervised_m_step(
                m_step_iterations,
                m_step_tolerance,
                regularization_penalty
            ));
            m_requires_eta_ = true;
            return *this;
        }

        /**
         * Create a SupervisedMStep.
         *
         * You can also see a description of the parameters at
         * SupervisedMStep::SupervisedMStep.
         *
         * @param m_step_iterations      The maximum number of gradient descent
         *                               iterations
         * @param m_step_tolerance       The minimum relative improvement in
         *                               the log likelihood between consecutive
         *                               gradient descent iterations
         * @param regularization_penalty The L2 penalty for logistic regression
         */
        std::shared_ptr<em::MStepInterface<Scalar> > get_supervised_m_step(
            size_t m_step_iterations = 10,
            Scalar m_step_tolerance = 1e-2,
            Scalar regularization_penalty = 1e-2
        );
        /**
         * See the corresponding get_*_m_step() method.
         */
        LDABuilder & set_supervised_m_step(
            size_t m_step_iterations = 10,
            Scalar m_step_tolerance = 1e-2,
            Scalar regularization_penalty = 1e-2
        ) {
            set_m(get_supervised_m_step(
                m_step_iterations,
                m_step_tolerance,
                regularization_penalty
            ));
            m_requires_eta_ = true;
            return *this;
        }

        /**
         * Create an FastOnlineSupervisedMStep without specifying class weights.
         *
         * You can also see a description of the parameters at
         * FastOnlineSupervisedMStep::FastOnlineSupervisedMStep.
         *
         * @param num_classes            The number of classes
         * @param regularization_penalty The L2 penalty for the logistic
         *                               regression
         * @param minibatch_size         After that many documents call
         *                               m_step()
         * @param eta_momentum           The momentum for the SGD update
         *                               of \f$\eta\f$
         * @param eta_learning_rate      The learning rate for the SGD
         *                               update of \f$\eta\f$
         * @param beta_weight            The weight for the online update
         *                               of \f$\beta\f$
         */
        std::shared_ptr<em::MStepInterface<Scalar> > get_fast_supervised_online_m_step(
            size_t num_classes,
            Scalar regularization_penalty = 1e-2,
            size_t minibatch_size = 128,
            Scalar eta_momentum = 0.9,
            Scalar eta_learning_rate = 0.01,
            Scalar beta_weight = 0.9
        );
        LDABuilder & set_fast_supervised_online_m_step(
            size_t num_classes,
            Scalar regularization_penalty = 1e-2,
            size_t minibatch_size = 128,
            Scalar eta_momentum = 0.9,
            Scalar eta_learning_rate = 0.01,
            Scalar beta_weight = 0.9
        ) {
            set_m(get_fast_supervised_online_m_step(
                num_classes,
                regularization_penalty,
                minibatch_size,
                eta_momentum,
                eta_learning_rate,
                beta_weight
            ));
            m_requires_eta_ = true;
            return *this;
        }

        /**
         * Create an FastOnlineSupervisedMStep.
         *
         * You can also see a description of the parameters at
         * FastOnlineSupervisedMStep::FastOnlineSupervisedMStep.
         *
         * @param class_weights          Weights to account for class
         *                               imbalance
         * @param regularization_penalty The L2 penalty for the logistic
         *                               regression
         * @param minibatch_size         After that many documents call
         *                               m_step()
         * @param eta_momentum           The momentum for the SGD update
         *                               of \f$\eta\f$
         * @param eta_learning_rate      The learning rate for the SGD
         *                               update of \f$\eta\f$
         * @param beta_weight            The weight for the online update
         *                                   of \f$\beta\f$
         */
        std::shared_ptr<em::MStepInterface<Scalar> > get_fast_supervised_online_m_step(
            std::vector<Scalar> class_weights,
            Scalar regularization_penalty = 1e-2,
            size_t minibatch_size = 128,
            Scalar eta_momentum = 0.9,
            Scalar eta_learning_rate = 0.01,
            Scalar beta_weight = 0.9
        );
        LDABuilder & set_fast_supervised_online_m_step(
            std::vector<Scalar> class_weights,
            Scalar regularization_penalty = 1e-2,
            size_t minibatch_size = 128,
            Scalar eta_momentum = 0.9,
            Scalar eta_learning_rate = 0.01,
            Scalar beta_weight = 0.9
        ) {
            set_m(get_fast_supervised_online_m_step(
                class_weights,
                regularization_penalty,
                minibatch_size,
                eta_momentum,
                eta_learning_rate,
                beta_weight
            ));
            m_requires_eta_ = true;
            return *this;
        }

        /**
         * Create an FastOnlineSupervisedMStep.
         *
         * You can also see a description of the parameters at
         * FastOnlineSupervisedMStep::FastOnlineSupervisedMStep.
         *
         * @param class_weights          Weights to account for class
         *                               imbalance
         * @param regularization_penalty The L2 penalty for the logistic
         *                               regression
         * @param minibatch_size         After that many documents call
         *                               m_step()
         * @param eta_momentum           The momentum for the SGD update
         *                               of \f$\eta\f$
         * @param eta_learning_rate      The learning rate for the SGD
         *                               update of \f$\eta\f$
         * @param beta_weight            The weight for the online update
         *                                   of \f$\beta\f$
         */
        std::shared_ptr<em::MStepInterface<Scalar> > get_fast_supervised_online_m_step(
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> class_weights,
            Scalar regularization_penalty = 1e-2,
            size_t minibatch_size = 128,
            Scalar eta_momentum = 0.9,
            Scalar eta_learning_rate = 0.01,
            Scalar beta_weight = 0.9
        );
        LDABuilder & set_fast_supervised_online_m_step(
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> class_weights,
            Scalar regularization_penalty = 1e-2,
            size_t minibatch_size = 128,
            Scalar eta_momentum = 0.9,
            Scalar eta_learning_rate = 0.01,
            Scalar beta_weight = 0.9
        ) {
            set_m(get_fast_supervised_online_m_step(
                class_weights,
                regularization_penalty,
                minibatch_size,
                eta_momentum,
                eta_learning_rate,
                beta_weight
            ));
            m_requires_eta_ = true;
            return *this;
        }

        /**
         * Create a SemiSupervisedMStep.
         *
         * You can also see a description of the parameters at
         * SemiSupervisedMStep::SemiSupervisedMStep.
         *
         * @param m_step_iterations      The maximum number of gradient descent
         *                               iterations
         * @param m_step_tolerance       The minimum relative improvement in
         *                               the log likelihood between consecutive
         *                               gradient descent iterations
         * @param regularization_penalty The L2 penalty for logistic regression
         */
        std::shared_ptr<em::MStepInterface<Scalar> > get_semi_supervised_m_step(
            size_t m_step_iterations = 10,
            Scalar m_step_tolerance = 1e-2,
            Scalar regularization_penalty = 1e-2
        );
        /**
         * See the corresponding get_*_m_step() method.
         */
        LDABuilder & set_semi_supervised_m_step(
            size_t m_step_iterations = 10,
            Scalar m_step_tolerance = 1e-2,
            Scalar regularization_penalty = 1e-2
        ) {
            set_m(get_semi_supervised_m_step(
                m_step_iterations,
                m_step_tolerance,
                regularization_penalty
            ));
            m_requires_eta_ = true;
            return *this;
        }

        /**
         * Create a MultinomialSupervisedMStep.
         *
         * You can also see a description of the parameters at
         * MultinomialSupervisedMStep::MultinomialSupervisedMStep.
         *
         * @param mu A uniform Dirichlet prior for the supervised parameters
         */
        std::shared_ptr<em::MStepInterface<Scalar> > get_multinomial_supervised_m_step(
            Scalar mu = 2.
        );
        /**
         * See the corresponding get_*_m_step() method.
         */
        LDABuilder & set_multinomial_supervised_m_step(
            Scalar mu = 2.
        ) {
            set_m(get_multinomial_supervised_m_step(mu));
            m_requires_eta_ = true;
            return *this;
        }

        /**
         * Create a CorrespondenceSupervisedMStep.
         *
         * You can also see a description of the parameters at
         * CorrespondenceSupervisedMStep::CorrespondenceSupervisedMStep.
         *
         * @param mu A uniform Dirichlet prior for the supervised parameters
         */
        std::shared_ptr<em::MStepInterface<Scalar> > get_correspondence_supervised_m_step(
            Scalar mu = 2.
        );
        /**
         * See the corresponding get_*_m_step() method.
         */
        LDABuilder & set_correspondence_supervised_m_step(
            Scalar mu = 2.
        ) {
            set_m(get_correspondence_supervised_m_step(mu));
            m_requires_eta_ = true;
            return *this;
        }

        /**
         * Set a maximization step.
         *
         * Can be used in conjuction with the get_*_m_step() methods.
         */
        LDABuilder & set_m(std::shared_ptr<em::MStepInterface<Scalar> > m_step) {
            m_requires_eta_ = false; // clear require eta because we do not know
                                     // this m_step
            m_step_ = m_step;
            return *this;
        }

        /**
         * Initialize the topic over words distributions by seeding them from
         * the passed in documents.
         *
         * 1. For each topic t
         * 3. For N times sample a document d
         * 4. Add the word distribution of d to topic t
         *
         * This initialization also initializes alpha as 1.0 / topics
         *
         * @param X            The word counts for each document
         * @param topics       The number of topics
         * @param N            The number of documents to use for seeding
         * @param random_state The initial state of the random number generator
         */
        LDABuilder & initialize_topics_seeded(
            const Eigen::MatrixXi &X,
            size_t topics,
            size_t N = 30,
            int random_state = 0
        );

        /**
         * Initialize the topic over words distributions by seeding them from
         * the passed in documents.
         *
         * 1. For each topic t
         * 3. For N times sample a document d
         * 4. Add the word distribution of d to topic t
         *
         * This initialization also initializes alpha as 1.0 / topics
         *
         * @param corpus       The word counts for each document
         * @param topics       The number of topics
         * @param N            The number of documents to use for seeding
         * @param random_state The initial state of the random number generator
         */
        LDABuilder & initialize_topics_seeded(
            std::shared_ptr<corpus::Corpus> corpus,
            size_t topics,
            size_t N = 30,
            int random_state = 0
        );

        /**
         * Initialize the topics over words distributions as random
         * distributions.
         *
         * This initialization also initializes alpha as 1.0 / topics
         *
         * @param words        The number of distinct words in the vocabulary
         * @param topics       The number of topics
         * @param random_state The initial state of the random number generator
         */
        LDABuilder & initialize_topics_random(
            size_t words,
            size_t topics,
            int random_state = 0
        );

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
         * Initialize the supervised model parameters which generate the class
         * label with zeros.
         *
         * @param num_classes The number of classes
         */
        LDABuilder & initialize_eta_zeros(size_t num_classes);

        /**
         * Initialize the supervised model parameters which generate the class
         * label with a uniform multinomial distribution.
         *
         * @param num_classes The number of classes
         */
        LDABuilder & initialize_eta_uniform(size_t num_classes);

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
                model_parameters_->eta.rows() == 0 &&
                (e_requires_eta_ || m_requires_eta_)
            ) {
                throw std::runtime_error("An E step or M step seems to be supervised "
                                         "yet you have not initialized eta. "
                                         "Call initialize_eta_*()");
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
        std::shared_ptr<em::EStepInterface<Scalar> > e_step_;
        std::shared_ptr<em::MStepInterface<Scalar> > m_step_;

        // the model parameters
        std::shared_ptr<parameters::SupervisedModelParameters<Scalar> > model_parameters_;

        // A flag to keep track of having set EM steps that require the eta
        // model parameters.
        bool e_requires_eta_;
        bool m_requires_eta_;
};


} // namespace ldaplusplus
#endif  // _LDAPLUSPLUS_LDABUILDER_HPP_
