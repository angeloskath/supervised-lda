#ifndef _LDA_BUILDER_HPP_
#define _LDA_BUILDER_HPP_


#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include <Eigen/Core>

#include "ldaplusplus/Document.hpp"
#include "ldaplusplus/em/ApproximatedSupervisedEStep.hpp"
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
         * @param e_step_iterations The maximum iterations for each document's
         *                          expectation step
         * @param e_step_tolerance  The minimum relative change in the ELBO
         *                          (less than that and we stop iterating)
         */
        std::shared_ptr<em::EStepInterface<Scalar> > get_classic_e_step(
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
        std::shared_ptr<em::EStepInterface<Scalar> > get_fast_classic_e_step(
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
        std::shared_ptr<em::EStepInterface<Scalar> > get_supervised_e_step(
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
        std::shared_ptr<em::EStepInterface<Scalar> > get_fast_supervised_e_step(
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
        std::shared_ptr<em::EStepInterface<Scalar> > get_multinomial_supervised_e_step(
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
        std::shared_ptr<em::EStepInterface<Scalar> > get_correspondence_supervised_e_step(
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
        LDABuilder & set_e(std::shared_ptr<em::EStepInterface<Scalar> > e_step) {
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
            return set_m(get_classic_m_step());
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
            return set_m(get_supervised_m_step(
                m_step_iterations,
                m_step_tolerance,
                regularization_penalty
            ));
        }

        /**
         * Create a SecondOrderSupervisedMStep.
         *
         * You can also see a description of the parameters at
         * SecondOrderSupervisedMStep::SecondOrderSupervisedMStep.
         *
         * @param m_step_iterations      The maximum number of gradient descent
         *                               iterations
         * @param m_step_tolerance       The minimum relative improvement in
         *                               the log likelihood between consecutive
         *                               gradient descent iterations
         * @param regularization_penalty The L2 penalty for logistic regression
         */
        std::shared_ptr<em::MStepInterface<Scalar> > get_second_order_supervised_m_step(
            size_t m_step_iterations = 10,
            Scalar m_step_tolerance = 1e-2,
            Scalar regularization_penalty = 1e-2
        );
        /**
         * See the corresponding get_*_m_step() method.
         */
        LDABuilder & set_second_order_supervised_m_step(
            size_t m_step_iterations = 10,
            Scalar m_step_tolerance = 1e-2,
            Scalar regularization_penalty = 1e-2
        ) {
            return set_m(get_second_order_supervised_m_step(
                m_step_iterations,
                m_step_tolerance,
                regularization_penalty
            ));
        }

        /**
         * Create an OnlineSupervisedMStep without specifying class weights.
         *
         * You can also see a description of the parameters at
         * OnlineSupervisedMStep::OnlineSupervisedMStep.
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
         *                                   of \f$\beta\f$
         */
        std::shared_ptr<em::MStepInterface<Scalar> > get_supervised_online_m_step(
            size_t num_classes,
            Scalar regularization_penalty = 1e-2,
            size_t minibatch_size = 128,
            Scalar eta_momentum = 0.9,
            Scalar eta_learning_rate = 0.01,
            Scalar beta_weight = 0.9
        );
        LDABuilder & set_supervised_online_m_step(
            size_t num_classes,
            Scalar regularization_penalty = 1e-2,
            size_t minibatch_size = 128,
            Scalar eta_momentum = 0.9,
            Scalar eta_learning_rate = 0.01,
            Scalar beta_weight = 0.9
        ) {
            return set_m(get_supervised_online_m_step(
                num_classes,
                regularization_penalty,
                minibatch_size,
                eta_momentum,
                eta_learning_rate,
                beta_weight
            ));
        }

        /**
         * Create an OnlineSupervisedMStep.
         *
         * You can also see a description of the parameters at
         * OnlineSupervisedMStep::OnlineSupervisedMStep.
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
        std::shared_ptr<em::MStepInterface<Scalar> > get_supervised_online_m_step(
            std::vector<Scalar> class_weights,
            Scalar regularization_penalty = 1e-2,
            size_t minibatch_size = 128,
            Scalar eta_momentum = 0.9,
            Scalar eta_learning_rate = 0.01,
            Scalar beta_weight = 0.9
        );
        LDABuilder & set_supervised_online_m_step(
            std::vector<Scalar> class_weights,
            Scalar regularization_penalty = 1e-2,
            size_t minibatch_size = 128,
            Scalar eta_momentum = 0.9,
            Scalar eta_learning_rate = 0.01,
            Scalar beta_weight = 0.9
        ) {
            return set_m(get_supervised_online_m_step(
                class_weights,
                regularization_penalty,
                minibatch_size,
                eta_momentum,
                eta_learning_rate,
                beta_weight
            ));
        }

        /**
         * Create an OnlineSupervisedMStep.
         *
         * You can also see a description of the parameters at
         * OnlineSupervisedMStep::OnlineSupervisedMStep.
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
        std::shared_ptr<em::MStepInterface<Scalar> > get_supervised_online_m_step(
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> class_weights,
            Scalar regularization_penalty = 1e-2,
            size_t minibatch_size = 128,
            Scalar eta_momentum = 0.9,
            Scalar eta_learning_rate = 0.01,
            Scalar beta_weight = 0.9
        );
        LDABuilder & set_supervised_online_m_step(
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> class_weights,
            Scalar regularization_penalty = 1e-2,
            size_t minibatch_size = 128,
            Scalar eta_momentum = 0.9,
            Scalar eta_learning_rate = 0.01,
            Scalar beta_weight = 0.9
        ) {
            return set_m(get_supervised_online_m_step(
                class_weights,
                regularization_penalty,
                minibatch_size,
                eta_momentum,
                eta_learning_rate,
                beta_weight
            ));
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
            return set_m(get_semi_supervised_m_step(
                m_step_iterations,
                m_step_tolerance,
                regularization_penalty
            ));
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
            return set_m(get_multinomial_supervised_m_step(mu));
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
            return set_m(get_correspondence_supervised_m_step(mu));
        }

        /**
         * Set a maximization step.
         *
         * Can be used in conjuction with the get_*_m_step() methods.
         */
        LDABuilder & set_m(std::shared_ptr<em::MStepInterface<Scalar> > m_step) {
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
         * Initialize the topics over words distributions as uniform
         * distributions.
         *
         * This initialization also initializes alpha as 1.0 / topics
         *
         * @param words  The number of distinct words in the vocabulary
         * @param topics The number of topics
         */
        LDABuilder & initialize_topics_uniform(
            size_t words,
            size_t topics
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
};


}
#endif  //_LDA_BUILDER_HPP_
