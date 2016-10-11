#include "ldaplusplus/LDABuilder.hpp"
#include "ldaplusplus/em/CorrespondenceSupervisedEStep.hpp"
#include "ldaplusplus/em/CorrespondenceSupervisedMStep.hpp"
#include "ldaplusplus/em/FastOnlineSupervisedMStep.hpp"
#include "ldaplusplus/em/FastSupervisedMStep.hpp"
#include "ldaplusplus/em/MultinomialSupervisedEStep.hpp"
#include "ldaplusplus/em/MultinomialSupervisedMStep.hpp"
#include "ldaplusplus/em/SemiSupervisedEStep.hpp"
#include "ldaplusplus/em/SemiSupervisedMStep.hpp"
#include "ldaplusplus/em/SupervisedEStep.hpp"
#include "ldaplusplus/em/SupervisedMStep.hpp"
#include "ldaplusplus/em/UnsupervisedEStep.hpp"
#include "ldaplusplus/em/UnsupervisedMStep.hpp"
#include "ldaplusplus/utils.hpp"

namespace ldaplusplus {

template <typename Scalar>
LDABuilder<Scalar>::LDABuilder()
    : iterations_(20),
      workers_(std::thread::hardware_concurrency()),
      e_step_(std::make_shared<em::UnsupervisedEStep<Scalar> >()),
      m_step_(std::make_shared<em::UnsupervisedMStep<Scalar> >()),
      model_parameters_(
        std::make_shared<parameters::SupervisedModelParameters<Scalar> >()
      ),
      e_requires_eta_(false),
      m_requires_eta_(false)
{}

template <typename Scalar>
LDABuilder<Scalar> & LDABuilder<Scalar>::set_iterations(size_t iterations) {
    iterations_ = iterations;

    return *this;
}
template <typename Scalar>
LDABuilder<Scalar> & LDABuilder<Scalar>::set_workers(size_t workers) {
    workers_ = workers;

    return *this;
}

template <typename Scalar>
std::shared_ptr<em::EStepInterface<Scalar> > LDABuilder<Scalar>::get_classic_e_step(
    size_t e_step_iterations,
    Scalar e_step_tolerance,
    Scalar compute_likelihood,
    int random_state
) {
    return std::make_shared<em::UnsupervisedEStep<Scalar> >(
        e_step_iterations,
        e_step_tolerance,
        compute_likelihood,
        random_state
    );
}

template <typename Scalar>
std::shared_ptr<em::EStepInterface<Scalar> > LDABuilder<Scalar>::get_supervised_e_step(
    size_t e_step_iterations,
    Scalar e_step_tolerance,
    size_t fixed_point_iterations,
    Scalar compute_likelihood,
    int random_state
) {
    return std::make_shared<em::SupervisedEStep<Scalar> >(
        e_step_iterations,
        e_step_tolerance,
        fixed_point_iterations,
        compute_likelihood,
        random_state
    );
}

template <typename Scalar>
std::shared_ptr<em::EStepInterface<Scalar> > LDABuilder<Scalar>::get_fast_supervised_e_step(
    size_t e_step_iterations,
    Scalar e_step_tolerance,
    Scalar C,
    Scalar compute_likelihood,
    int random_state
) {
    return std::make_shared<em::FastSupervisedEStep<Scalar> >(
        e_step_iterations,
        e_step_tolerance,
        C,
        em::FastSupervisedEStep<Scalar>::CWeightType::Constant,
        compute_likelihood,
        random_state
    );
}

template <typename Scalar>
std::shared_ptr<em::EStepInterface<Scalar> > LDABuilder<Scalar>::get_semi_supervised_e_step(
    std::shared_ptr<em::EStepInterface<Scalar> > supervised_step,
    std::shared_ptr<em::EStepInterface<Scalar> > unsupervised_step
) {
    if (supervised_step == nullptr) {
        supervised_step = get_fast_supervised_e_step();
    }
    if (unsupervised_step == nullptr) {
        unsupervised_step = get_classic_e_step();
    }

    return std::make_shared<em::SemiSupervisedEStep<Scalar> >(
        supervised_step,
        unsupervised_step
    );
}

template <typename Scalar>
std::shared_ptr<em::EStepInterface<Scalar> > LDABuilder<Scalar>::get_multinomial_supervised_e_step(
    size_t e_step_iterations,
    Scalar e_step_tolerance,
    Scalar mu,
    Scalar eta_weight,
    Scalar compute_likelihood,
    int random_state
) {
    return std::make_shared<em::MultinomialSupervisedEStep<Scalar> >(
        e_step_iterations,
        e_step_tolerance,
        mu,
        eta_weight,
        compute_likelihood,
        random_state
    );
}

template <typename Scalar>
std::shared_ptr<em::EStepInterface<Scalar> > LDABuilder<Scalar>::get_correspondence_supervised_e_step(
    size_t e_step_iterations,
    Scalar e_step_tolerance,
    Scalar mu,
    Scalar compute_likelihood,
    int random_state
) {
    return std::make_shared<em::CorrespondenceSupervisedEStep<Scalar> >(
        e_step_iterations,
        e_step_tolerance,
        mu,
        compute_likelihood,
        random_state
    );
}

template <typename Scalar>
std::shared_ptr<em::MStepInterface<Scalar> > LDABuilder<Scalar>::get_classic_m_step() {
    return std::make_shared<em::UnsupervisedMStep<Scalar> >();
}

template <typename Scalar>
std::shared_ptr<em::MStepInterface<Scalar> > LDABuilder<Scalar>::get_fast_supervised_m_step(
    size_t m_step_iterations,
    Scalar m_step_tolerance,
    Scalar regularization_penalty
) {
    return std::make_shared<em::FastSupervisedMStep<Scalar> >(
        m_step_iterations,
        m_step_tolerance,
        regularization_penalty
    );
}

template <typename Scalar>
std::shared_ptr<em::MStepInterface<Scalar> > LDABuilder<Scalar>::get_supervised_m_step(
    size_t m_step_iterations,
    Scalar m_step_tolerance,
    Scalar regularization_penalty
) {
    return std::make_shared<em::SupervisedMStep<Scalar> >(
        m_step_iterations,
        m_step_tolerance,
        regularization_penalty
    );
}

template <typename Scalar>
std::shared_ptr<em::MStepInterface<Scalar> > LDABuilder<Scalar>::get_fast_supervised_online_m_step(
    size_t num_classes,
    Scalar regularization_penalty,
    size_t minibatch_size,
    Scalar eta_momentum,
    Scalar eta_learning_rate,
    Scalar beta_weight
) {
    return std::make_shared<em::FastOnlineSupervisedMStep<Scalar> >(
        num_classes,
        regularization_penalty,
        minibatch_size,
        eta_momentum,
        eta_learning_rate,
        beta_weight
    );
}

template <typename Scalar>
std::shared_ptr<em::MStepInterface<Scalar> > LDABuilder<Scalar>::get_fast_supervised_online_m_step(
    std::vector<Scalar> class_weights,
    Scalar regularization_penalty,
    size_t minibatch_size,
    Scalar eta_momentum,
    Scalar eta_learning_rate,
    Scalar beta_weight
) {
    // Construct an Eigen Matrix and copy the weights
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weights(class_weights.size());
    for (size_t i=0; i<class_weights.size(); i++) {
        weights[i] = class_weights[i];
    }

    return get_fast_supervised_online_m_step(
        weights,
        regularization_penalty,
        minibatch_size,
        eta_momentum,
        eta_learning_rate,
        beta_weight
    );
}

template <typename Scalar>
std::shared_ptr<em::MStepInterface<Scalar> > LDABuilder<Scalar>::get_fast_supervised_online_m_step(
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> class_weights,
    Scalar regularization_penalty,
    size_t minibatch_size,
    Scalar eta_momentum,
    Scalar eta_learning_rate,
    Scalar beta_weight
) {
    return std::make_shared<em::FastOnlineSupervisedMStep<Scalar> >(
        class_weights,
        regularization_penalty,
        minibatch_size,
        eta_momentum,
        eta_learning_rate,
        beta_weight
    );
}

template <typename Scalar>
std::shared_ptr<em::MStepInterface<Scalar> > LDABuilder<Scalar>::get_semi_supervised_m_step(
    size_t m_step_iterations,
    Scalar m_step_tolerance,
    Scalar regularization_penalty
) {
    return std::make_shared<em::SemiSupervisedMStep<Scalar> >(
        m_step_iterations,
        m_step_tolerance,
        regularization_penalty
    );
}

template <typename Scalar>
std::shared_ptr<em::MStepInterface<Scalar> > LDABuilder<Scalar>::get_multinomial_supervised_m_step(
    Scalar mu
) {
    return std::make_shared<em::MultinomialSupervisedMStep<Scalar> >(mu);
}

template <typename Scalar>
std::shared_ptr<em::MStepInterface<Scalar> > LDABuilder<Scalar>::get_correspondence_supervised_m_step(
    Scalar mu
) {
    return std::make_shared<em::CorrespondenceSupervisedMStep<Scalar> >(mu);
}

template <typename Scalar>
LDABuilder<Scalar> & LDABuilder<Scalar>::initialize_topics_seeded(
    const Eigen::MatrixXi &X,
    size_t topics,
    size_t N,
    int random_state
) {
    return initialize_topics_seeded(
        std::make_shared<corpus::EigenCorpus>(X),
        topics,
        N,
        random_state
    );
}

template <typename Scalar>
LDABuilder<Scalar> & LDABuilder<Scalar>::initialize_topics_seeded(
    std::shared_ptr<corpus::Corpus> corpus,
    size_t topics,
    size_t N,
    int random_state
) {
    // Initialize alpha as 1/topics
    model_parameters_->alpha = VectorX::Constant(topics, 1.0 / topics);
    
    // Initliaze beta with ones to implement add one smoothing
    model_parameters_->beta = MatrixX::Constant(
        topics,
        (corpus->at(0)->get_words()).rows(),
        1.0
    );
    
    std::mt19937 rng(random_state);
    std::uniform_int_distribution<> document(0, corpus->size()-1);

    // Initialize _beta
    for (size_t k=0; k<topics; k++) {
        // Choose randomly a bunch of documents to initialize beta
        for (size_t r=0; r<N; r++) {
            model_parameters_->beta.row(k) += corpus->at(document(rng))->get_words().cast<Scalar>().transpose();
        }
        model_parameters_->beta.row(k) = model_parameters_->beta.row(k) / model_parameters_->beta.row(k).sum();
    }

    return *this;
}

template <typename Scalar>
LDABuilder<Scalar> & LDABuilder<Scalar>::initialize_topics_random(
    size_t words,
    size_t topics,
    int random_state
) {
    // Initialize alpha as 1/topics
    model_parameters_->alpha = VectorX::Constant(topics, 1.0 / topics);

    // Allocate memory for beta
    model_parameters_->beta = MatrixX::Zero(topics, words);
    
    std::mt19937 rng(random_state);
    std::uniform_real_distribution<> dis(0, 1);
    for (size_t k=0; k<topics; k++) {
        for (size_t w=0; w<words; w++) {
            model_parameters_->beta(k, w) = dis(rng);
        }
    }

    // Normalize beta
    math_utils::normalize_rows(model_parameters_->beta);

    return *this;
}

template <typename Scalar>
LDABuilder<Scalar> & LDABuilder<Scalar>::initialize_eta_zeros(
    size_t num_classes
) {
    // Figure out the number of topics and throw if not initialized
    size_t topics = model_parameters_->beta.rows();
    if (topics <= 0) {
        throw std::runtime_error("You need to call initialize_topics_*() "
                                 "before initializing eta");
    }

    // Zero the supervised parameters
    model_parameters_->eta = MatrixX::Zero(
        topics,
        num_classes
    );

    return *this;
}

template <typename Scalar>
LDABuilder<Scalar> & LDABuilder<Scalar>::initialize_eta_uniform(
    size_t num_classes
) {
    // Figure out the number of topics and throw if not initialized
    size_t topics = model_parameters_->beta.rows();
    if (topics <= 0) {
        throw std::runtime_error("You need to call initialize_topics_*() "
                                 "before initializing eta");
    }

    // Zero the supervised parameters
    model_parameters_->eta = MatrixX::Constant(
        topics,
        num_classes,
        1.0 / num_classes
    );

    return *this;
}

// Just the template instantiations all the rest is defined in the headers.
template class LDABuilder<float>;
template class LDABuilder<double>;

}
