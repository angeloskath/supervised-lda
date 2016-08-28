
#include <algorithm>
#include <numeric>
#include <random>
#include <utility>

#include "ldaplusplus/LDA.hpp"
#include "ldaplusplus/events/ProgressEvents.hpp"

namespace ldaplusplus {


template <typename Scalar>
LDA<Scalar>::LDA(
    std::shared_ptr<Parameters> model_parameters,
    std::shared_ptr<em::IEStep<Scalar> > e_step,
    std::shared_ptr<em::IMStep<Scalar> > m_step,
    size_t iterations,
    size_t workers
) : model_parameters_(model_parameters),
    e_step_(e_step),
    m_step_(m_step),
    iterations_(iterations),
    workers_(workers),
    event_dispatcher_(std::make_shared<events::SameThreadEventDispatcher>())
{
    set_up_event_dispatcher();
}

template <typename Scalar>
LDA<Scalar>::LDA(LDA<Scalar> &&lda)
    : model_parameters_(std::move(lda.model_parameters_)),
      e_step_(std::move(lda.e_step_)),
      m_step_(std::move(lda.m_step_)),
      iterations_(lda.iterations_),
      workers_(lda.workers_.size()),
      event_dispatcher_(std::move(lda.event_dispatcher_))
{}

template <typename Scalar>
void LDA<Scalar>::set_up_event_dispatcher() {
    auto event_dispatcher = get_event_dispatcher();

    e_step_->set_event_dispatcher(event_dispatcher);
    m_step_->set_event_dispatcher(event_dispatcher);
}


template <typename Scalar>
std::shared_ptr<corpus::Corpus> LDA<Scalar>::get_corpus(
    const MatrixXi &X,
    const VectorXi &y
) {
    return std::make_shared<corpus::EigenClassificationCorpus>(X, y);
}


template <typename Scalar>
std::shared_ptr<corpus::Corpus> LDA<Scalar>::get_corpus(const MatrixXi &X) {
    return std::make_shared<corpus::EigenCorpus>(X);
}


template <typename Scalar>
void LDA<Scalar>::fit(const MatrixXi &X, const VectorXi &y) {
    auto corpus = get_corpus(X, y);

    for (size_t i=0; i<iterations_; i++) {
        partial_fit(corpus);
    }
}


template <typename Scalar>
void LDA<Scalar>::partial_fit(const MatrixXi &X, const VectorXi &y) {
    partial_fit(get_corpus(X, y));
}


template <typename Scalar>
void LDA<Scalar>::partial_fit(std::shared_ptr<corpus::Corpus> corpus) {
    // Shuffle the documents for a randomized pass through
    corpus->shuffle();

    // Queue all the documents
    for (size_t i=0; i<corpus->size(); i++) {
        queue_in_.emplace_back(corpus, i);
    }

    // create the thread pool
    create_worker_pool();

    // Extract variational parameters and calculate the doc_m_step
    for (size_t i=0; i<corpus->size(); i++) {
        std::shared_ptr<Parameters> variational_parameters;
        size_t index;

        std::tie(variational_parameters, index) = extract_vp_from_queue();

        // tell the thread safe event dispatcher to process the events from the
        // workers
        process_worker_events();

        // perform the online part of m step
        m_step_->doc_m_step(
            corpus->at(index),
            variational_parameters,
            model_parameters_  // output
        );
    }

    // destroy the thread pool
    destroy_worker_pool();

    // Perform any corpuswise action related to e step
    e_step_->e_step();

    // perform the batch part of m step
    m_step_->m_step(
        model_parameters_  // output
    );

    // inform the world that the epoch is over
    get_event_dispatcher()->template dispatch<events::EpochProgressEvent<Scalar> >(model_parameters_);
}


template <typename Scalar>
typename LDA<Scalar>::MatrixX LDA<Scalar>::transform(const MatrixXi& X) {
    // cast the parameters to what is needed
    auto model = std::static_pointer_cast<ModelParameters<Scalar> >(
        model_parameters_
    );

    // make some room for the transformed data
    MatrixX gammas(model->beta.rows(), X.cols());

    // make a corpus to use
    auto corpus = get_corpus(X);

    // Queue all the documents
    for (size_t i=0; i<corpus->size(); i++) {
        queue_in_.emplace_back(corpus, i);
    }

    // create the thread pool
    create_worker_pool();

    // Extract variational parameters and calculate the doc_e_step
    for (size_t i=0; i<corpus->size(); i++) {
        std::shared_ptr<Parameters> vp;
        size_t index;

        std::tie(vp, index) = extract_vp_from_queue();
        gammas.col(index) = std::static_pointer_cast<VariationalParameters<Scalar> >(vp)->gamma;

        // tell the thread safe event dispatcher to process the events from the
        // workers
        process_worker_events();
    }

    // destroy the thread pool
    destroy_worker_pool();

    return gammas;
}


template <typename Scalar>
typename LDA<Scalar>::MatrixX LDA<Scalar>::decision_function(const MatrixXi &X) {
    return decision_function(transform(X));
}


template <typename Scalar>
typename LDA<Scalar>::MatrixX LDA<Scalar>::decision_function(const MatrixX &X) {
    // this function requires a supervised LDA so let's cast our models
    // parameters accordingly
    auto model = std::static_pointer_cast<SupervisedModelParameters<Scalar> >(
        model_parameters_
    );

    // the linear model is trained on
    // E_q[\bar z] = \fraction{\gamma - \alpha}{\sum_i \gamma_i}
    MatrixX expected_z_bar = X.colwise() - model->alpha;
    expected_z_bar.array().rowwise() /= expected_z_bar.array().colwise().sum();

    // finally return the linear scores like a boss
    return model->eta.transpose() * expected_z_bar;
}


template <typename Scalar>
VectorXi LDA<Scalar>::predict(const MatrixXi &X) {
    return predict(decision_function(X));
}


template <typename Scalar>
VectorXi LDA<Scalar>::predict(const MatrixX &scores) {
    VectorXi predictions(scores.cols());

    for (int d=0; d<scores.cols(); d++) {
        scores.col(d).maxCoeff( &predictions[d] );
    }

    return predictions;
}


template <typename Scalar>
std::tuple<typename LDA<Scalar>::MatrixX, VectorXi> LDA<Scalar>::transform_predict(
    const MatrixXi &X
) {
    MatrixX gammas = transform(X);
    VectorXi predictions = predict(decision_function(gammas));

    return std::make_tuple(gammas, predictions);
}


template <typename Scalar>
void LDA<Scalar>::create_worker_pool() {
    for (auto & t : workers_) {
        t = std::thread(
            std::bind(&LDA<Scalar>::doc_e_step_worker, this)
        );
    }
}


template <typename Scalar>
void LDA<Scalar>::destroy_worker_pool() {
    for (auto & t : workers_) {
        t.join();
    }
}


template <typename Scalar>
void LDA<Scalar>::doc_e_step_worker() {
    std::shared_ptr<corpus::Corpus> corpus;
    size_t index;

    while (true) {
        // extract a job
        {
            std::lock_guard<std::mutex> lock(queue_in_mutex_);
            if (queue_in_.empty())
                break;
            std::tie(corpus, index) = queue_in_.front();
            queue_in_.pop_front();
        }

        // do said job
        auto vp = e_step_->doc_e_step(
            corpus->at(index),
            model_parameters_
        );

        // show some results
        {
            std::lock_guard<std::mutex> lock(queue_out_mutex_);
            queue_out_.emplace_back(vp, index);
        }
        // talk about those results
        queue_out_cv_.notify_one();
    }
}


template <typename Scalar>
std::tuple<std::shared_ptr<Parameters>, size_t> LDA<Scalar>::extract_vp_from_queue() {
    std::unique_lock<std::mutex> lock(queue_out_mutex_);
    queue_out_cv_.wait(lock, [this](){ return !queue_out_.empty(); });

    auto f = queue_out_.front();
    queue_out_.pop_front();

    return f;
}


// Template instantiation
template class LDA<float>;
template class LDA<double>;

}
