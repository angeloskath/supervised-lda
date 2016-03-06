
#include <algorithm>
#include <numeric>
#include <random>

#include "LDA.hpp"
#include "InternalsFactory.hpp"
#include "ProgressEvents.hpp"


template <typename Scalar>
LDA<Scalar>::LDA(
    std::shared_ptr<Parameters> model_parameters,
    std::shared_ptr<IEStep<Scalar> > e_step,
    std::shared_ptr<IMStep<Scalar> > m_step,
    size_t iterations
) : model_parameters_(model_parameters),
    e_step_(e_step),
    m_step_(m_step),
    iterations_(iterations),
    event_dispatcher_(std::make_shared<EventDispatcher>())
{
    set_up_event_dispatcher();
}

template <typename Scalar>
void LDA<Scalar>::set_up_event_dispatcher() {
    auto event_dispatcher = get_event_dispatcher();

    e_step_->set_event_dispatcher(event_dispatcher);
    m_step_->set_event_dispatcher(event_dispatcher);
}


template <typename Scalar>
std::shared_ptr<Corpus> LDA<Scalar>::get_corpus(
    const MatrixXi &X,
    const VectorXi &y
) {
    return std::make_shared<EigenClassificationCorpus>(X, y);
}


template <typename Scalar>
std::shared_ptr<Corpus> LDA<Scalar>::get_corpus(const MatrixXi &X) {
    return std::make_shared<EigenCorpus>(X);
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
void LDA<Scalar>::partial_fit(std::shared_ptr<Corpus> corpus) {
    // Shuffle the documents for a randomized pass through
    corpus->shuffle();

    // For each document
    for (size_t i=0; i<corpus->size(); i++) {
        // perform the expectation step
        auto variational_parameters = e_step_->doc_e_step(
            corpus->at(i),
            model_parameters_
        );

        // perform the online part of m step
        m_step_->doc_m_step(
            corpus->at(i),
            variational_parameters,
            model_parameters_  // output
        );

        // inform the world that the iterations are moving
        get_event_dispatcher()->template dispatch<ExpectationProgressEvent<Scalar> >(
            i,
            0
        );
    }

    // perform the batch part of m step
    m_step_->m_step(
        model_parameters_  // output
    );

    // inform the world that the epoch is over
    get_event_dispatcher()->template dispatch<EpochProgressEvent<Scalar> >(model_parameters_);
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

    for (size_t i=0; i<corpus->size(); i++) {
        // perform the e step
        auto variational_parameters =
            std::static_pointer_cast<VariationalParameters<Scalar> >(
                e_step_->doc_e_step(
                    corpus->at(i),
                    model_parameters_
                )
            );

        // fill in the gammas array
        gammas.col(i) = variational_parameters->gamma;

        // and inform the world about us finishing another document
        get_event_dispatcher()->template dispatch<ExpectationProgressEvent<Scalar> >(
            i,
            0
        );
    }

    return gammas;
}


template <typename Scalar>
typename LDA<Scalar>::MatrixX LDA<Scalar>::decision_function(const MatrixXi &X) {
    // this function requires a supervised LDA so let's cast our models
    // parameters accordingly
    auto model = std::static_pointer_cast<SupervisedModelParameters<Scalar> >(
        model_parameters_
    );

    // get the representation
    MatrixX gammas = transform(X);

    // the linear model is trained on
    // E_q[\bar z] = \fraction{\gamma - \alpha}{\sum_i \gamma_i}
    MatrixX expected_z_bar = gammas.colwise() - model->alpha;
    expected_z_bar.array().rowwise() /= expected_z_bar.array().colwise().sum();

    // finally return the linear scores like a boss
    return model->eta.transpose() * expected_z_bar;
}


template <typename Scalar>
VectorXi LDA<Scalar>::predict(const MatrixXi &X) {
    VectorXi predictions(X.cols());
    MatrixX scores = decision_function(X);

    for (int d=0; d<X.cols(); d++) {
        scores.col(d).maxCoeff( &predictions[d] );
    }

    return predictions;
}


// Template instantiation
template class LDA<float>;
template class LDA<double>;
