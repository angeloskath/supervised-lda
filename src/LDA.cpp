
#include <algorithm>
#include <numeric>
#include <random>

#include "LDA.hpp"
#include "InternalsFactory.hpp"
#include "ProgressEvents.hpp"


template <typename Scalar>
LDA<Scalar>::LDA(
    std::shared_ptr<IInitialization<Scalar> > initialization,
    std::shared_ptr<IEStep<Scalar> > unsupervised_e_step,
    std::shared_ptr<IMStep<Scalar> > unsupervised_m_step,
    std::shared_ptr<IEStep<Scalar> > e_step,
    std::shared_ptr<IMStep<Scalar> > m_step,
    size_t iterations
) : initialization_(initialization),
    unsupervised_e_step_(unsupervised_e_step),
    unsupervised_m_step_(unsupervised_m_step),
    e_step_(e_step),
    m_step_(m_step),
    iterations_(iterations),
    event_dispatcher_(std::make_shared<EventDispatcher>())
{
    set_up_event_dispatcher();
}


template <typename Scalar>
LDA<Scalar>::LDA(
    LDAState lda_state,
    size_t iterations
) : iterations_(iterations),
    event_dispatcher_(std::make_shared<EventDispatcher>())
{
    // extract the model parameters
    alpha_ = *lda_state.alpha;
    beta_ = *lda_state.beta;
    eta_ = *lda_state.eta;

    // recreate the model implementations
    auto factory = std::make_shared<InternalsFactory<Scalar> >();
    initialization_ = factory->create_initialization(
        lda_state.ids[0],
        lda_state.parameters[0]
    );
    unsupervised_e_step_ = factory->create_e_step(
        lda_state.ids[1],
        lda_state.parameters[1]
    );
    unsupervised_m_step_ = factory->create_m_step(
        lda_state.ids[2],
        lda_state.parameters[2]
    );
    e_step_ = factory->create_e_step(
        lda_state.ids[3],
        lda_state.parameters[3]
    );
    m_step_ = factory->create_m_step(
        lda_state.ids[4],
        lda_state.parameters[4]
    );

    set_up_event_dispatcher();
}


template <typename Scalar>
void LDA<Scalar>::set_up_event_dispatcher() {
    auto event_dispatcher = get_event_dispatcher();

    unsupervised_e_step_->set_event_dispatcher(event_dispatcher);
    unsupervised_m_step_->set_event_dispatcher(event_dispatcher);
    e_step_->set_event_dispatcher(event_dispatcher);
    m_step_->set_event_dispatcher(event_dispatcher);
}


template <typename Scalar>
void LDA<Scalar>::fit(const MatrixXi &X, const VectorXi &y) {
    for (size_t i=0; i<iterations_; i++) {
        partial_fit(X, y);
    }
}


template <typename Scalar>
void LDA<Scalar>::partial_fit(const MatrixXi &X, const VectorXi &y) {
    // This means we have never been called before so allocate whatever needs
    // to be allocated and initialize the model parameters
    if (beta_.rows() == 0) {
        initialization_->initialize_model_parameters(
            X,
            y,
            alpha_,
            beta_,
            eta_
        );
    }

    // allocate space for the variational parameters (they are both per
    // document)
    MatrixX phi(beta_.rows(), X.rows());
    VectorX gamma(beta_.rows());

    // allocate space for accumulating values to use in the maximization step
    MatrixX expected_z_bar(beta_.rows(), X.cols());
    MatrixX b = MatrixX::Zero(beta_.rows(), X.rows());

    // create an array with shuffled indexes which we use to view the dataset
    // in a randomized fashion
    std::vector<int> idxs(X.cols());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::random_shuffle(idxs.begin(), idxs.end());
    size_t cnt = 0;

    // consider moving the following to another function so that the above
    // allocations do not happen again and again for every iteration
    Scalar likelihood = 0;
    for (auto d : idxs) {
        if (X.col(d).sum() == 0) {
            continue;
        }

        likelihood = e_step_->doc_e_step(
            X.col(d),
            y[d],
            alpha_,
            beta_,
            eta_,
            phi,
            gamma
        );

        m_step_->doc_m_step(
            X.col(d),
            phi,
            b,
            expected_z_bar.col(d)
        );

        get_event_dispatcher()->template dispatch<ExpectationProgressEvent<Scalar> >(
            ++cnt,
            likelihood
        );
    }

    m_step_->m_step(
        expected_z_bar,
        b,
        y,
        beta_,
        eta_
    );

    get_event_dispatcher()->template dispatch<EpochProgressEvent<Scalar> >(get_state());
}


template <typename Scalar>
typename LDA<Scalar>::MatrixX LDA<Scalar>::transform(const MatrixXi& X) {
    // space for the variational parameters
    MatrixX phi(beta_.rows(), X.rows());
    VectorX gamma(beta_.rows());

    // space for the representation and an unused b parameter
    MatrixX expected_z_bar(beta_.rows(), X.cols());
    MatrixX b(beta_.rows(), X.rows());

    for (int d=0; d<X.cols(); d++) {
        unsupervised_e_step_->doc_e_step(
            X.col(d),
            -1,
            alpha_,
            beta_,
            eta_,
            phi,
            gamma
        );

        unsupervised_m_step_->doc_m_step(
            X.col(d),
            phi,
            b,
            expected_z_bar.col(d)
        );
    }

    return expected_z_bar;
}


template <typename Scalar>
typename LDA<Scalar>::MatrixX LDA<Scalar>::decision_function(const MatrixXi &X) {
    return eta_.transpose() * transform(X);
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
