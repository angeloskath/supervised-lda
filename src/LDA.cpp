
#include <algorithm>
#include <numeric>
#include <random>

#include "LDA.hpp"
#include "InternalsFactory.hpp"


template <typename Scalar>
LDA<Scalar>::LDA(
    std::shared_ptr<IInitialization<Scalar> > initialization,
    std::shared_ptr<IEStep<Scalar> > unsupervised_e_step,
    std::shared_ptr<IMStep<Scalar> > unsupervised_m_step,
    std::shared_ptr<IEStep<Scalar> > e_step,
    std::shared_ptr<IMStep<Scalar> > m_step,
    size_t topics,
    size_t iterations
) : initialization_(initialization),
    unsupervised_e_step_(unsupervised_e_step),
    unsupervised_m_step_(unsupervised_m_step),
    e_step_(e_step),
    m_step_(m_step),
    topics_(topics),
    iterations_(iterations)
{}


template <typename Scalar>
LDA<Scalar>::LDA(
    LDAState lda_state,
    size_t topics,
    size_t iterations
) : topics_(topics),
    iterations_(iterations)
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
}


template <typename Scalar>
void LDA<Scalar>::fit(const MatrixXi &X, const VectorXi &y) {
    for (size_t i=0; i<topics_; i++) {
        partial_fit(X, y);
    }
}


template <typename Scalar>
void LDA<Scalar>::partial_fit(const MatrixXi &X, const VectorXi &y) {
    // This means we have never been called before so allocate whatever needs
    // to be allocated and initialize the model parameters
    if (beta_.rows() == 0) {
        alpha_ = VectorX(topics_);
        eta_ = MatrixX(topics_, y.maxCoeff() + 1);
        beta_ = MatrixX(topics_, X.rows());

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
    MatrixX phi(topics_, X.rows());
    VectorX gamma(topics_);

    // allocate space for accumulating values to use in the maximization step
    MatrixX expected_z_bar(topics_, X.cols());
    MatrixX b = MatrixX::Zero(topics_, X.rows());

    // create an array with shuffled indexes which we use to view the dataset
    // in a randomized fashion
    std::vector<int> idxs(X.cols());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::random_shuffle(idxs.begin(), idxs.end());

    // consider moving the following to another function so that the above
    // allocations do not happen again and again for every iteration
    Scalar likelihood = 0;
    for (auto d : idxs) {
        if (X.col(d).sum() == 0) {
            continue;
        }

        likelihood += e_step_->doc_e_step(
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

        get_progress_visitor()->visit(Progress<Scalar>{
            ProgressState::Expectation,
            likelihood,
            static_cast<size_t>(d),
            0
        });
    }

    m_step_->m_step(
        expected_z_bar,
        b,
        y,
        beta_,
        eta_
    );

    get_progress_visitor()->visit(Progress<Scalar>{
        ProgressState::IterationFinished,
        0,
        0,
        0,
        get_state()
    });
}


template <typename Scalar>
typename LDA<Scalar>::MatrixX LDA<Scalar>::transform(const MatrixXi& X) {
    // space for the variational parameters
    MatrixX phi(topics_, X.rows());
    VectorX gamma(topics_);

    // space for the representation and an unused b parameter
    MatrixX expected_z_bar(topics_, X.cols());
    MatrixX b(topics_, X.rows());

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
