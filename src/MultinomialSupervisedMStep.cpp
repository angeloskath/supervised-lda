#include "MultinomialSupervisedMStep.hpp"
#include "ProgressEvents.hpp"
#include "utils.hpp"

namespace ldaplusplus {


template <typename Scalar>
void MultinomialSupervisedMStep<Scalar>::m_step(
    std::shared_ptr<Parameters> parameters
) {
    // Normalize according to the statistics
    auto model = std::static_pointer_cast<SupervisedModelParameters<Scalar> >(parameters);
    model->beta = b_;
    model->eta = h_.array() + mu_ - 1;
    normalize_rows(model->beta);
    normalize_rows(model->eta);

    // Report the log_py
    this->get_event_dispatcher()->template dispatch<MaximizationProgressEvent<Scalar> >(
        log_py_
    );

    // Reset the statistics buffers
    b_.fill(0);
    h_.fill(0);
    log_py_ = 0;
}

template <typename Scalar>
void MultinomialSupervisedMStep<Scalar>::doc_m_step(
    const std::shared_ptr<Document> doc,
    const std::shared_ptr<Parameters> v_parameters,
    std::shared_ptr<Parameters> m_parameters
) {
    // Words and class from document
    const VectorXi &X = doc->get_words();
    int y = std::static_pointer_cast<ClassificationDocument>(doc)->get_class();

    // Cast Parameters to VariationalParameters in order to have access to phi
    const MatrixX &phi = std::static_pointer_cast<VariationalParameters<Scalar> >(v_parameters)->phi;

    // Cast model parameters to model for liberal use
    auto model = std::static_pointer_cast<SupervisedModelParameters<Scalar> >(m_parameters);

    // Allocate memory for our sufficient statistics buffers
    if (b_.rows() == 0) {
        b_ = MatrixX::Zero(phi.rows(), phi.cols());
        phi_scaled_ = MatrixX::Zero(phi.rows(), phi.cols());
        phi_scaled_sum_ = VectorX::Zero(phi.rows());

        h_ = MatrixX::Zero(model->eta.rows(), model->eta.cols());

        log_py_ = 0;
    }

    // Scale phi according to the word counts
    phi_scaled_ = phi.array().rowwise() * X.cast<Scalar>().transpose().array();
    phi_scaled_sum_ = phi_scaled_.rowwise().sum();

    // Update for beta without smoothing
    b_ += phi_scaled_;

    // Update for eta with smoothing
    h_.col(y) += phi_scaled_sum_;

    // Calculate E_q[log(p(y | z, \eta))] to report it in the maximization step
    log_py_ += (phi_scaled_sum_.transpose() * model->eta.col(y).array().log().matrix()).value();
    //log_py_ -= (X.sum() - 1) * std::log(doc->get_corpus<ClassificationCorpus>()->get_prior(y));
}


// Template instantiation
template class MultinomialSupervisedMStep<float>;
template class MultinomialSupervisedMStep<double>;

}
