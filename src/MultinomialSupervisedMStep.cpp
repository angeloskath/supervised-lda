#include "MultinomialSupervisedMStep.hpp"
#include "utils.hpp"

template <typename Scalar>
void MultinomialSupervisedMStep<Scalar>::m_step(
    std::shared_ptr<Parameters> parameters
) {
    // Normalize according to the statistics
    auto model = std::static_pointer_cast<SupervisedModelParameters<Scalar> >(parameters);
    model->beta = b_;
    model->eta = h_;
    normalize_rows(model->beta);
    normalize_rows(model->eta);

    // Reset the statistics buffers
    b_.fill(0);
    h_.fill(0);
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

    // Allocate memory for our sufficient statistics buffers
    if (b_.rows() == 0) {
        b_ = MatrixX::Zero(phi.rows(), phi.cols());

        auto model = std::static_pointer_cast<SupervisedModelParameters<Scalar> >(m_parameters);
        h_ = MatrixX::Zero(model->eta.rows(), model->eta.cols());
    }

    // Scale phi according to the word counts
    auto phi_scaled = phi.array().rowwise() * X.cast<Scalar>().transpose().array();

    // Update for beta without smoothing
    b_.array() += phi_scaled;

    // Update for eta with smoothing
    h_.col(y).array() += phi_scaled.rowwise().sum();
    h_.array() += mu_ - 1;
}


// Template instantiation
template class MultinomialSupervisedMStep<float>;
template class MultinomialSupervisedMStep<double>;
