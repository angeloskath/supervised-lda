#include <utility>

#include "ldaplusplus/em/OnlineSupervisedMStep.hpp"
#include "ldaplusplus/optimization/MultinomialLogisticRegression.hpp"
#include "ldaplusplus/events/ProgressEvents.hpp"

namespace ldaplusplus {

using em::OnlineSupervisedMStep;


template <typename Scalar>
OnlineSupervisedMStep<Scalar>::OnlineSupervisedMStep(
    VectorX class_weights,
    Scalar regularization_penalty,
    size_t minibatch_size,
    Scalar eta_momentum,
    Scalar eta_learning_rate,
    Scalar beta_weight
) : class_weights_(std::move(class_weights)),
    num_classes_(class_weights_.rows()),
    minibatch_size_(minibatch_size),
    regularization_penalty_(regularization_penalty),
    beta_weight_(beta_weight),
    eta_momentum_(eta_momentum),
    eta_learning_rate_(eta_learning_rate),
    docs_seen_so_far_(0)
{}

template <typename Scalar>
OnlineSupervisedMStep<Scalar>::OnlineSupervisedMStep(
    size_t num_classes,
    Scalar regularization_penalty,
    size_t minibatch_size,
    Scalar eta_momentum,
    Scalar eta_learning_rate,
    Scalar beta_weight
) : OnlineSupervisedMStep(
        VectorX::Constant(num_classes, 1),
        regularization_penalty,
        minibatch_size,
        eta_momentum,
        eta_learning_rate,
        beta_weight
    )
{}

template <typename Scalar>
void OnlineSupervisedMStep<Scalar>::doc_m_step(
    const std::shared_ptr<corpus::Document> doc,
    const std::shared_ptr<parameters::Parameters> v_parameters,
    std::shared_ptr<parameters::Parameters> m_parameters
) {
    // Data from document doc
    const Eigen::VectorXi & X = doc->get_words();
    int y = std::static_pointer_cast<corpus::ClassificationDocument>(doc)->get_class(); 
    // Variational parameters
    const MatrixX & phi = std::static_pointer_cast<parameters::VariationalParameters<Scalar> >(v_parameters)->phi;
    const VectorX &gamma = std::static_pointer_cast<parameters::VariationalParameters<Scalar> >(v_parameters)->gamma;
    // Supervised model parameters
    const VectorX &alpha = std::static_pointer_cast<parameters::SupervisedModelParameters<Scalar> >(m_parameters)->alpha;

    // Initialize our variables
    if (b_.rows() == 0) {
        b_ = MatrixX::Zero(phi.rows(), phi.cols());

        expected_z_bar_ = MatrixX::Zero(phi.rows(), minibatch_size_);
        y_ = Eigen::VectorXi::Zero(minibatch_size_);
        eta_velocity_ = MatrixX::Zero(phi.rows(), num_classes_);
        eta_gradient_ = MatrixX::Zero(phi.rows(), num_classes_);
    }

    // Unsupervised sufficient statistics
    b_.array() += phi.array().rowwise() * X.cast<Scalar>().transpose().array();

    // Supervised suff stats
    expected_z_bar_.col(docs_seen_so_far_) = gamma - alpha;
    expected_z_bar_.col(docs_seen_so_far_).array() /= expected_z_bar_.col(docs_seen_so_far_).sum();
    y_(docs_seen_so_far_) = y;

    // mark another document as seen
    docs_seen_so_far_++;

    // Check if we need to update the parameters
    if (docs_seen_so_far_ >= minibatch_size_)
        m_step(m_parameters);
}

template <typename Scalar>
void OnlineSupervisedMStep<Scalar>::m_step(
    std::shared_ptr<parameters::Parameters> parameters
) {
    // Check whether we should actually perform the m_step
    if (docs_seen_so_far_ < minibatch_size_)
        return;

    docs_seen_so_far_ = 0;

    // Extract the parameters from the struct
    auto model = std::static_pointer_cast<parameters::SupervisedModelParameters<Scalar> >(parameters);
    MatrixX & beta = model->beta;
    MatrixX & eta = model->eta;

    // update the topic distributions
    // TODO: Change the update to something more formal like the online update
    //       of Hoffman et al.
    beta.array() = (
        beta_weight_ * beta.array() +
        (1-beta_weight_) * (b_.array().colwise() / b_.array().rowwise().sum())
    );

    // update the eta
    optimization::MultinomialLogisticRegression<Scalar> mlr(
        expected_z_bar_,
        y_,
        regularization_penalty_
    );
    mlr.gradient(eta, eta_gradient_);
    eta_velocity_ = eta_momentum_ * eta_velocity_ - eta_learning_rate_ * eta_gradient_;
    eta += eta_velocity_;

    this->get_event_dispatcher()->template dispatch<events::MaximizationProgressEvent<Scalar> >(
        -mlr.value(eta)  // minus the value to be minimized is the log likelihood
    );
}


// Instantiations
template class OnlineSupervisedMStep<float>;
template class OnlineSupervisedMStep<double>;

}
