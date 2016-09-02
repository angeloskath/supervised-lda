#include "ldaplusplus/events/ProgressEvents.hpp"
#include "ldaplusplus/em/CorrespondenceSupervisedEStep.hpp"
#include "ldaplusplus/e_step_utils.hpp"

namespace ldaplusplus {

using em::CorrespondenceSupervisedEStep;


template <typename Scalar>
CorrespondenceSupervisedEStep<Scalar>::CorrespondenceSupervisedEStep(
    size_t e_step_iterations,
    Scalar e_step_tolerance,
    Scalar mu
) {
    e_step_iterations_ = e_step_iterations;
    e_step_tolerance_ = e_step_tolerance;
    mu_ = mu;
}

template <typename Scalar>
std::shared_ptr<parameters::Parameters> CorrespondenceSupervisedEStep<Scalar>::doc_e_step(
    const std::shared_ptr<corpus::Document> doc,
    const std::shared_ptr<parameters::Parameters> parameters
) {
    // Words form Document doc
    const Eigen::VectorXi &X = doc->get_words();
    int num_words = X.sum();
    int voc_size = X.rows();
    VectorX X_ratio = X.cast<Scalar>() / num_words;

    // Get the document's class
    int y = std::static_pointer_cast<corpus::ClassificationDocument>(doc)->get_class();
    int corpus_size = doc->get_corpus()->size();

    // Cast parameters to model parameters in order to save all necessary
    // matrixes
    const VectorX &alpha = std::static_pointer_cast<parameters::SupervisedModelParameters<Scalar> >(parameters)->alpha;
    const MatrixX &beta = std::static_pointer_cast<parameters::SupervisedModelParameters<Scalar> >(parameters)->beta;
    const MatrixX &eta = std::static_pointer_cast<parameters::SupervisedModelParameters<Scalar> >(parameters)->eta;
    int num_topics = beta.rows();

    // The variational parameters to be computed
    MatrixX phi = MatrixX::Constant(num_topics, voc_size, 1.0/num_topics);
    VectorX gamma = alpha.array() + static_cast<Scalar>(num_words)/num_topics;
    VectorX tau = VectorX::Constant(voc_size, 1.0/voc_size);

    // to check for convergence
    VectorX gamma_old = VectorX::Zero(num_topics);

    for (size_t iteration=0; iteration<e_step_iterations_; iteration++) {
        // check for early stopping
        if (converged(gamma_old, gamma)) {
            break;
        }
        gamma_old = gamma;

        // Maximize the likelihood w.r.t phi
        e_step_utils::compute_supervised_correspondence_phi<Scalar>(
            X,
            y,
            beta,
            eta,
            gamma,
            tau,
            phi
        );

        // Maximize the likelihood w.r.t tau
        e_step_utils::compute_supervised_correspondence_tau<Scalar>(
            X,
            y,
            eta,
            phi,
            tau
        );

        // Equation (6) in Supervised topic models, Blei, McAulife 2008
        e_step_utils::compute_gamma<Scalar>(X, alpha, phi, gamma);
    }

    // notify that the e step has finished
    this->get_event_dispatcher()->template dispatch<events::ExpectationProgressEvent<Scalar> >(
        e_step_utils::compute_supervised_correspondence_likelihood<Scalar>(
            X,
            y,
            alpha,
            beta,
            eta,
            phi,
            gamma,
            tau,
            mu_,
            1.0 / corpus_size
        )
    );

    return std::make_shared<parameters::SupervisedCorrespondenceVariationalParameters<Scalar> >(
        gamma,
        phi,
        tau
    );
}


template <typename Scalar>
bool CorrespondenceSupervisedEStep<Scalar>::converged(
    const VectorX & gamma_old,
    const VectorX & gamma
) {
    Scalar mean_change = (gamma_old - gamma).array().abs().sum() / gamma.rows();

    return mean_change < e_step_tolerance_;
}

// Template instantiation
template class CorrespondenceSupervisedEStep<float>;
template class CorrespondenceSupervisedEStep<double>;

}
