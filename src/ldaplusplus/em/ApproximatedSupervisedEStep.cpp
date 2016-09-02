#include <cmath>

#include "ldaplusplus/events/ProgressEvents.hpp"
#include "ldaplusplus/em/ApproximatedSupervisedEStep.hpp"
#include "ldaplusplus/e_step_utils.hpp"

namespace ldaplusplus {

using em::ApproximatedSupervisedEStep;


template <typename Scalar>
ApproximatedSupervisedEStep<Scalar>::ApproximatedSupervisedEStep(
    size_t e_step_iterations,
    Scalar e_step_tolerance,
    Scalar C,
    CWeightType weight_type,
    bool compute_likelihood
) {
    e_step_iterations_ = e_step_iterations;
    e_step_tolerance_ = e_step_tolerance;
    C_ = C;
    weight_type_ = weight_type;
    compute_likelihood_ = compute_likelihood;
    epochs_ = 0;
}

template <typename Scalar>
std::shared_ptr<parameters::Parameters> ApproximatedSupervisedEStep<Scalar>::doc_e_step(
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

    // Cast parameters to model parameters in order to save all necessary
    // matrixes
    const VectorX &alpha = std::static_pointer_cast<parameters::SupervisedModelParameters<Scalar> >(parameters)->alpha;
    const MatrixX &beta = std::static_pointer_cast<parameters::SupervisedModelParameters<Scalar> >(parameters)->beta;
    const MatrixX &eta = std::static_pointer_cast<parameters::SupervisedModelParameters<Scalar> >(parameters)->eta;
    int num_topics = beta.rows();

    // The variational parameters to be computed
    MatrixX phi = MatrixX::Constant(num_topics, voc_size, 1.0/num_topics);
    VectorX gamma = alpha.array() + static_cast<Scalar>(num_words)/num_topics;

    // to check for convergence
    VectorX gamma_old = VectorX::Zero(num_topics);

    for (size_t iteration=0; iteration<e_step_iterations_; iteration++) {
        // check for early stopping
        if (converged(gamma_old, gamma)) {
            break;
        }
        gamma_old = gamma;

        e_step_utils::compute_supervised_approximate_phi<Scalar>(
            X_ratio,
            num_words,
            y,
            beta,
            eta,
            gamma,
            get_weight(),
            phi
        );

        // Equation (6) in Supervised topic models, Blei, McAulife 2008
        e_step_utils::compute_gamma<Scalar>(X, alpha, phi, gamma);
    }

    // notify that the e step has finished
    if (compute_likelihood_) {
        this->get_event_dispatcher()->template dispatch<events::ExpectationProgressEvent<Scalar> >(
            e_step_utils::compute_supervised_likelihood<Scalar>(
                X,
                y,
                alpha,
                beta,
                eta,
                phi,
                gamma
            )
        );
    } else {
        this->get_event_dispatcher()->template dispatch<events::ExpectationProgressEvent<Scalar> >(0);
    }

    return std::make_shared<parameters::VariationalParameters<Scalar> >(gamma, phi);
}


template <typename Scalar>
void ApproximatedSupervisedEStep<Scalar>::e_step() {
    epochs_ ++;
}


template <typename Scalar>
bool ApproximatedSupervisedEStep<Scalar>::converged(
    const VectorX & gamma_old,
    const VectorX & gamma
) {
    Scalar mean_change = (gamma_old - gamma).array().abs().sum() / gamma.rows();

    return mean_change < e_step_tolerance_;
}


template <typename Scalar>
Scalar ApproximatedSupervisedEStep<Scalar>::get_weight() {
    switch (weight_type_) {
        case ExponentialDecay:
            return std::pow(C_, epochs_);
        default:
        case Constant:
            return C_;
    }
}

// Template instantiation
template class ApproximatedSupervisedEStep<float>;
template class ApproximatedSupervisedEStep<double>;


}
