#include "ldaplusplus/events/ProgressEvents.hpp"
#include "ldaplusplus/em/SupervisedEStep.hpp"
#include "ldaplusplus/e_step_utils.hpp"
#include "ldaplusplus/utils.hpp"

namespace ldaplusplus {

using em::SupervisedEStep;


template <typename Scalar>
SupervisedEStep<Scalar>::SupervisedEStep(
    size_t e_step_iterations,
    Scalar e_step_tolerance,
    size_t fixed_point_iterations
) {
    e_step_iterations_ = e_step_iterations;
    fixed_point_iterations_ = fixed_point_iterations;
    e_step_tolerance_ = e_step_tolerance;
}

template <typename Scalar>
std::shared_ptr<parameters::Parameters> SupervisedEStep<Scalar>::doc_e_step(
    const std::shared_ptr<corpus::Document> doc,
    const std::shared_ptr<parameters::Parameters> parameters
) {
    // Words form Document doc
    const VectorXi &X = doc->get_words();
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

    // allocate memory for helper variables
    VectorX h(num_topics);

    // to check for convergence
    Scalar old_likelihood = -INFINITY, new_likelihood = -INFINITY;

    for (size_t iteration=0; iteration<e_step_iterations_; iteration++) {
        e_step_utils::compute_supervised_phi_gamma<Scalar>(
            X,
            X_ratio,
            y,
            beta,
            eta,
            fixed_point_iterations_,
            phi,
            gamma,
            h
        );

        new_likelihood = e_step_utils::compute_supervised_likelihood<Scalar>(
            X,
            y,
            alpha,
            beta,
            eta,
            phi,
            gamma,
            h
        );
        if ((new_likelihood - old_likelihood)/(-old_likelihood) < e_step_tolerance_) {
            break;
        }
        old_likelihood = new_likelihood;
    }

    // notify that the e step has finished
    this->get_event_dispatcher()->template dispatch<events::ExpectationProgressEvent<Scalar> >(new_likelihood);

    return std::make_shared<parameters::VariationalParameters<Scalar> >(gamma, phi);
}

// Template instantiation
template class SupervisedEStep<float>;
template class SupervisedEStep<double>;


}
