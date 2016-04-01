#include "ProgressEvents.hpp"
#include "SupervisedEStep.hpp"
#include "e_step_utils.hpp"
#include "utils.hpp"

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
std::shared_ptr<Parameters> SupervisedEStep<Scalar>::doc_e_step(
    const std::shared_ptr<Document> doc,
    const std::shared_ptr<Parameters> parameters
) {
    // Words form Document doc
    const VectorXi &X = doc->get_words();
    int num_words = X.sum();
    int voc_size = X.rows();
    VectorX X_ratio = X.cast<Scalar>() / num_words;

    // Get the document's class
    int y = std::static_pointer_cast<ClassificationDocument>(doc)->get_class();

    // Cast parameters to model parameters in order to save all necessary
    // matrixes
    const VectorX &alpha = std::static_pointer_cast<SupervisedModelParameters<Scalar> >(parameters)->alpha;
    const MatrixX &beta = std::static_pointer_cast<SupervisedModelParameters<Scalar> >(parameters)->beta;
    const MatrixX &eta = std::static_pointer_cast<SupervisedModelParameters<Scalar> >(parameters)->eta;
    int num_topics = beta.rows();

    // The variational parameters to be computed
    MatrixX phi = MatrixX::Constant(num_topics, voc_size, 1.0/num_topics);
    VectorX gamma = alpha.array() + static_cast<Scalar>(num_words)/num_topics;

    // allocate memory for helper variables
    VectorX h(num_topics);

    // to check for convergence
    Scalar old_likelihood = -INFINITY, new_likelihood = -INFINITY;

    for (size_t iteration=0; iteration<e_step_iterations_; iteration++) {
        e_step_utils::compute_supervised_phi<Scalar>(
            X,
            X_ratio,
            y,
            beta,
            eta,
            gamma,
            fixed_point_iterations_,
            phi,
            h
        );

        // Equation (6) in Supervised topic models, Blei, McAulife 2008
        e_step_utils::compute_gamma<Scalar>(X, alpha, phi, gamma);

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
    this->get_event_dispatcher()->template dispatch<ExpectationProgressEvent<Scalar> >(new_likelihood);

    return std::make_shared<VariationalParameters<Scalar> >(gamma, phi);
}

// Template instantiation
template class SupervisedEStep<float>;
template class SupervisedEStep<double>;

