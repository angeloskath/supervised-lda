#include "ProgressEvents.hpp"
#include "UnsupervisedEStep.hpp"
#include "e_step_utils.hpp"
#include "utils.hpp"

template <typename Scalar>
UnsupervisedEStep<Scalar>::UnsupervisedEStep(
    size_t e_step_iterations,
    Scalar e_step_tolerance
) {
    e_step_iterations_ = e_step_iterations;
    e_step_tolerance_ = e_step_tolerance;
}

template <typename Scalar>
std::shared_ptr<Parameters> UnsupervisedEStep<Scalar>::doc_e_step(
    const std::shared_ptr<Document> doc,
    const std::shared_ptr<Parameters> parameters
) {
    // Words form Document doc
    const VectorXi &X = doc->get_words();
    int num_words = X.sum();
    
    // Cast parameters to model parameters in order to save all necessary
    // matrixes
    const VectorX &alpha = std::static_pointer_cast<ModelParameters<Scalar> >(parameters)->alpha;
    const MatrixX &beta = std::static_pointer_cast<ModelParameters<Scalar> >(parameters)->beta;
    int num_topics = beta.rows();

    // These are the variational parameters to be computed
    MatrixX phi = MatrixX::Constant(num_topics, X.rows(), 1.0/num_topics);
    VectorX gamma = alpha.array() + static_cast<Scalar>(num_words)/num_topics;

    // to check for convergence
    Scalar old_likelihood = -INFINITY, new_likelihood = -INFINITY;

    for (size_t iteration=0; iteration<e_step_iterations_; iteration++) {
        new_likelihood = e_step_utils::compute_unsupervised_likelihood(X, alpha, beta, phi, gamma);
        if ((new_likelihood - old_likelihood)/(-old_likelihood) < e_step_tolerance_) {
            break;
        }
        old_likelihood = new_likelihood;

        // Update Multinomial parameter phi, according to the following
        // pseudocode
        //
        // for n=1 to Nd do
        //  for i=1 to K do
        //      phi_{n,i}^{t+1} = beta_{i, w_n}exp(\psi(\gamma_i) - \psi(sum_i \gamma_i))
        //  end
        //  normalize phi_{n,i}^{t+1} sum to 1
        // end
        //
        // Equation (6) in Latent Dirichlet Allocation, Blei 2003
        e_step_utils::compute_unsupervised_phi<Scalar>(beta, gamma, phi);

        // Update Dirichlet parameters according 
        //
        // gamma_i ^ {t+1} =  alpha_i + \sum_n \phi_{n,i}^{t+1}
        //
        // Equation (7) in Latent Dirichlet Allocation, Blei 2003 
        e_step_utils::compute_gamma<Scalar>(X, alpha, phi, gamma);
    }

    // notify that the e step has finished
    this->get_event_dispatcher()->template dispatch<ExpectationProgressEvent<Scalar> >(new_likelihood);

    return std::make_shared<VariationalParameters<Scalar> >(gamma, phi);
}

// Template instantiation
template class UnsupervisedEStep<float>;
template class UnsupervisedEStep<double>;

