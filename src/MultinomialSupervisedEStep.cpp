#include "ProgressEvents.hpp"
#include "MultinomialSupervisedEStep.hpp"
#include "e_step_utils.hpp"
#include "utils.hpp"

namespace ldaplusplus {


template <typename Scalar>
MultinomialSupervisedEStep<Scalar>::MultinomialSupervisedEStep(
    size_t e_step_iterations,
    Scalar e_step_tolerance,
    Scalar mu,
    Scalar eta_weight
) {
    e_step_iterations_ = e_step_iterations;
    e_step_tolerance_ = e_step_tolerance;
    mu_ = mu;
    eta_weight_ = eta_weight;
}

template <typename Scalar>
std::shared_ptr<Parameters> MultinomialSupervisedEStep<Scalar>::doc_e_step(
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
    int corpus_size = doc->get_corpus()->size();
    Scalar prior_y = doc->get_corpus<ClassificationCorpus>()->get_prior(y);

    // Cast parameters to model parameters in order to save all necessary
    // matrixes
    const VectorX &alpha = std::static_pointer_cast<SupervisedModelParameters<Scalar> >(parameters)->alpha;
    const MatrixX &beta = std::static_pointer_cast<SupervisedModelParameters<Scalar> >(parameters)->beta;
    const MatrixX &eta = std::static_pointer_cast<SupervisedModelParameters<Scalar> >(parameters)->eta;
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

        e_step_utils::compute_supervised_multinomial_phi<Scalar>(
            X,
            y,
            beta,
            eta,
            gamma,
            eta_weight_,
            phi
        );

        // Equation (6) in Supervised topic models, Blei, McAulife 2008
        e_step_utils::compute_gamma<Scalar>(X, alpha, phi, gamma);
    }

    // notify that the e step has finished
    this->get_event_dispatcher()->template dispatch<ExpectationProgressEvent<Scalar> >(
        e_step_utils::compute_supervised_multinomial_likelihood<Scalar>(
            X,
            y,
            alpha,
            beta,
            eta,
            phi,
            gamma,
            prior_y,
            mu_,
            1.0 / corpus_size
        )
    );

    return std::make_shared<VariationalParameters<Scalar> >(gamma, phi);
}


template <typename Scalar>
bool MultinomialSupervisedEStep<Scalar>::converged(
    const VectorX & gamma_old,
    const VectorX & gamma
) {
    Scalar mean_change = (gamma_old - gamma).array().abs().sum() / gamma.rows();

    return mean_change < e_step_tolerance_;
}

// Template instantiation
template class MultinomialSupervisedEStep<float>;
template class MultinomialSupervisedEStep<double>;


}
