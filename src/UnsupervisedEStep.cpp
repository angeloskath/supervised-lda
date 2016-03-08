#include "ProgressEvents.hpp"
#include "UnsupervisedEStep.hpp"
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
    auto cwise_digamma = CwiseDigamma<Scalar>();

    // Words form Document doc
    const VectorXi &X = doc->get_words();
    int num_words = X.sum();
    
    // Cast parameters to model parameters in order to save all necessary
    // matrixes
    const VectorX &alpha = std::static_pointer_cast<ModelParameters<Scalar> >(parameters)->alpha;
    const MatrixX &beta = std::static_pointer_cast<ModelParameters<Scalar> >(parameters)->beta;
    int num_topics = beta.rows();
    
    MatrixX phi = MatrixX::Constant(num_topics, X.rows(), 1.0/num_topics);
    VectorX gamma = alpha.array() + static_cast<Scalar>(num_words)/num_topics;

    // to check for convergence
    Scalar old_likelihood = -INFINITY, new_likelihood = -INFINITY;

    for (size_t iteration=0; iteration<e_step_iterations_; iteration++) {
        new_likelihood = compute_likelihood(X, alpha, beta, phi, gamma);
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
        auto t1 = gamma.unaryExpr(cwise_digamma).array();
        auto t2 = digamma(gamma.sum());
        phi = beta.array().colwise() * (t1 - t2).exp();
        phi = phi.array().rowwise() / phi.colwise().sum().array();

        // Update Dirichlet parameters according 
        //
        // gamma_i ^ {t+1} =  alpha_i + \sum_n \phi_{n,i}^{t+1}
        //
        // Equation (7) in Latent Dirichlet Allocation, Blei 2003 
        gamma = alpha.array() + (phi.array().rowwise() * X.cast<Scalar>().transpose().array()).rowwise().sum();
    }

    // notify that the e step has finished
    this->get_event_dispatcher()->template dispatch<ExpectationProgressEvent<Scalar> >(new_likelihood);

    return std::make_shared<VariationalParameters<Scalar> >(gamma, phi);
}

template <typename Scalar>
Scalar UnsupervisedEStep<Scalar>::compute_likelihood(
    const VectorXi &X,
    const VectorX &alpha,
    const MatrixX &beta,
    const MatrixX &phi,
    const VectorX &gamma
) {
    auto cwise_digamma = CwiseDigamma<Scalar>();
    auto cwise_lgamma = CwiseLgamma<Scalar>();

    Scalar likelihood = 0;

    // \Psi(\gamma) - \Psi(\sum_j \gamma)
    VectorX t1 = gamma.unaryExpr(cwise_digamma).array() - digamma(gamma.sum());

    // E_q[log p(\theta | \alpha)]
    likelihood += ((alpha.array() - 1.0).matrix().transpose() * t1).value();
    likelihood += std::lgamma(alpha.sum()) - alpha.unaryExpr(cwise_lgamma).sum();

    // E_q[log p(z | \theta)]
    likelihood += (phi.transpose() * t1).sum();

    // E_q[log p(w | z, \beta)]
    auto phi_scaled = phi.array().rowwise() * X.cast<Scalar>().transpose().array();
    likelihood += (phi_scaled * beta.array().log()).sum();

    // H(q)
    likelihood += -((gamma.array() - 1).matrix().transpose() * t1).value();
    likelihood += -std::lgamma(gamma.sum()) + gamma.unaryExpr(cwise_lgamma).sum();
    likelihood += -(phi.array() * (phi.array() + 1e-44).log()).sum();
    
    return likelihood;
}


// Template instantiation
template class UnsupervisedEStep<float>;
template class UnsupervisedEStep<double>;

