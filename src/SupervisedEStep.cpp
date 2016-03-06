#include "SupervisedEStep.hpp"
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
std::shared_ptr<Parameters> UnsupervisedEStep<Scalar>::doc_e_step(
    const std::shared_ptr<Document> doc,
    const std::shared_ptr<Parameters> model_parameters
) {
    auto cwise_digamma = CwiseDigamma<Scalar>();

    int num_topics = gamma.rows();
    int num_words = X.sum();
    int voc_size = X.rows();

    gamma = alpha.array() + static_cast<Scalar>(num_words)/num_topics;
    phi.fill(1.0/num_topics);

    // allocate memory for helper variables
    MatrixX h(num_topics, voc_size);
    MatrixX phi_old(num_topics, voc_size);

    // to check for convergence
    Scalar old_likelihood = -INFINITY, new_likelihood = -INFINITY;

    for (size_t iteration=0; iteration<e_step_iterations_; iteration++) {
        compute_h(X, eta, phi, h);

        new_likelihood = compute_likelihood(X, y, alpha, beta, eta, phi, gamma, h);
        if ((new_likelihood - old_likelihood)/(-old_likelihood) < e_step_tolerance_) {
            break;
        }
        old_likelihood = new_likelihood;
        
        for (size_t i=0; i<fixed_point_iterations_; i++) {
            phi_old = phi;

            auto t1 = gamma.unaryExpr(cwise_digamma);
            auto t2 = eta.col(y) * (X.cast<Scalar>().transpose() / num_words);
            // TODO: h.transpose() * phi_old can be cached
            auto t3 = h.array().rowwise() / (h.transpose() * phi_old).diagonal().transpose().array();

            phi = beta.array() * ((t2.colwise() + t1).array() - t3.array()).exp();
            phi = phi.array().rowwise() / phi.colwise().sum().array();
        }

        // Equation (6) in Supervised topic models, Blei, McAulife 2008
        gamma = alpha.array() + (phi.array().rowwise() * X.cast<Scalar>().transpose().array()).rowwise().sum();
    }

    return new_likelihood;
}

template <typename Scalar>
void SupervisedEStep<Scalar>::compute_h(
    const VectorXi &X,
    const MatrixX &eta,
    const MatrixX &phi,
    Ref<MatrixX> h
) {
    MatrixX exp_eta(h.rows(), h.cols());
    VectorX products(phi.cols());
    int num_words = X.sum();
    int num_classes = eta.cols();

    h.fill(0);
    for (int y=0; y<num_classes; y++) {
        auto eta_scaled = eta.col(y) * (X.cast<Scalar>() / num_words).transpose();
        exp_eta = eta_scaled.array().exp();
        products = (exp_eta.transpose() * phi).diagonal();

        auto t1 = (products.prod() / products.array()).matrix();
        auto t2 = exp_eta * t1.asDiagonal();

        h += t2;
    }
}

template <typename Scalar>
Scalar SupervisedEStep<Scalar>::compute_likelihood(
    const VectorXi &X,
    int y,
    const VectorX &alpha,
    const MatrixX &beta,
    const MatrixX &eta,
    const MatrixX &phi,
    const VectorX &gamma,
    const MatrixX &h
) {
    Scalar likelihood = UnsupervisedEStep<Scalar>::compute_likelihood(
        X,
        alpha,
        beta,
        phi,
        gamma
    );

    // E_q[log p(y | z,n)] approximated using Jensens inequality
    likelihood += (eta.col(y).transpose() * phi * X.cast<Scalar>()).value() / X.sum();
    likelihood += - std::log((h.col(0).transpose() * phi.col(0)).value());
    
    return likelihood;
}

// Template instantiation
template class SupervisedEStep<float>;
template class SupervisedEStep<double>;

