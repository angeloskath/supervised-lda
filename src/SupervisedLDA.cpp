
#include <cmath>

#include "utils.hpp"
#include "MultinomialLogisticRegression.hpp"
#include "GradientDescent.hpp"

#include "SupervisedLDA.hpp"


template <typename Scalar>
void SupervisedLDA<Scalar>::fit(const MatrixXi &X, const VectorXi &y) {
}


template <typename Scalar>
void SupervisedLDA<Scalar>::partial_fit(const MatrixXi &X, const VectorXi &y) {
}


template <typename Scalar>
void SupervisedLDA<Scalar>::compute_h(
    const VectorXi &X,
    const MatrixX &eta,
    const MatrixX &phi,
    MatrixX &h
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
Scalar SupervisedLDA<Scalar>::doc_e_step(
    const VectorXi &X,
    int y,
    const VectorX &alpha,
    const MatrixX &beta,
    const MatrixX &eta,
    MatrixX &phi,
    VectorX &gamma,
    int fixed_point_iterations,
    int max_iter,
    Scalar convergence_tolerance
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

    while (max_iter-- > 0) {
        compute_h(X, eta, phi, h);

        new_likelihood = compute_likelihood(X, y, alpha, beta, eta, phi, gamma, h);
        if ((new_likelihood - old_likelihood)/(-old_likelihood) < convergence_tolerance) {
            break;
        }
        old_likelihood = new_likelihood;

        for (int i=0; i<fixed_point_iterations; i++) {
            phi_old = phi;

            auto t1 = gamma.unaryExpr(cwise_digamma);
            auto t2 = eta.col(y) * (X.cast<Scalar>().transpose() / num_words);
            // TODO: h.transpose() * phi_old can be cached
            auto t3 = h.array().rowwise() / (h.transpose() * phi_old).diagonal().transpose().array();

            phi = beta.array() * ((t2.colwise() + t1).array() - t3.array()).exp();
            phi = phi.array().rowwise() / phi.colwise().sum().array();
        }

        gamma = alpha.array() + (phi.array().rowwise() * X.cast<Scalar>().transpose().array()).rowwise().sum();
    }

    return new_likelihood;
}


template <typename Scalar>
void SupervisedLDA<Scalar>::doc_m_step(
    const VectorXi &X,
    const MatrixX &phi,
    MatrixX &b,
    VectorX &expected_z_bar
) {
    auto t1 = X.cast<Scalar>().transpose().array() / X.sum();
    auto t2 = phi.array().rowwise() * t1;

    b.array() += t2;
    expected_z_bar = t2.rowwise().sum();
}


template <typename Scalar>
Scalar SupervisedLDA<Scalar>::m_step(
    const MatrixX &expected_z_bar,
    const MatrixX &b,
    const VectorXi &y,
    MatrixX &beta,
    MatrixX &eta,
    Scalar L
) {
    // we maximized w.r.t \beta during each doc_m_step
    beta = b.array().rowwise() / b.array().colwise().sum();

    // we need to maximize w.r.t to \eta
    Progress<Scalar> progress{
        ProgressState::Maximization,
        -INFINITY,
        0,
        0
    };
    MultinomialLogisticRegression<Scalar> mlr(expected_z_bar, y, L);
    GradientDescent<MultinomialLogisticRegression<Scalar>, MatrixX> minimizer(
        std::make_shared<ConstantLineSearch<MultinomialLogisticRegression<Scalar>, MatrixX> >(
            0.01
        ),
        [this, &progress](Scalar value, Scalar gradNorm, size_t iterations) {
            progress.value = value;
            progress.partial_iteration = iterations;

            this->get_progress_visitor()->visit(progress);

            return iterations < 100 && gradNorm > 1e-3;
        }
    );
    minimizer.minimize(mlr, eta);

    return mlr.value(eta);
}


template <typename Scalar>
Scalar SupervisedLDA<Scalar>::compute_likelihood(
    const VectorXi &X,
    int y,
    const VectorX &alpha,
    const MatrixX &beta,
    const MatrixX &eta,
    const MatrixX &phi,
    const VectorX &gamma,
    const MatrixX &h
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
    likelihood += -(phi.array() * phi.array().log()).sum();

    // E_q[log p(y | z,n)] approximated using Jensens inequality
    likelihood += (eta.col(y).transpose() * phi * X.cast<Scalar>()).value() / X.sum();
    likelihood += - std::log((h.col(0).transpose() * phi.col(0)).value());

    return likelihood;
}


// Template instantiation
template class SupervisedLDA<float>;
template class SupervisedLDA<double>;
