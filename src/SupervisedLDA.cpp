
#include <cmath>
#include <random>
#include <iostream>

#include "utils.hpp"
#include "MultinomialLogisticRegression.hpp"
#include "GradientDescent.hpp"

#include "SupervisedLDA.hpp"


template <typename Scalar>
SupervisedLDA<Scalar>::SupervisedLDA(
    size_t topics,
    size_t iterations,
    Scalar e_step_tolerance,
    Scalar m_step_tolerance,
    size_t e_step_iterations,
    size_t m_step_iterations,
    size_t fixed_point_iterations,
    Scalar regularization_penalty
) {
    topics_ = topics;
    iterations_ = iterations;
    e_step_tolerance_ = e_step_tolerance;
    m_step_tolerance_ = m_step_tolerance;
    e_step_iterations_ = e_step_iterations;
    m_step_iterations_ = m_step_iterations;
    fixed_point_iterations_ = fixed_point_iterations;
    regularization_penalty_ = regularization_penalty;
}

template <typename Scalar>
void SupervisedLDA<Scalar>::initialize_model_parameters(
    const MatrixXi &X,
    const VectorXi &y,
    Ref<VectorX> alpha,
    Ref<MatrixX> beta,
    Ref<MatrixX> eta,
    size_t topics
) {
    alpha = VectorX::Constant(topics, 1.0 / topics);
    // Eigen has no unique function, therefore we use maxcoeff instead to
    // calculate the number of classes, which is C
    eta = MatrixX::Zero(topics, y.maxCoeff() + 1);
    
    beta = MatrixX::Constant(topics, X.rows(), 1.0);
    std::mt19937 rng;
    rng.seed(0);
    std::uniform_int_distribution<> initializations(10, static_cast<int>(X.cols()/2));
    std::uniform_int_distribution<> document(0, X.cols()-1);
    auto N = initializations(rng);

    // Initialize _beta
    for (int k=0; k<topics_; k++) {
        // Choose randomly a bunch of documents to initialize beta
        for (int r=0; r<N; r++) {
            beta.row(k) += X.cast<Scalar>().col(document(rng)).transpose();
        }
        beta.row(k) = beta.row(k) / beta.row(k).sum();
    }
}

template <typename Scalar>
void SupervisedLDA<Scalar>::fit(const MatrixXi &X, const VectorXi &y) {
    for (int i=0; i<iterations_; i++) {
        partial_fit(X, y);
    }
}


template <typename Scalar>
void SupervisedLDA<Scalar>::partial_fit(const MatrixXi &X, const VectorXi &y) {
    // This means we have never been called before so allocate whatever needs
    // to be allocated and initialize the model parameters
    if (beta_.rows() == 0) {
        alpha_ = VectorX(topics_);
        eta_ = MatrixX(topics_, y.maxCoeff() + 1);
        beta_ = MatrixX(topics_, X.rows());

        initialize_model_parameters(
            X,
            y,
            alpha_,
            beta_,
            eta_,
            topics_
        );
    }

    // allocate space for the variational parameters (they are both per
    // document)
    MatrixX phi(topics_, X.rows());
    VectorX gamma(topics_);

    // allocate space for accumulating values to use in the maximization step
    MatrixX expected_z_bar(topics_, X.cols());
    MatrixX b(topics_, X.rows());

    // consider moving the following to another function so that the above
    // allocations do not happen again and again for every iteration
    Scalar likelihood = 0;
    for (int d=0; d<X.cols(); d++) {
        likelihood += doc_e_step(
            X.col(d),
            y[d],
            alpha_,
            beta_,
            eta_,
            phi,
            gamma,
            fixed_point_iterations_,
            e_step_iterations_,
            e_step_tolerance_
        );

        doc_m_step(
            X.col(d),
            phi,
            b,
            expected_z_bar.col(d)
        );

        get_progress_visitor()->visit(Progress<Scalar>{
            ProgressState::Expectation,
            likelihood,
            static_cast<size_t>(d),
            0
        });
    }

    m_step(
        expected_z_bar,
        b,
        y,
        beta_,
        eta_,
        regularization_penalty_,
        m_step_iterations_,
        m_step_tolerance_
    );
}

template <typename Scalar>
typename SupervisedLDA<Scalar>::MatrixX SupervisedLDA<Scalar>::transform(const MatrixXi& X) {
    // space for the variational parameters
    MatrixX phi(topics_, X.rows());
    VectorX gamma(topics_);

    // space for the representation and an unused b parameter
    MatrixX expected_z_bar(topics_, X.cols());
    MatrixX b(topics_, X.rows());

    for (int d=0; d<X.rows(); d++) {
        doc_e_step(
            X.col(d),
            -1,
            alpha_,
            beta_,
            eta_,
            phi,
            gamma,
            fixed_point_iterations_,
            e_step_iterations_,
            e_step_tolerance_
        );

        doc_m_step(
            X.col(d),
            phi,
            b,
            expected_z_bar.col(d)
        );
    }

    return expected_z_bar;
}

template <typename Scalar>
typename SupervisedLDA<Scalar>::MatrixX SupervisedLDA<Scalar>::decision_function(const MatrixXi &X) {
    MatrixX scores(eta_.cols(), X.cols());

    scores = (eta_.transpose() * transform(X)).eval();

    return scores;
}

template <typename Scalar>
VectorXi SupervisedLDA<Scalar>::predict(const MatrixXi &X) {
    VectorXi predictions(X.cols());
    MatrixX scores = decision_function(X);

    for (int d=0; d<X.cols(); d++) {
        scores.col(d).maxCoeff( &predictions[d] );
    }

    return predictions;
}

template <typename Scalar>
void SupervisedLDA<Scalar>::compute_h(
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
Scalar SupervisedLDA<Scalar>::doc_e_step(
    const VectorXi &X,
    int y,
    const VectorX &alpha,
    const MatrixX &beta,
    const MatrixX &eta,
    Ref<MatrixX> phi,
    Ref<VectorX> gamma,
    size_t fixed_point_iterations,
    size_t e_step_iterations,
    Scalar e_step_tolerance
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

    while (e_step_iterations-- > 0) {
        compute_h(X, eta, phi, h);

        new_likelihood = compute_likelihood(X, y, alpha, beta, eta, phi, gamma, h);
        if ((new_likelihood - old_likelihood)/(-old_likelihood) < e_step_tolerance) {
            break;
        }
        old_likelihood = new_likelihood;

        // supervised inference
        if (y >= 0) {
            for (int i=0; i<fixed_point_iterations; i++) {
                phi_old = phi;

                auto t1 = gamma.unaryExpr(cwise_digamma);
                auto t2 = eta.col(y) * (X.cast<Scalar>().transpose() / num_words);
                // TODO: h.transpose() * phi_old can be cached
                auto t3 = h.array().rowwise() / (h.transpose() * phi_old).diagonal().transpose().array();

                phi = beta.array() * ((t2.colwise() + t1).array() - t3.array()).exp();
                phi.array() += 1.0;
                phi = phi.array().rowwise() / phi.colwise().sum().array();
            }
        }
        // unsupervised inference
        else {
            phi = beta.array() * gamma.unaryExpr(cwise_digamma).array().exp();
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
    Ref<MatrixX> b,
    Ref<VectorX> expected_z_bar
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
    Ref<MatrixX> beta,
    Ref<MatrixX> eta,
    Scalar L,
    Scalar m_step_iterations,
    Scalar m_step_tolerance
) {
    // we maximized w.r.t \beta during each doc_m_step
    beta = b;
    beta.array() += 1.0;
    beta = beta.array().rowwise() / beta.array().colwise().sum();

    // we need to maximize w.r.t to \eta
    Scalar initial_value = INFINITY;
    MultinomialLogisticRegression<Scalar> mlr(expected_z_bar, y, L);
    GradientDescent<MultinomialLogisticRegression<Scalar>, MatrixX> minimizer(
        std::make_shared<ArmijoLineSearch<MultinomialLogisticRegression<Scalar>, MatrixX> >(),
        [this, &initial_value, m_step_tolerance, m_step_iterations](
            Scalar value,
            Scalar gradNorm,
            size_t iterations
        ) {
            this->get_progress_visitor()->visit(Progress<Scalar>{
                ProgressState::Maximization,
                value,
                iterations,
                0
            });

            Scalar relative_improvement = (initial_value - value) / value;
            initial_value = value;

            return (
                iterations < m_step_iterations &&
                relative_improvement > m_step_tolerance
            );
        }
    );
    minimizer.minimize(mlr, eta);

    return initial_value;
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
    if (std::isnan(likelihood)) {
        std::cout << "1" << std::endl;
        return likelihood;
    }
    likelihood += std::lgamma(alpha.sum()) - alpha.unaryExpr(cwise_lgamma).sum();
    if (std::isnan(likelihood)) {
        std::cout << "2" << std::endl;
        return likelihood;
    }

    // E_q[log p(z | \theta)]
    likelihood += (phi.transpose() * t1).sum();
    if (std::isnan(likelihood)) {
        std::cout << "3" << std::endl;
        return likelihood;
    }

    // E_q[log p(w | z, \beta)]
    auto phi_scaled = phi.array().rowwise() * X.cast<Scalar>().transpose().array();
    likelihood += (phi_scaled * beta.array().log()).sum();
    if (std::isnan(likelihood)) {
        std::cout << "4" << std::endl;
        return likelihood;
    }

    // H(q)
    likelihood += -((gamma.array() - 1).matrix().transpose() * t1).value();
    if (std::isnan(likelihood)) {
        std::cout << "5" << std::endl;
        return likelihood;
    }
    likelihood += -std::lgamma(gamma.sum()) + gamma.unaryExpr(cwise_lgamma).sum();
    if (std::isnan(likelihood)) {
        std::cout << "6" << std::endl;
        return likelihood;
    }
    likelihood += -(phi.array() * phi.array().log()).sum();
    if (std::isnan(likelihood)) {
        std::cout << "7" << std::endl;
        return likelihood;
    }

    // unsupervised
    if (y < 0) {
        return likelihood;
    }

    // E_q[log p(y | z,n)] approximated using Jensens inequality
    likelihood += (eta.col(y).transpose() * phi * X.cast<Scalar>()).value() / X.sum();
    if (std::isnan(likelihood)) {
        std::cout << "8" << std::endl;
        return likelihood;
    }
    likelihood += - std::log((h.col(0).transpose() * phi.col(0)).value());
    if (std::isnan(likelihood)) {
        std::cout << "9" << std::endl;
        return likelihood;
    }

    return likelihood;
}


// Template instantiation
template class SupervisedLDA<float>;
template class SupervisedLDA<double>;
