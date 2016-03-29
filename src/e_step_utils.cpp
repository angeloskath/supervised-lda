#include "e_step_utils.hpp"
#include "utils.hpp"

namespace e_step_utils
{


template <typename Scalar>
Scalar compute_unsupervised_likelihood(
    const VectorXi & X,
    const VectorX<Scalar> &alpha,
    const MatrixX<Scalar> &beta,
    const MatrixX<Scalar> &phi,
    const VectorX<Scalar> &gamma
) {
    auto cwise_digamma = CwiseDigamma<Scalar>();
    auto cwise_lgamma = CwiseLgamma<Scalar>();

    Scalar likelihood = 0;

    // \Psi(\gamma) - \Psi(\sum_j \gamma)
    VectorX<Scalar> t1 = gamma.unaryExpr(cwise_digamma).array() - digamma(gamma.sum());

    // E_q[log p(\theta | \alpha)]
    likelihood += ((alpha.array() - 1.0).matrix().transpose() * t1).value();
    likelihood += std::lgamma(alpha.sum()) - alpha.unaryExpr(cwise_lgamma).sum();

    // E_q[log p(z | \theta)]
    likelihood += (phi.transpose() * t1).sum();

    // E_q[log p(w | z, \beta)]
    auto phi_scaled = phi.array().rowwise() * X.cast<Scalar>().transpose().array();
    likelihood += (phi_scaled * (beta.array() + 1e-44).log()).sum();

    // H(q)
    likelihood += -((gamma.array() - 1).matrix().transpose() * t1).value();
    likelihood += -std::lgamma(gamma.sum()) + gamma.unaryExpr(cwise_lgamma).sum();
    likelihood += -(phi.array() * (phi.array() + 1e-44).log()).sum();
    
    return likelihood;
}


template <typename Scalar>
Scalar compute_supervised_likelihood(
    const VectorXi & X,
    int y,
    const VectorX<Scalar> &alpha,
    const MatrixX<Scalar> &beta,
    const MatrixX<Scalar> &eta,
    const MatrixX<Scalar> &phi,
    const VectorX<Scalar> &gamma
) {
    // Computing h is an overkill and should be changed
    MatrixX<Scalar> h(phi.rows(), phi.cols());
    VectorX<Scalar> X_ratio = X.cast<Scalar>() / X.sum();
    compute_h<Scalar>(X, X_ratio, eta, phi, h);

    return compute_supervised_likelihood(X, y, alpha, beta, eta, phi, gamma, h);
}
template <typename Scalar>
Scalar compute_supervised_likelihood(
    const VectorXi & X,
    int y,
    const VectorX<Scalar> &alpha,
    const MatrixX<Scalar> &beta,
    const MatrixX<Scalar> &eta,
    const MatrixX<Scalar> &phi,
    const VectorX<Scalar> &gamma,
    const MatrixX<Scalar> &h
) {
    Scalar likelihood = compute_unsupervised_likelihood(
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

template <typename Scalar>
Scalar compute_supervised_multinomial_likelihood(
    const VectorXi & X,
    int y,
    const VectorX<Scalar> &alpha,
    const MatrixX<Scalar> &beta,
    const MatrixX<Scalar> &eta,
    const MatrixX<Scalar> &phi,
    const VectorX<Scalar> &gamma,
    Scalar mu
) {
    auto cwise_lgamma = CwiseLgamma<Scalar>();

    Scalar likelihood = compute_unsupervised_likelihood(
        X,
        alpha,
        beta,
        phi,
        gamma
    );
    int num_classes = eta.cols();

    // E_q[log p(y | z, \eta)]
    auto phi_scaled = phi.array().rowwise() * X.cast<Scalar>().transpose().array();
    likelihood += (phi_scaled.transpose() * eta.col(y).array().log().matrix()).sum();
    
    // E_q[log p(\eta | \mu)]
    likelihood += (mu - 1.0) * eta.array().log().sum();
    likelihood += std::lgamma(num_classes * mu) - num_classes * mu.unaryExpr(cwise_lgamma);

    return likelihood;
}

template <typename Scalar>
void compute_h(
    const VectorXi & X,
    const VectorX<Scalar> & X_ratio,
    const MatrixX<Scalar> &eta,
    const MatrixX<Scalar> &phi,
    Ref<MatrixX<Scalar> > h
) {
    auto cwise_fast_exp = CwiseFastExp<Scalar>();

    MatrixX<Scalar> exp_eta(h.rows(), h.cols());
    VectorX<Scalar> products(phi.cols());
    int num_classes = eta.cols();

    h.fill(0);
    for (int y=0; y<num_classes; y++) {
        auto eta_scaled = eta.col(y) * X_ratio.transpose();
        //exp_eta = eta_scaled.array().exp();
        exp_eta = eta_scaled.array().unaryExpr(cwise_fast_exp);
        products = (exp_eta.transpose() * phi).diagonal();

        // products.prod() / products.array() accounting for the zeros in phi
        auto t1 = products.unaryExpr(
            CwiseScalarDivideByMatrix<Scalar>(
                product_of_nonzeros(products)
            )
        );
        auto t2 = exp_eta * t1.asDiagonal();

        h += t2;
    }
}

template <typename Scalar>
void fixed_point_iteration(
    const VectorX<Scalar> & X_ratio,
    int y,
    const MatrixX<Scalar> & beta,
    const MatrixX<Scalar> & eta,
    const VectorX<Scalar> & gamma,
    const MatrixX<Scalar> &h,
    Ref<MatrixX<Scalar> > phi_old,
    Ref<MatrixX<Scalar> > phi
) {
    auto cwise_digamma = CwiseDigamma<Scalar>();
    auto cwise_fast_exp = CwiseFastExp<Scalar>();

    // Frist thing 's first copy the phi to the old phi
    phi_old = phi;

    auto t1 = gamma.unaryExpr(cwise_digamma);
    auto t2 = eta.col(y) * X_ratio.transpose();
    // TODO: h.transpose() * phi_old can be cached
    // auto t3 = h.array().rowwise() / (h.transpose() * phi_old).diagonal().transpose().array();
    auto t3 = h / (h.col(0).transpose() * phi_old.col(0)).value();

    phi = beta.array() * ((t2.colwise() + t1).array() - t3.array()).unaryExpr(cwise_fast_exp);
    //phi = beta.array() * ((t2.colwise() + t1).array() - t3.array()).exp();
    normalize_cols(phi);
}

template <typename Scalar>
void compute_gamma(
    const VectorXi & X,
    const VectorX<Scalar> & alpha,
    const MatrixX<Scalar> & phi,
    Ref<VectorX<Scalar> > gamma
) {
    gamma = alpha.array() + (phi.array().rowwise() * X.cast<Scalar>().transpose().array()).rowwise().sum();
}

template <typename Scalar>
void compute_unsupervised_phi(
    const MatrixX<Scalar> & beta,
    const VectorX<Scalar> & gamma,
    Ref<MatrixX<Scalar> > phi
) {
    auto cwise_digamma = CwiseDigamma<Scalar>();

    auto t1 = gamma.unaryExpr(cwise_digamma).array();
    auto t2 = digamma(gamma.sum());
    phi = beta.array().colwise() * (t1 - t2).exp();
    phi = phi.array().rowwise() / phi.colwise().sum().array();
}

template <typename Scalar>
void compute_supervised_approximate_phi(
    const VectorX<Scalar> & X_ratio,
    int num_words,
    int y,
    const MatrixX<Scalar> & beta,
    const MatrixX<Scalar> & eta,
    const VectorX<Scalar> & gamma,
    Ref<MatrixX<Scalar> > phi
) {
    auto cwise_digamma = CwiseDigamma<Scalar>();
    auto cwise_fast_exp = CwiseFastExp<Scalar>();

    MatrixX<Scalar> z_bar = (
        phi.array().rowwise() * X_ratio.transpose().array()
    ).rowwise().sum();
    MatrixX<Scalar> t = (eta.transpose() * z_bar).unaryExpr(cwise_fast_exp);
    t = t.array().rowwise() / t.colwise().sum().array();

    auto t1 = gamma.unaryExpr(cwise_digamma).array();
    auto t2 = digamma(gamma.sum()) + 1;

    phi = beta.array().colwise() * (
        t1 - t2 + (eta.col(y) - eta * t).array() / num_words
    ).unaryExpr(cwise_fast_exp).array();
    //phi = phi.array().rowwise() / phi.colwise().sum().array();
    normalize_cols(phi);
}

template <typename Scalar>
void compute_supervised_multinomial_phi(
    const VectorXi & X,
    int y,
    const MatrixX<Scalar> & beta,
    const MatrixX<Scalar> & eta,
    const VectorX<Scalar> & gamma,
    Ref<MatrixX<Scalar> > phi
) {
    auto cwise_digamma = CwiseDigamma<Scalar>();
    auto cwise_fast_exp = CwiseFastExp<Scalar>();

    auto t1 = gamma.unaryExpr(cwise_digamma).array();
    auto t2 = digamma(gamma.sum()) + 1;

    phi = beta.array().colwise() * (
        (t1 - t2).unaryExpr(cwise_fast_exp).array() * eta.col(y).array()
    );
    //phi = phi.array().rowwise() / phi.colwise().sum().array();
    normalize_cols(phi);
}

// Template instantiations
template float compute_unsupervised_likelihood(
    const VectorXi & X,
    const VectorX<float> &alpha,
    const MatrixX<float> &beta,
    const MatrixX<float> &phi,
    const VectorX<float> &gamma
);
template double compute_unsupervised_likelihood(
    const VectorXi & X,
    const VectorX<double> &alpha,
    const MatrixX<double> &beta,
    const MatrixX<double> &phi,
    const VectorX<double> &gamma
);
template float compute_supervised_likelihood(
    const VectorXi & X,
    int y,
    const VectorX<float> &alpha,
    const MatrixX<float> &beta,
    const MatrixX<float> &eta,
    const MatrixX<float> &phi,
    const VectorX<float> &gamma
);
template double compute_supervised_likelihood(
    const VectorXi & X,
    int y,
    const VectorX<double> &alpha,
    const MatrixX<double> &beta,
    const MatrixX<double> &eta,
    const MatrixX<double> &phi,
    const VectorX<double> &gamma
);
template float compute_supervised_likelihood(
    const VectorXi & X,
    int y,
    const VectorX<float> &alpha,
    const MatrixX<float> &beta,
    const MatrixX<float> &eta,
    const MatrixX<float> &phi,
    const VectorX<float> &gamma,
    const MatrixX<float> &h
);
template double compute_supervised_likelihood(
    const VectorXi & X,
    int y,
    const VectorX<double> &alpha,
    const MatrixX<double> &beta,
    const MatrixX<double> &eta,
    const MatrixX<double> &phi,
    const VectorX<double> &gamma,
    const MatrixX<double> &h
);
template void compute_h(
    const VectorXi & X,
    const VectorX<float> & X_ratio,
    const MatrixX<float> &eta,
    const MatrixX<float> &phi,
    Ref<MatrixX<float> > h
);
template void compute_h(
    const VectorXi & X,
    const VectorX<double> & X_ratio,
    const MatrixX<double> &eta,
    const MatrixX<double> &phi,
    Ref<MatrixX<double> > h
);

template void fixed_point_iteration(
    const VectorX<float> & X_ratio,
    int y,
    const MatrixX<float> & beta,
    const MatrixX<float> & eta,
    const VectorX<float> & gamma,
    const MatrixX<float> &h,
    Ref<MatrixX<float> > phi_old,
    Ref<MatrixX<float> > phi
);
template void fixed_point_iteration(
    const VectorX<double> & X_ratio,
    int y,
    const MatrixX<double> & beta,
    const MatrixX<double> & eta,
    const VectorX<double> & gamma,
    const MatrixX<double> &h,
    Ref<MatrixX<double> > phi_old,
    Ref<MatrixX<double> > phi
);
template void compute_gamma(
    const VectorXi & X,
    const VectorX<float> & alpha,
    const MatrixX<float> & phi,
    Ref<VectorX<float> > gamma
);
template void compute_gamma(
    const VectorXi & X,
    const VectorX<double> & alpha,
    const MatrixX<double> & phi,
    Ref<VectorX<double> > gamma
);
template void compute_unsupervised_phi(
    const MatrixX<float> & beta,
    const VectorX<float> & gamma,
    Ref<MatrixX<float> > phi
);
template void compute_unsupervised_phi(
    const MatrixX<double> & beta,
    const VectorX<double> & gamma,
    Ref<MatrixX<double> > phi
);
template void compute_supervised_approximate_phi(
    const VectorX<float> & X_ratio,
    int num_words,
    int y,
    const MatrixX<float> & beta,
    const MatrixX<float> & eta,
    const VectorX<float> & gamma,
    Ref<MatrixX<float> > phi
);
template void compute_supervised_approximate_phi(
    const VectorX<double> & X_ratio,
    int num_words,
    int y,
    const MatrixX<double> & beta,
    const MatrixX<double> & eta,
    const VectorX<double> & gamma,
    Ref<MatrixX<double> > phi
);
template void compute_supervised_multinomial_phi(
    const VectorXi & X,
    int y,
    const MatrixX<float> & beta,
    const MatrixX<float> & eta,
    const VectorX<float> & gamma,
    Ref<MatrixX<float> > phi
);
template void compute_supervised_multinomial_phi(
    const VectorXi & X,
    int y,
    const MatrixX<double> & beta,
    const MatrixX<double> & eta,
    const VectorX<double> & gamma,
    Ref<MatrixX<double> > phi
);
}
