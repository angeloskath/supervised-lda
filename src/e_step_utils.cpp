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
    VectorX<Scalar> h(phi.rows());
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
    const VectorX<Scalar> &h
) {
    Scalar likelihood = compute_unsupervised_likelihood(
        X,
        alpha,
        beta,
        phi,
        gamma
    );

    // E_q[log p(y | z,n)] approximated using Jensens inequality
    int n;
    for (n=X.rows()-1; n>=0; n--) {
        if (X[n] > 0)
            break;
    }
    likelihood += (eta.col(y).transpose() * phi * X.cast<Scalar>()).value() / X.sum();
    likelihood += - std::log((h.transpose() * phi.col(n)).value());

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
    Scalar prior_y,
    Scalar mu,
    Scalar portion
) {
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
    likelihood += (phi_scaled.colwise() * eta.col(y).array().log()).sum();
    likelihood -= (X.sum() - 1) * std::log(prior_y);

    // E_q[log p(\eta | \mu)]
    likelihood += portion * (mu - 1.0) * eta.array().log().sum();
    likelihood += portion * (std::lgamma(num_classes * mu) - num_classes * std::lgamma(mu));

    return likelihood;
}

template <typename Scalar>
Scalar compute_supervised_correspondence_likelihood(
    const VectorXi & X,
    int y,
    const VectorX<Scalar> &alpha,
    const MatrixX<Scalar> &beta,
    const MatrixX<Scalar> &eta,
    const MatrixX<Scalar> &phi,
    const VectorX<Scalar> &gamma,
    const VectorX<Scalar> &tau,
    Scalar mu,
    Scalar portion
) {
    Scalar likelihood = compute_unsupervised_likelihood(
        X,
        alpha,
        beta,
        phi,
        gamma
    );

    // Add extra term in H(q)
    likelihood += -(tau.array() * X.cast<Scalar>().array() * (tau.array() + 1e-44).log()).sum();

    // E_q[log p(y | \lambda, \eta, z)]
    auto phi_scaled = phi.array().rowwise() * (X.cast<Scalar>().transpose().array() * tau.transpose().array());
    likelihood += (phi_scaled.colwise() * eta.col(y).array().log()).sum();

    // E_q[log p(\lambda | N)]
    likelihood += -std::log(X.sum());

    // E_q[log p(\eta | \mu)]
    int num_classes = eta.cols();
    likelihood += portion * (mu - 1.0) * eta.array().log().sum();
    likelihood += portion * (std::lgamma(num_classes * mu) - num_classes * std::lgamma(mu));

    return likelihood;
}

template <typename Scalar>
void compute_h(
    const VectorXi & X,
    const VectorX<Scalar> & X_ratio,
    const MatrixX<Scalar> &eta,
    const MatrixX<Scalar> &phi,
    Ref<VectorX<Scalar> > h
) {
    auto cwise_fast_exp = CwiseFastExp<Scalar>();

    MatrixX<Scalar> exp_eta_scaled(eta.rows(), eta.cols());
    VectorX<Scalar> products = VectorX<Scalar>::Constant(eta.cols(), 1.0);

    // Compute the products that will allow us to compute h for the last word
    for (int n=0; n<X.rows(); n++) {
        if (X[n] == 0)
            continue;

        exp_eta_scaled = (eta * X_ratio[n]).unaryExpr(cwise_fast_exp);
        //exp_eta_scaled = (eta * X_ratio[n]).array().exp();
        products.array() *= (exp_eta_scaled.transpose() * phi.col(n)).array();
    }

    // Traverse words in reverse order in order to find the "last word" of the
    // document, that occurs at least one time in the document 
    for (int n=X.rows()-1; n>=0; n--) {
        // Skip this word if it is not in the document
        if (X[n] == 0)
            continue;

        exp_eta_scaled = (eta * X_ratio[n]).unaryExpr(cwise_fast_exp);
        //exp_eta_scaled = (eta * X_ratio[n]).array().exp();

        // Remove the nth word
        products.array() /= (exp_eta_scaled.transpose() * phi.col(n)).array();

        // Compute h w.r.t phi_n
        h = exp_eta_scaled * products;
        break;
    }
}

template <typename Scalar>
void compute_supervised_phi_gamma(
    const VectorXi & X,
    const VectorX<Scalar> & X_ratio,
    int y,
    const MatrixX<Scalar> & beta,
    const MatrixX<Scalar> & eta,
    size_t fixed_point_iterations,
    Ref<MatrixX<Scalar> > phi,
    Ref<VectorX<Scalar> > gamma,
    Ref<VectorX<Scalar> > h
) {
    auto cwise_digamma = CwiseDigamma<Scalar>();
    auto cwise_fast_exp = CwiseFastExp<Scalar>();

    MatrixX<Scalar> exp_eta_scaled(eta.rows(), eta.cols());
    VectorX<Scalar> products = VectorX<Scalar>::Constant(eta.cols(), 1.0);
    VectorX<Scalar> psi_gamma = gamma.unaryExpr(cwise_digamma);
    VectorX<Scalar> old_phi_n = VectorX<Scalar>::Zero(eta.cols());

    // Compute the products that will allow us to compute and update h
    for (int n=0; n<X.rows(); n++) {
        exp_eta_scaled = (eta * X_ratio[n]).unaryExpr(cwise_fast_exp);
        //exp_eta_scaled = (eta * X_ratio[n]).array().exp();
        products.array() *= (exp_eta_scaled.transpose() * phi.col(n)).array();
    }

    for (int n=0; n<X.rows(); n++) {
        // Skip this word if it is not in the document
        if (X[n] == 0)
            continue;

        exp_eta_scaled = (eta * X_ratio[n]).unaryExpr(cwise_fast_exp);
        //exp_eta_scaled = (eta * X_ratio[n]).array().exp();

        // Remove the nth word
        products.array() /= (exp_eta_scaled.transpose() * phi.col(n)).array();

        // Compute h w.r.t phi_n
        h = exp_eta_scaled * products;

        // Fixed point iterations
        old_phi_n = phi.col(n);
        for (size_t i=0; i<fixed_point_iterations; i++) {
            Scalar t = 1. / (h.transpose() * phi.col(n)).value() / X[n];

            phi.col(n).array() = beta.col(n).array() * (
                psi_gamma + X_ratio[n]*eta.col(y) - h * t
            ).array().unaryExpr(cwise_fast_exp);
            //phi.col(n).array() = beta.col(n).array() * (
            //    psi_gamma + X_ratio[n]*eta.col(y) - h * t
            //).array().exp();

            phi.col(n).array() /= phi.col(n).sum();
        }

        // Recompute the products with the updated phi
        products.array() *= (exp_eta_scaled.transpose() * phi.col(n)).array();

        // Recompute gamma by removing the old phi and inserting the new one
        gamma += X[n]*(phi.col(n) - old_phi_n);
        psi_gamma = gamma.unaryExpr(cwise_digamma);
    }
}

template <typename Scalar>
void compute_gamma(
    const VectorXi & X,
    const VectorX<Scalar> & alpha,
    const MatrixX<Scalar> & phi,
    Ref<VectorX<Scalar> > gamma
) {
    // We implement the following line in a safer manner to avoid possible NaN by 0 * inf
    //
    // gamma = alpha.array() + (phi.array().rowwise() * X.cast<Scalar>().transpose().array()).rowwise().sum();

    gamma = alpha;
    sum_cols_scaled(phi, X, gamma);
}

template <typename Scalar>
void compute_unsupervised_phi(
    const MatrixX<Scalar> & beta,
    const VectorX<Scalar> & gamma,
    Ref<MatrixX<Scalar> > phi
) {
    auto cwise_digamma = CwiseDigamma<Scalar>();
    auto cwise_fast_exp = CwiseFastExp<Scalar>();

    auto exp_psi_gamma = gamma.unaryExpr(cwise_digamma).unaryExpr(cwise_fast_exp).array();
    phi = beta.array().colwise() * exp_psi_gamma;
    //phi = phi.array().rowwise() / phi.colwise().sum().array();
    normalize_cols(phi);
}

template <typename Scalar>
void compute_supervised_approximate_phi(
    const VectorX<Scalar> & X_ratio,
    int num_words,
    int y,
    const MatrixX<Scalar> & beta,
    const MatrixX<Scalar> & eta,
    const VectorX<Scalar> & gamma,
    Scalar C,
    Ref<MatrixX<Scalar> > phi
) {
    auto cwise_digamma = CwiseDigamma<Scalar>();
    auto cwise_fast_exp = CwiseFastExp<Scalar>();

    auto psi_gamma = gamma.unaryExpr(cwise_digamma).array();
    VectorX<Scalar> z_bar = VectorX<Scalar>::Zero(phi.rows());
    sum_cols_scaled(phi, X_ratio, z_bar);

    VectorX<Scalar> softmax_eta_z = (eta.transpose() * z_bar).unaryExpr(cwise_fast_exp);
    softmax_eta_z = softmax_eta_z / softmax_eta_z.sum();

    Scalar max_eta = eta.maxCoeff();
    MatrixX<Scalar> eta_scaled = eta;
    if (max_eta > 0)
        eta_scaled /= max_eta;

    phi = beta.array().colwise() * (
        psi_gamma + C*(eta_scaled.col(y) - eta_scaled * softmax_eta_z).array()
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
    Scalar eta_weight,
    Ref<MatrixX<Scalar> > phi
) {
    auto cwise_digamma = CwiseDigamma<Scalar>();
    auto cwise_fast_exp = CwiseFastExp<Scalar>();

    auto t1 = gamma.unaryExpr(cwise_digamma).array();
    auto t2 = digamma(gamma.sum()) + 1;

    phi = beta.array().colwise() * (
        (eta_weight*eta.col(y).array().log() + t1 - t2).unaryExpr(cwise_fast_exp).array()
    );
    //phi = phi.array().rowwise() / phi.colwise().sum().array();
    normalize_cols(phi);
}

template <typename Scalar>
void compute_supervised_correspondence_phi(
    const VectorXi & X,
    int y,
    const MatrixX<Scalar> & beta,
    const MatrixX<Scalar> & eta,
    const VectorX<Scalar> & gamma,
    const VectorX<Scalar> & tau,
    Ref<MatrixX<Scalar> > phi
) {
    auto cwise_digamma = CwiseDigamma<Scalar>();
    auto cwise_fast_exp = CwiseFastExp<Scalar>();

    auto t1 = gamma.unaryExpr(cwise_digamma).array();

    phi = (beta.array().colwise() * t1.unaryExpr(cwise_fast_exp)).array() *
        (eta.col(y).array().log().matrix() * tau.transpose()).unaryExpr(cwise_fast_exp).array();
    //phi = phi.array().rowwise() / phi.colwise().sum().array();
    normalize_cols(phi);
}

template <typename Scalar>
void compute_supervised_correspondence_tau(
    const VectorXi & X,
    int y,
    const MatrixX<Scalar> & eta,
    const MatrixX<Scalar> & phi,
    Ref<VectorX<Scalar> > tau
) {
    auto cwise_fast_exp = CwiseFastExp<Scalar>();

    tau = (
        phi.transpose() * eta.col(y).array().log().matrix()
    ).unaryExpr(cwise_fast_exp);
    // tau = tau.array() / tau.sum();
    normalize_cols(tau);
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
    const VectorX<float> &h
);
template double compute_supervised_likelihood(
    const VectorXi & X,
    int y,
    const VectorX<double> &alpha,
    const MatrixX<double> &beta,
    const MatrixX<double> &eta,
    const MatrixX<double> &phi,
    const VectorX<double> &gamma,
    const VectorX<double> &h
);
template float compute_supervised_multinomial_likelihood(
    const VectorXi & X,
    int y,
    const VectorX<float> &alpha,
    const MatrixX<float> &beta,
    const MatrixX<float> &eta,
    const MatrixX<float> &phi,
    const VectorX<float> &gamma,
    float prior_y,
    float mu,
    float portion
);
template double compute_supervised_multinomial_likelihood(
    const VectorXi & X,
    int y,
    const VectorX<double> &alpha,
    const MatrixX<double> &beta,
    const MatrixX<double> &eta,
    const MatrixX<double> &phi,
    const VectorX<double> &gamma,
    double prior_y,
    double mu,
    double portion
);
template float compute_supervised_correspondence_likelihood(
    const VectorXi & X,
    int y,
    const VectorX<float> &alpha,
    const MatrixX<float> &beta,
    const MatrixX<float> &eta,
    const MatrixX<float> &phi,
    const VectorX<float> &gamma,
    const VectorX<float> &tau,
    float mu,
    float portion
);
template double compute_supervised_correspondence_likelihood(
    const VectorXi & X,
    int y,
    const VectorX<double> &alpha,
    const MatrixX<double> &beta,
    const MatrixX<double> &eta,
    const MatrixX<double> &phi,
    const VectorX<double> &gamma,
    const VectorX<double> &tau,
    double mu,
    double portion
);
template void compute_h(
    const VectorXi & X,
    const VectorX<float> & X_ratio,
    const MatrixX<float> &eta,
    const MatrixX<float> &phi,
    Ref<VectorX<float> > h
);
template void compute_h(
    const VectorXi & X,
    const VectorX<double> & X_ratio,
    const MatrixX<double> &eta,
    const MatrixX<double> &phi,
    Ref<VectorX<double> > h
);
template void compute_supervised_phi_gamma(
    const VectorXi & X,
    const VectorX<float> & X_ratio,
    int y,
    const MatrixX<float> & beta,
    const MatrixX<float> & eta,
    size_t fixed_point_iterations,
    Ref<MatrixX<float> > phi,
    Ref<VectorX<float> > gamma,
    Ref<VectorX<float> > h
);
template void compute_supervised_phi_gamma(
    const VectorXi & X,
    const VectorX<double> & X_ratio,
    int y,
    const MatrixX<double> & beta,
    const MatrixX<double> & eta,
    size_t fixed_point_iterations,
    Ref<MatrixX<double> > phi,
    Ref<VectorX<double> > gamma,
    Ref<VectorX<double> > h
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
    float C,
    Ref<MatrixX<float> > phi
);
template void compute_supervised_approximate_phi(
    const VectorX<double> & X_ratio,
    int num_words,
    int y,
    const MatrixX<double> & beta,
    const MatrixX<double> & eta,
    const VectorX<double> & gamma,
    double C,
    Ref<MatrixX<double> > phi
);
template void compute_supervised_multinomial_phi(
    const VectorXi & X,
    int y,
    const MatrixX<float> & beta,
    const MatrixX<float> & eta,
    const VectorX<float> & gamma,
    float eta_weight,
    Ref<MatrixX<float> > phi
);
template void compute_supervised_multinomial_phi(
    const VectorXi & X,
    int y,
    const MatrixX<double> & beta,
    const MatrixX<double> & eta,
    const VectorX<double> & gamma,
    double eta_weight,
    Ref<MatrixX<double> > phi
);
template void compute_supervised_correspondence_phi(
    const VectorXi & X,
    int y,
    const MatrixX<float> & beta,
    const MatrixX<float> & eta,
    const VectorX<float> & gamma,
    const VectorX<float> & tau,
    Ref<MatrixX<float> > phi
);
template void compute_supervised_correspondence_phi(
    const VectorXi & X,
    int y,
    const MatrixX<double> & beta,
    const MatrixX<double> & eta,
    const VectorX<double> & gamma,
    const VectorX<double> & tau,
    Ref<MatrixX<double> > phi
);
template void compute_supervised_correspondence_tau(
    const VectorXi & X,
    int y,
    const MatrixX<float> & eta,
    const MatrixX<float> & phi,
    Ref<VectorX<float> > tau
);
template void compute_supervised_correspondence_tau(
    const VectorXi & X,
    int y,
    const MatrixX<double> & eta,
    const MatrixX<double> & phi,
    Ref<VectorX<double> > tau
);
}
