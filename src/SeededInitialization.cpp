#include <random>

#include "SeededInitialization.hpp"

template <typename Scalar>
void SeededInitialization<Scalar>::initialize_model_parameters(
    const MatrixXi &X,
    const VectorXi &y,
    Ref<VectorX> alpha,
    Ref<MatrixX> beta,
    Ref<MatrixX> eta
) {
    alpha = VectorX::Constant(topics_, 1.0 / topics_);
    // Eigen has no unique function, therefore we use maxcoeff instead to
    // calculate the number of classes, which is C
    eta = MatrixX::Zero(topics_, y.maxCoeff() + 1);
    
    beta = MatrixX::Constant(topics_, X.rows(), 1);
    std::mt19937 rng;
    rng.seed(0);
    std::uniform_int_distribution<> initializations(10, 50);
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
int SeededInitialization<Scalar>::get_id() {
    return IInitialization<Scalar>::Seeded;
}

template <typename Scalar>
std::vector<Scalar> SeededInitialization<Scalar>::get_parameters() {
    return {
        static_cast<Scalar>(topics_)
    };
}
template <typename Scalar>
void SeededInitialization<Scalar>::set_parameters(std::vector<Scalar> parameters) {
    topics_ = static_cast<size_t>(parameters[0]);
}
