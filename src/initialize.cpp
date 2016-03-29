#include <random>

#include "initialize.hpp"

template <typename T>
using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using VectorX = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template<typename Scalar>
void initialize_topics_seeded(
    const std::shared_ptr<Parameters> parameters,
    const std::shared_ptr<Corpus> corpus,
    size_t topics,
    int random_state
) {
    // Initialize alpha as 1/topics
    auto model_parameters = std::static_pointer_cast<ModelParameters<Scalar> >(parameters);
    model_parameters->alpha = VectorX<Scalar>::Constant(topics, 1.0 / topics); 
    
    // Initliaze beta in a seeded way
    model_parameters->beta = MatrixX<Scalar>::Constant(
        topics,
        (corpus->at(0)->get_words()).rows(),
        1.0
    ); 
    
    std::mt19937 rng(random_state);
    std::uniform_int_distribution<> initializations(10, 50);
    std::uniform_int_distribution<> document(
        0, 
        corpus->size()-1
    );
    auto N = initializations(rng);

    // Initialize _beta
    for (size_t k=0; k<topics; k++) {
        // Choose randomly a bunch of documents to initialize beta
        for (int r=0; r<N; r++) {
            model_parameters->beta.row(k) += corpus->at(document(rng))->get_words().cast<Scalar>().transpose();
        }
        model_parameters->beta.row(k) = model_parameters->beta.row(k) / model_parameters->beta.row(k).sum();
    }
}

template <typename Scalar>
void initialize_topics_random(
    const std::shared_ptr<Parameters> parameters,
    const std::shared_ptr<Corpus> corpus,
    size_t topics,
    int random_state
) {

}

template <typename Scalar>
void initialize_eta_zeros(
    const std::shared_ptr<Parameters> parameters,
    const std::shared_ptr<Corpus> corpus,
    size_t topics
) {
    auto model_parameters = std::static_pointer_cast<SupervisedModelParameters<Scalar> >(parameters);
    
    int max = -1;
    // Find the total number of different classes
    for (size_t i=0; i < corpus->size(); i++) {
        max = std::max(
            max,
            std::static_pointer_cast<ClassificationDocument>(
                corpus->at(i)
            )->get_class()
        );
    }
    
    model_parameters->eta = MatrixX<Scalar>::Zero(topics, max+1);
}

template <typename Scalar>
void initialize_eta_multinomial(
    const std::shared_ptr<Parameters> parameters,
    const std::shared_ptr<Corpus> corpus,
    size_t topics=600
) {
    auto model_parameters = std::static_pointer_cast<SupervisedModelParameters<Scalar> >(parameters);

    int max = -1;
    // Find the total number of different classes
    for (size_t i=0; i < corpus->size(); i++) {
        max = std::max(
            max,
            std::static_pointer_cast<ClassificationDocument>(
                corpus->at(i)
            )->get_class()
        );
    }

    model_parameters->eta = MatrixX<Scalar>::Constant(topics, max+1, 1. / (max+1));
}

// Template instantiation
template void initialize_topics_seeded<float>(
    const std::shared_ptr<Parameters>,
    const std::shared_ptr<Corpus>,
    size_t,
    int 
);
template void initialize_topics_seeded<double>(
    const std::shared_ptr<Parameters>,
    const std::shared_ptr<Corpus>,
    size_t,
    int
);
template void initialize_topics_random<float>(
    const std::shared_ptr<Parameters>,
    const std::shared_ptr<Corpus>,
    size_t,
    int
);
template void initialize_topics_random<double>(
    const std::shared_ptr<Parameters>,
    const std::shared_ptr<Corpus>,
    size_t,
    int
);
template void initialize_eta_zeros<float>(
    const std::shared_ptr<Parameters>,
    const std::shared_ptr<Corpus>,
    size_t
);
template void initialize_eta_zeros<double>(
    const std::shared_ptr<Parameters>,
    const std::shared_ptr<Corpus>,
    size_t
);
template void initialize_eta_multinomial<float>(
    const std::shared_ptr<Parameters>,
    const std::shared_ptr<Corpus>,
    size_t
);
template void initialize_eta_multinomial<double>(
    const std::shared_ptr<Parameters>,
    const std::shared_ptr<Corpus>,
    size_t
);
