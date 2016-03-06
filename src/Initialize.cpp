#include <random>
#include <vector>

#include "Initialize.hpp"

template<typename Scalar>
void initialize_topics_seeded(
    const std::shared_ptr<Parameters> parameters,
    const std::shared_ptr<Corpus> corpus,
    size_t topics,
    int random_state
) {
    // Initialize alpha as 1/topics
    auto model_parameters = std::static_pointer_cast<ModelParameters<Scalar> >(parameters);
    model_parameters->alpha = VectorX::Constant(topics, 1.0 / topics); 
    
    // Initliaze beta in a seeded way
    model_parameters->beta = MatrixX::Constant(
        topics,
        (corpus->at(0)->get_words()).rows(),
        1.0
    ); 
    
    std::mt19937 rng(random_state);
    std::uniform_int_distribution<> initializations(10, 50);
    std::uniform_int_distribution<> document(
        0, 
        corpus.size()-1
    );
    auto N = initializations(rng);

    // Initialize _beta
    for (size_t k=0; k<topics; k++) {
        // Choose randomly a bunch of documents to initialize beta
        for (int r=0; r<N; r++) {
            model_parameters->beta.row(k) += (corpus->at(document(rng))->get_words()).transpose() 
        }
        model_parameters->beta.row(k) = model_parameters->beta.row(k) / model_parameters->beta.row(k).sum();
    }
}

template <typename Scalar>
void initialize_topics_random(
    const std::shared_ptr<Parameters> parameters,
    const std::shared_ptr<Corpus> corpus,
    size_t topics
) {

}

template <typename Scalar>
void initialize_eta_zeros(
    const std::shared_ptr<Parameters> parameters,
    const std::shared_ptr<Corpus> corpus,
    size_t topics
) {
    auto model_parameters = std::static_pointer_cast<SupervisedModelParameters<Scalar> >(parameters);
    
    std::vector<int> class_labels;
    // Find the total number of different classes
    for (int i=0; i < corpus->size(); i++) {
       class_labels.push_back(corpus->at(i)->get_class());
    }
    
    // Remove duplicates
    std::sort(class_labels.begin(), class_labels.end());
    std::erase(
        std::unique(class_labels.begin(), class_labels.end()),
        class_labels.end()
    );

    model_parameters->eta = MatrixX::Zero(topics, class_labels.size()+1);
}

// Template instantiation
template void initialize_topics_seeded<float>;
template void initialize_topics_seeded<double>;
