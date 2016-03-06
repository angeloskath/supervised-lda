
#include <numeric>

#include "Document.hpp"

// 
// EigenDocument
//
EigenDocument::EigenDocument(const VectorXi &X) : X_(X) {}

const VectorXi & EigenDocument::get_words() const {
    return X_;
}


// 
// ClassificationDecorator
//
ClassificationDecorator::ClassificationDecorator(
    std::shared_ptr<Document> doc,
    int y
) : document_(doc),
    y_(y)
{}

const VectorXi & ClassificationDecorator::get_words() const {
    return document_->get_words();
}

int ClassificationDecorator::get_class() const {
    return y_;
}


// 
// CorpusIndexes
//
CorpusIndexes::CorpusIndexes(int N, int random_state)
    : indices_(N),
      prng_(random_state)
{
    std::iota(indices_.begin(), indices_.end(), 0);
}

void CorpusIndexes::shuffle() {
    std::shuffle(indices_.begin(), indices_.end(), prng_);
}


// 
// EigenCorpus
//
EigenCorpus::EigenCorpus(const MatrixXi &X, int random_state)
    : indices_(X.cols(), random_state),
      X_(X)
{}

size_t EigenCorpus::size() const {
    return X_.cols();
}

const std::shared_ptr<Document> EigenCorpus::at(size_t index) const {
    int i = indices_.get_index(index);

    return std::make_shared<EigenDocument>(X_.col(i));
}

void EigenCorpus::shuffle() {
    indices_.shuffle();
}


// 
// EigenClassificationCorpus
//
EigenClassificationCorpus::EigenClassificationCorpus(
    const MatrixXi & X,
    const VectorXi & y,
    int random_state
) : EigenCorpus(X, random_state),
    y_(y)
{}

const std::shared_ptr<Document> EigenClassificationCorpus::at(size_t index) const {
    int i = indices_.get_index(index);

    return std::make_shared<ClassificationDecorator>(
        std::make_shared<EigenDocument>(X_.col(i)),
        y_[i]
    );
}
