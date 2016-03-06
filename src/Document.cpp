
#include <numeric>

#include "Document.hpp"


EigenDocument::EigenDocument(const VectorXi &X) : X_(X) {}

const VectorXi & EigenDocument::get_words() const {
    return X_;
}


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


EigenClassificationCorpus::EigenClassificationCorpus(
    const MatrixXi & X,
    const VectorXi & y,
    int random_state
) : X_(X),
    y_(y),
    indices_(y.rows()),
    prng_(random_state)
{
    std::iota(indices_.begin(), indices_.end(), 0);
}

size_t EigenClassificationCorpus::size() const {
    return indices_.size();
}

const std::shared_ptr<Document> EigenClassificationCorpus::operator[] (size_t index) const {
    int i = indices_[index];

    return std::make_shared<ClassificationDecorator>(
        std::make_shared<EigenDocument>(X_.col(i)),
        y_[i]
    );
}

void EigenClassificationCorpus::shuffle() {
    std::shuffle(indices_.begin(), indices_.end(), prng_);
}
