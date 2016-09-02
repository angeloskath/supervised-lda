
#include <numeric>
#include <utility>

#include "ldaplusplus/Document.hpp"
#include "ldaplusplus/utils.hpp"

namespace ldaplusplus {
namespace corpus {


// 
// EigenDocument
//
EigenDocument::EigenDocument(Eigen::VectorXi X, std::shared_ptr<const Corpus> corpus)
    : X_(std::move(X)),
      corpus_(corpus)
{}

const std::shared_ptr<const Corpus> EigenDocument::get_corpus() const {
    return corpus_;
}

const Eigen::VectorXi & EigenDocument::get_words() const {
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

const std::shared_ptr<const Corpus> ClassificationDecorator::get_corpus() const {
    return document_->get_corpus();
}

const Eigen::VectorXi & ClassificationDecorator::get_words() const {
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
EigenCorpus::EigenCorpus(const Eigen::MatrixXi &X, int random_state)
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
    const Eigen::MatrixXi & X,
    const Eigen::VectorXi & y,
    int random_state
) : indices_(X.cols(), random_state),
    X_(X),
    y_(y),
    priors_(y.maxCoeff()+1)
{
    priors_.fill(0);
    for (int i=0; i<y_.rows(); i++) {
        priors_[y_[i]] ++;
    }
    priors_.array() /= priors_.sum();
}

size_t EigenClassificationCorpus::size() const {
    return X_.cols();
}

const std::shared_ptr<Document> EigenClassificationCorpus::at(size_t index) const {
    int i = indices_.get_index(index);

    return std::make_shared<ClassificationDecorator>(
        std::make_shared<EigenDocument>(
            X_.col(i),
            std::shared_ptr<const Corpus>(this, [](const Corpus*){})
        ),
        y_[i]
    );
}

void EigenClassificationCorpus::shuffle() {
    indices_.shuffle();
}

float EigenClassificationCorpus::get_prior(int y) const {
    return priors_[y];
}

}  // namespace corpus
}  // namespace ldaplusplus
