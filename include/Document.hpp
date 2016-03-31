#ifndef _DOCUMENT_HPP_
#define _DOCUMENT_HPP_


#include <memory>
#include <random>
#include <vector>

#include <Eigen/Core>

using namespace Eigen;

// Forward declaration for the compiler
class Corpus;


/**
 * A Document is the minimal document needed for any type of LDA
 * implementation.
 */
class Document
{
    public:
        virtual const std::shared_ptr<const Corpus> get_corpus() const = 0;
        virtual const VectorXi & get_words() const = 0;

        template <typename T>
        const std::shared_ptr<const T> get_corpus() const {
            return std::static_pointer_cast<const T>(get_corpus());
        }
};


/**
 * The classification Document contains the words and a category in the form of
 * an integer.
 */
class ClassificationDocument : public Document
{
    public:
        virtual int get_class() const = 0;
};


/**
 * A corpus is a collection of documents.
 */
class Corpus
{
    public:
        /** The number of documents in the corpus */
        virtual size_t size() const = 0;
        /** The ith document */
        virtual const std::shared_ptr<Document> at(size_t index) const = 0;
        /**
         * Shuffle the documents so that the ith document is any of the
         * documents with probability 1.0/size() .
         */
        virtual void shuffle() = 0;
};


/**
 * A corpus that contains information about the documents' classes (namely
 * their priors).
 */
class ClassificationCorpus : public Corpus
{
    public:
        virtual float get_prior(int y) const = 0;
};


/**
 * Eigen Document is a document backed by an Eigen::VectorXi.
 */
class EigenDocument : public Document
{
    public:
        EigenDocument(VectorXi X) : EigenDocument(X, nullptr) {}
        EigenDocument(VectorXi X, std::shared_ptr<const Corpus> corpus);

        const std::shared_ptr<const Corpus> get_corpus() const override;
        const VectorXi & get_words() const override;

    private:
        VectorXi X_;
        std::shared_ptr<const Corpus> corpus_;
};


/**
 * ClassificationDecorator decorates any Document with classification
 * information.
 */
class ClassificationDecorator : public ClassificationDocument
{
    public:
        ClassificationDecorator(std::shared_ptr<Document> doc, int y);

        const std::shared_ptr<const Corpus> get_corpus() const override;
        const VectorXi & get_words() const override;
        int get_class() const override;

    private:
        std::shared_ptr<Document> document_;
        int y_;
};


/**
 * Implement a shuffle method that shuffles indexes and provides them to
 * classes who want to implement the corpus interface.
 */
class CorpusIndexes
{
    public:
        CorpusIndexes(int N, int random_state=0);

        /**
         * Return the document index that corresponds to the sequential index
         * after a shuffle.
         */
        int get_index(int index) const { return indices_[index]; }

        /**
         * Shuffle the indexes around.
         */
        void shuffle();

    private:
        /**
         * A vector of indices that implements the function mapping the ith in
         * sequence document to a jth actual document
         */
        std::vector<int> indices_;

        /** A pseudo random number generator for the shuffling */
        std::mt19937 prng_;
};


/**
 * Wrap a matrix X and implement the corpus interface.
 */
class EigenCorpus : public Corpus
{
    public:
        EigenCorpus(const MatrixXi & X, int random_state=0);

        size_t size() const override;
        virtual const std::shared_ptr<Document> at(size_t index) const override;
        void shuffle() override;

    protected:
        /** To implement shuffle */
        CorpusIndexes indices_;

        /**
         * We keep X_ protected instead of private to reduce boilerplate in
         * creating a supervised version of this Corpus.
         */
        const MatrixXi & X_;

};


/**
 * EigenClassificationCorpus wraps a pair of matrices X, y and implements the
 * Corpus interface with them using X as the words and y as the classes.
 */
class EigenClassificationCorpus : public ClassificationCorpus
{
    public:
        EigenClassificationCorpus(
            const MatrixXi &X,
            const VectorXi &y,
            int random_state = 0
        );

        size_t size() const override;
        virtual const std::shared_ptr<Document> at(size_t index) const override;
        void shuffle() override;
        float get_prior(int y) const override;

    private:
        /** To implement shuffle */
        CorpusIndexes indices_;

        // The data
        const MatrixXi & X_;
        const VectorXi & y_;

        // The class priors
        VectorXf priors_;
};


#endif  // _DOCUMENT_HPP_
