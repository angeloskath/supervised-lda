#ifndef _DOCUMENT_HPP_
#define _DOCUMENT_HPP_


#include <memory>
#include <random>
#include <vector>

#include <Eigen/Core>

namespace ldaplusplus {
namespace corpus {


// Forward declaration for the compiler
class Corpus;


/**
 * A Document is the minimal document needed for any type of LDA
 * implementation.
 *
 * It is meant to abstract away the source of the document data and the storage
 * type etc.
 */
class Document
{
    public:
        /**
         * @return The corpus this document belongs to
         */
        virtual const std::shared_ptr<const Corpus> get_corpus() const = 0;
        /**
         * @return The bag of words dense vector
         */
        virtual const Eigen::VectorXi & get_words() const = 0;

        /**
         * @return The corpus this documents belongs to after casting it to
         *         another pointer type for saving a few keystrokes.
         */
        template <typename T>
        const std::shared_ptr<const T> get_corpus() const {
            return std::static_pointer_cast<const T>(get_corpus());
        }

        virtual ~Document(){};
};


/**
 * The classification Document contains the words and a category in the form of
 * an integer.
 */
class ClassificationDocument : public Document
{
    public:
        /**
         * @return The class of the document
         */
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

        virtual ~Corpus(){};
};


/**
 * A corpus that contains information about the documents' classes (namely
 * their priors).
 */
class ClassificationCorpus : public Corpus
{
    public:
        /**
         * @param  y A class
         * @return The count of the documents in class y divided by the count
         *         of all the documents
         */
        virtual float get_prior(int y) const = 0;
};


/**
 * Eigen Document is a document backed by an Eigen::VectorXi.
 */
class EigenDocument : public Document
{
    public:
        EigenDocument(Eigen::VectorXi X) : EigenDocument(X, nullptr) {}
        EigenDocument(Eigen::VectorXi X, std::shared_ptr<const Corpus> corpus);

        const std::shared_ptr<const Corpus> get_corpus() const override;
        const Eigen::VectorXi & get_words() const override;

    private:
        Eigen::VectorXi X_;
        std::shared_ptr<const Corpus> corpus_;
};


/**
 * ClassificationDecorator decorates any Document with classification
 * information.
 */
class ClassificationDecorator : public ClassificationDocument
{
    public:
        /**
         * @param doc The document to be decorated
         * @param y   The class of the document
         */
        ClassificationDecorator(std::shared_ptr<Document> doc, int y);

        const std::shared_ptr<const Corpus> get_corpus() const override;
        const Eigen::VectorXi & get_words() const override;
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
        EigenCorpus(const Eigen::MatrixXi & X, int random_state=0);

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
        const Eigen::MatrixXi & X_;

};


/**
 * EigenClassificationCorpus wraps a pair of matrices X, y and implements the
 * Corpus interface with them using X as the words and y as the classes.
 */
class EigenClassificationCorpus : public ClassificationCorpus
{
    public:
        EigenClassificationCorpus(
            const Eigen::MatrixXi &X,
            const Eigen::VectorXi &y,
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
        const Eigen::MatrixXi & X_;
        const Eigen::VectorXi & y_;

        // The class priors
        Eigen::VectorXf priors_;
};

}  // namespace corpus
}  // namespace ldaplusplus

#endif  // _DOCUMENT_HPP_
