#ifndef _DOCUMENT_HPP_
#define _DOCUMENT_HPP_


#include <memory>
#include <random>
#include <vector>

#include <Eigen/Core>

using namespace Eigen;


/**
 * A Document is the minimal document needed for any type of LDA
 * implementation.
 */
class Document
{
    public:
        virtual const VectorXi & get_words() const = 0;
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
        virtual const std::shared_ptr<Document> operator[] (size_t index) const = 0;
        /**
         * Shuffle the documents so that the ith document is any of the
         * documents with probability 1.0/size() .
         */
        virtual void shuffle() = 0;
};


/**
 * Eigen Document is a document backed by an Eigen::VectorXi.
 */
class EigenDocument : public Document
{
    public:
        EigenDocument(const VectorXi &X);

        const VectorXi & get_words() const override;

    private:
        const VectorXi & X_;
};


/**
 * ClassificationDecorator decorates any Document with classification
 * information.
 */
class ClassificationDecorator : public ClassificationDocument
{
    public:
        ClassificationDecorator(std::shared_ptr<Document> doc, int y);

        const VectorXi & get_words() const override;
        int get_class() const override;

    private:
        std::shared_ptr<Document> document_;
        int y_;
};


/**
 * EigenClassificationCorpus wraps a pair of matrices X, y and implements the
 * Corpus interface with them using X as the words and y as the classes.
 */
class EigenClassificationCorpus : public Corpus
{
    public:
        EigenClassificationCorpus(
            const MatrixXi &X,
            const VectorXi &y,
            int random_state = 0
        );

        size_t size() const override;
        const std::shared_ptr<Document> operator[] (size_t index) const override;
        void shuffle() override;

    private:
        // Keep X and y internally as constant references
        // TODO: Evaluate if these should be copied in or even moved in
        const MatrixXi & X_;
        const VectorXi & y_;

        // A vector of indices so that we can be iterating randomly without
        // shuffling the large X_ in memory. It may suck for cache friendliness
        // but this is the least of our optimization problems.
        std::vector<int> indices_;

        // A pseudo random number generator for the shuffling
        std::mt19937 prng_;
};


#endif  // _DOCUMENT_HPP_
