#ifndef _SUPERVISED_LDA_HPP_
#define _SUPERVISED_LDA_HPP_


#include <cstddef>

#include <eigen3/Eigen/Core>


using namespace Eigen;  // Should we be using the namespace?


/**
 * SupervisedLDA computes supervised topic models.
 */
template <typename Scalar>
class SupervisedLDA
{
    public:
        SupervisedLDA(size_t K) : SupervisedLDA(K, 20) {}
        /**
         * @param K        The number of topics for this model
         * @param max_iter The maximum number of iterations to perform
         */
        SupervisedLDA(size_t K, size_t max_iter) : K_(K),
                                                   max_iter_(max_iter)
        {}

        /**
         * Compute a supervised topic model for word counts X and classes y.
         *
         * Perform as many em iterations as configured and stop when reaching
         * max_iter_ or any other stopping criterion.
         *
         * @param X The word counts in column-major order
         * @param y The classes as integers
         */
        void fit(MatrixXi X, VectorXi y);

        /**
         * Perform a single em iteration.
         *
         * @param X The word counts in column-major order
         * @param y The classes as integers
         */
        void partial_fit(MatrixXi X, VectorXi y);

    private:
        size_t K_;
        size_t max_iter_;
};


#endif  //_SUPERVISED_LDA_HPP_
