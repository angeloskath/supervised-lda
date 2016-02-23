#ifndef _SUPERVISED_LDA_HPP_
#define _SUPERVISED_LDA_HPP_


#include <cstddef>

#include <Eigen/Core>


using namespace Eigen;  // Should we be using the namespace?


/**
 * SupervisedLDA computes supervised topic models.
 */
template <typename Scalar>
class SupervisedLDA
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;

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
        void fit(const MatrixXi &X, const VectorXi &y);

        /**
         * Perform a single em iteration.
         *
         * @param X The word counts in column-major order
         * @param y The classes as integers
         */
        void partial_fit(const MatrixXi &X, const VectorXi &y);

        /**
         * Maximize the ELBO w.r.t phi and gamma
         *
         * We use the following update functions until convergence.
         *
         * \phi_n \prop \beta_{w_n} exp(
         *      \Psi(\gamma) +
         *      \frac{1}{N} \eta_y^T +
         *      \frac{h}{h^T \phi_n^{old}}
         *  )
         * \gamma = \alpha + \sum_{n=1}^N \phi_n
         */
        Scalar doc_e_step(
            const VectorXi &X,
            int y,
            const VectorX &alpha,
            const MatrixX &beta,
            const MatrixX &eta,
            MatrixX &phi,
            VectorX &gamma,
            int fixed_point_iterations,
            int max_iter,
            Scalar convergence_tolerance
        );

        /**
         * The value of the ELBO.
         */
        Scalar compute_likelihood(
            const VectorXi &X,
            int y,
            const VectorX &alpha,
            const MatrixX &beta,
            const MatrixX &eta,
            const MatrixX &phi,
            const VectorX &gamma,
            const MatrixX &h
        );

        /**
         * h \in \mathbb{R}^{K \times V}
         *
         * h_{n} = \sum_{y \in Y} \left(
         *      \prod_{l=1, l \neq n}^V \phi_l^T \left( exp(\frac{X_l}{\sum X} \eta^T y) \right)
         *  \right) exp(\frac{X_n}{\sum X} \eta^T y)
         */
        void compute_h(
            const VectorXi &X,
            const MatrixX &eta,
            const MatrixX &phi,
            MatrixX &h
        );

    private:
        size_t K_;
        size_t max_iter_;
};


#endif  //_SUPERVISED_LDA_HPP_
