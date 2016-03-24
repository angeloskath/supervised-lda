#ifndef _E_STEP_UTILS_HPP_
#define _E_STEP_UTILS_HPP_

#include <Eigen/Core>


namespace e_step_utils
{
    using namespace Eigen;

    template <typename Scalar>
    using MatrixX = Matrix<Scalar, Dynamic, Dynamic>;
    template <typename Scalar>
    using VectorX = Matrix<Scalar, Dynamic, 1>;

    /**
     * Compute the value of the ELBO (using the unsupervised definition of the
     * model) for a given document, model parameters and variational
     * parameters.
     */
    template <typename Scalar>
    Scalar compute_unsupervised_likelihood(
        const VectorXi & X,
        const VectorX<Scalar> &alpha,
        const MatrixX<Scalar> &beta,
        const MatrixX<Scalar> &phi,
        const VectorX<Scalar> &gamma
    );

    /**
     * Compute the value of the ELBO (using the supervised definition of the
     * model) for a given document, model parameters and variational
     * parameters.
     *
     * @param X       The word counts in column-major order for a single 
     *                document
     * @param y       The class label as integer for the current document
     * @param alpha   The Dirichlet priors
     * @param beta    The over word topic distributiosn
     * @param eta     The classification parameters
     * @param phi     The Multinomial parameters
     * @param gamma   The Dirichlet parameters
     * @return        The log likelihood
     */
    template <typename Scalar>
    Scalar compute_supervised_likelihood(
        const VectorXi & X,
        int y,
        const VectorX<Scalar> &alpha,
        const MatrixX<Scalar> &beta,
        const MatrixX<Scalar> &eta,
        const MatrixX<Scalar> &phi,
        const VectorX<Scalar> &gamma
    );
    template <typename Scalar>
    Scalar compute_supervised_likelihood(
        const VectorXi & X,
        int y,
        const VectorX<Scalar> &alpha,
        const MatrixX<Scalar> &beta,
        const MatrixX<Scalar> &eta,
        const MatrixX<Scalar> &phi,
        const VectorX<Scalar> &gamma,
        const MatrixX<Scalar> &h
    );

    /**
     * Compute the Matrix h as defined in [Wang, C., Blei, D. and Li, F.F.,
     * 2009, June. Simultaneous image classification and annotation. In
     * Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE
     * Conference on (pp. 1903-1910). IEEE.]
     *
     * h \in \mathbb{R}^{K \times V}
     *
     * h_{n} = \sum_{y \in Y} \left(
     *      \prod_{l=1, l \neq n}^V \phi_l^T \left( exp(\frac{X_l}{\sum X} \eta^T y) \right)
     *  \right) exp(\frac{X_n}{\sum X} \eta^T y)
     *
     * @param X       The word counts in column-major order for a single 
     *                document
     * @param eta     The classification parameters
     * @param phi     The Multinomial parameters
     * @param h       The output value
     */
    template <typename Scalar>
    void compute_h(
        const VectorXi & X,
        const VectorX<Scalar> & X_ratio,
        const MatrixX<Scalar> &eta,
        const MatrixX<Scalar> &phi,
        Ref<MatrixX<Scalar> > h
    );

    /**
     * Perform a fixed point iteration update for the variational parameters
     * phi of the supervised lda model.
     */
    template <typename Scalar>
    void fixed_point_iteration(
        const VectorX<Scalar> & X_ratio,
        int y,
        const MatrixX<Scalar> & beta,
        const MatrixX<Scalar> & eta,
        const VectorX<Scalar> & gamma,
        const MatrixX<Scalar> &h,
        Ref<MatrixX<Scalar> > phi_old,
        Ref<MatrixX<Scalar> > phi
    );

    /**
     * Compute the gamma based on the prior and the variational parameter phi.
     *
     * This operation is the same in both supervised and unsupervised LDA and
     * it can be summarised to:
     *
     *     gamma_i ^ {t+1} =  alpha_i + \sum_n \phi_{n,i}^{t+1}
     *
     * Equation (7) in Latent Dirichlet Allocation, Blei 2003 
     */
    template <typename Scalar>
    void compute_gamma(
        const VectorXi & X,
        const VectorX<Scalar> & alpha,
        const MatrixX<Scalar> & phi,
        Ref<VectorX<Scalar> > gamma
    );

    /**
     * Update Multinomial parameter phi, according to the following
     * pseudocode
     *
     *     for n=1 to Nd do
     *      for i=1 to K do
     *          phi_{n,i}^{t+1} = beta_{i, w_n}exp(\psi(\gamma_i) - \psi(sum_i \gamma_i))
     *      end
     *      normalize phi_{n,i}^{t+1} sum to 1
     *     end
     * 
     * Equation (6) in Latent Dirichlet Allocation, Blei 2003
     */
    template <typename Scalar>
    void compute_unsupervised_phi(
        const MatrixX<Scalar> & beta,
        const VectorX<Scalar> & gamma,
        Ref<MatrixX<Scalar> > phi
    );

    /**
     * Update Multinomial parameter phi, according to the following approximation
     *
     *
     */
    template <typename Scalar>
    void compute_supervised_approximate_phi(
        const VectorX<Scalar> & X_ratio,
        int y,
        const MatrixX<Scalar> & beta,
        const MatrixX<Scalar> & eta,
        const VectorX<Scalar> & gamma,
        Ref<MatrixX<Scalar> > phi
    );
}

#endif  // _E_STEP_UTILS_HPP_
