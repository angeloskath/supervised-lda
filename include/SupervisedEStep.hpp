#ifndef _SUPERVISEDESTEP_HPP_
#define _SUPERVISEDESTEP_HPP_

#include "UnsupervisedEStep.hpp"

template <typename Scalar>
class SupervisedEStep : public UnsupervisedEStep<Scalar>
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;

    public:
        SupervisedEStep(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-2,
            size_t fixed_point_iterations = 20
        );

        /** Maximize the ELBO w.r.t phi and gamma
         *
         * We use the following update functions until convergence.
         *
         * \phi_n \prop \beta_{w_n} exp(
         *      \Psi(\gamma) +
         *      \frac{1}{N} \eta_y^T +
         *      \frac{h}{h^T \phi_n^{old}}
         *  )
         * \gamma = \alpha + \sum_{n=1}^N \phi_n
         *
         * @param doc          A sinle document
         * @param parameters   An instance of class Parameters, which
         *                     contains all necessary model parameters 
         *                     for e-step's implementation
         * @return             The variational parameters for the current
         *                     model, after e-step is completed
         */
        std::shared_ptr<Parameters> doc_e_step(
            const std::shared_ptr<Document> doc,
            const std::shared_ptr<Parameters> parameters
        ) override;

        /**
         * The value of the ELBO.
         *
         * @param X       The word counts in column-major order for a single 
         *                document
         * @param y       The class label as integer for the current document
         * @param alpha   The Dirichlet priors
         * @param beta    The over word topic distributiosn
         * @param eta     The classification parameters
         * @param phi     The Multinomial parameters
         * @param gamma   The Dirichlet parameters
         * @param h       The output of the equation implemented in function
         *                compute_h
         * @return        The likelihood
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
         *
         * @param X       The word counts in column-major order for a single 
         *                document
         * @param eta     The classification parameters
         * @param phi     The Multinomial parameters
         * @param h       The output value
         */
        void compute_h(
            const VectorXi &X,
            const MatrixX &eta,
            const MatrixX &phi,
            Ref<MatrixX> h
        );

    private:
        // The maximum number of iterations in E-step
        size_t e_step_iterations_;
        // The maximum number of iterations while maximizing phi in E-step
        size_t fixed_point_iterations_;
        // The convergence tolerance for the maximazation of the ELBO w.r.t.
        // phi and gamma in E-step
        Scalar e_step_tolerance_;
};
#endif  // _SUPERVISEDESTEP_HPP_

