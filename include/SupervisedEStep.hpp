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
            size_t fixed_point_iterations = 20,
            Scalar e_step_tolerance = 1e-4
        );

        Scalar doc_e_step(
            const VectorXi &X,
            int y,
            const VectorX &alpha,
            const MatrixX &beta,
            const MatrixX &eta,
            Ref<MatrixX> phi,
            Ref<VectorX> gamma
        ) override;
    
    protected:
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
#endif //  _SUPERVISEDESTEP_HPP_
