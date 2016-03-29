#ifndef _MULTINOMIALSUPERVISEDESTEP_HPP_
#define _MULTINOMIALSUPERVISEDESTEP_HPP__

#include "UnsupervisedEStep.hpp"

template<typename Scalar>
class MultinomialSupervisedEStep: public UnsupervisedEStep<Scalar>
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;

    public:
        MultinomialSupervisedEStep(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-2,
            int mu = 2,
            Scalar eta_weight = 1
        );

        /** Maximize the ELBO w.r.t phi and gamma
         *
         * We use the following update functions until convergence.
         *
         * \phi_n \prop \beta_{w_n} eta_y expi(\Psi(\gamma))
         *
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

    private:
        bool converged(const VectorX & gamma_old, const VectorX & gamma);

        // The maximum number of iterations in E-step
        size_t e_step_iterations_;
        // The convergence tolerance for the maximazation of the ELBO w.r.t.
        // phi and gamma in E-step
        Scalar e_step_tolerance_;
        int mu_;
        Scalar eta_weight_;
};

#endif   //  _MULTINOMIALSUPERVISEDESTEP_HPP_
