#ifndef _UNSUPERVISEDESTEP_HPP_
#define _UNSUPERVISEDESTEP_HPP_

#include <cmath>

#include "IEStep.hpp"


template <typename Scalar>
class UnsupervisedEStep : public IEStep<Scalar>
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;

    public:
        UnsupervisedEStep(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-4
        );

        /**
         * Maximize the ELBO w.r.t to phi and gamma
         *
         * @param doc          A sinle document
         * @param parameters   An instance of class Parameters, which
         *                     contains all necessary model parameters 
         *                     for e-step's implementation
         * @return             The variational parameters for the current
         *                     model, after e-step is completed
         */
        virtual std::shared_ptr<Parameters> doc_e_step(
            const std::shared_ptr<Document> doc,
            const std::shared_ptr<Parameters> parameters
        ) override;

        /**
         * Just do nothing for now.
         */
        virtual void e_step() override;

    protected:
        /**
         * The value of the ELBO.
         *
         * @param X       The word counts in column-major order for a single 
         *                document
         * @param alpha   The Dirichlet priors
         * @param beta    The over word topic distributiosn
         * @param eta     The classification parameters
         * @param phi     The Multinomial parameters
         * @param gamma   The Dirichlet parameters
         * @return        The likelihood
         */
        Scalar compute_likelihood(
            const VectorXi &X,
            const VectorX &alpha,
            const MatrixX &beta,
            const MatrixX &phi,
            const VectorX &gamma
        );

    private:
        // The maximum number of iterations in E-step
        size_t e_step_iterations_;
        // The convergence tolerance for the maximazation of the ELBO w.r.t.
        // phi and gamma in E-step
        Scalar e_step_tolerance_;
};
#endif //  _UNSUPERVISEDESTEP_HPP_

