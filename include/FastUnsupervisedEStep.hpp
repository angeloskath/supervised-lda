#ifndef _FAST_UNSUPERVISED_E_STEP_HPP_
#define _FAST_UNSUPERVISED_E_STEP_HPP_


#include <Eigen/Core>

#include "IEStep.hpp"


/**
 * FastSupervisedEStep doesn't compute the log likelihood but checks for
 * convergence based on the change of the variational parameters and thus
 * avoids a lot of time consuming computations.
 */
template <typename Scalar>
class FastUnsupervisedEStep : public IEStep<Scalar>
{
    typedef Eigen::Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Eigen::Matrix<Scalar, Dynamic, 1> VectorX;

    public:
        FastUnsupervisedEStep(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-2
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
        std::shared_ptr<Parameters> doc_e_step(
            const std::shared_ptr<Document> doc,
            const std::shared_ptr<Parameters> parameters
        ) override;

    private:
        bool converged(const VectorX & gamma_old, const VectorX & gamma);

        size_t e_step_iterations_;
        Scalar e_step_tolerance_;
};

#endif  // _FAST_UNSUPERVISED_E_STEP_HPP_
