#ifndef _FAST_UNSUPERVISED_E_STEP_HPP_
#define _FAST_UNSUPERVISED_E_STEP_HPP_


#include <Eigen/Core>

#include "ldaplusplus/IEStep.hpp"

namespace ldaplusplus {


/**
 * FastUnsupervisedEStep doesn't compute the log likelihood but checks for
 * convergence based on the change of the variational parameters and thus
 * avoids a lot of time consuming computations.
 *
 * For the mathematics see UnsupervisedEStep.
 */
template <typename Scalar>
class FastUnsupervisedEStep : public IEStep<Scalar>
{
    typedef Eigen::Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Eigen::Matrix<Scalar, Dynamic, 1> VectorX;

    public:
        /**
         * @param e_step_iterations The max number of times to alternate
         *                          between maximizing for \f$\gamma\f$ and for
         *                          \f$\phi\f$.
         * @param e_step_tolerance  The minimum relative change in the
         *                          variational parameter \f$\gamma\f$.
         */
        FastUnsupervisedEStep(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-2
        );

        /**
         * Maximize the ELBO w.r.t. to \f$\phi\f$ and \f$\gamma\f$.
         *
         * @param doc        A single document.
         * @param parameters An instance of class Parameters, which
         *                   contains all necessary model parameters 
         *                   for expecation step implementation.
         * @return           The variational parameters for the current
         *                   model, after expecation step is completed.
         */
        std::shared_ptr<Parameters> doc_e_step(
            const std::shared_ptr<Document> doc,
            const std::shared_ptr<Parameters> parameters
        ) override;

        void e_step() override;

    private:
        /**
         * Check for convergence based on the change of the variational
         * parameter \f$\gamma\f$.
         *
         * @param gamma_old The gamma of the previous iteration.
         * @param gamma     The gamma of this iteration.
         * @return Whether the change is small enough to indicate convergence.
         */
        bool converged(const VectorX & gamma_old, const VectorX & gamma);

        size_t e_step_iterations_;
        Scalar e_step_tolerance_;
};

}
#endif  // _FAST_UNSUPERVISED_E_STEP_HPP_
