#ifndef _FAST_SUPERVISED_E_STEP_HPP_
#define _FAST_SUPERVISED_E_STEP_HPP_


#include <Eigen/Core>

#include "ldaplusplus/em/IEStep.hpp"

namespace ldaplusplus {
namespace em {


/**
 * FastSupervisedEStep doesn't compute the log likelihood but checks for
 * convergence based on the change of the variational parameters.
 *
 * In the traditional categorical supervised expectation step the likelihood
 * estimation is less resource intensive (compared to the rest of the e step)
 * and thus FastSupervisedEStep is **not** recommended for use.
 *
 * For the specific mathematics used in the maximization see SupervisedEStep.
 */
template <typename Scalar>
class FastSupervisedEStep : public IEStep<Scalar>
{
    typedef Eigen::Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Eigen::Matrix<Scalar, Dynamic, 1> VectorX;

    public:
        /**
         * @param e_step_iterations      The max number of times to alternate
         *                               between maximizing for \f$\gamma\f$
         *                               and for \f$\phi\f$.
         * @param e_step_tolerance       The minimum relative change in the
         *                               variational parameter \f$\gamma\f$.
         * @param fixed_point_iterations The number of fixed point iterations
         *                               used in the maximization for
         *                               \f$\phi\f$.
         */
        FastSupervisedEStep(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-2,
            size_t fixed_point_iterations = 20
        );

        /**
         * See SupervisedEStep for the specific equations that are being
         * maximized.
         *
         * @param doc        A single document.
         * @param parameters An instance of class Parameters, which
         *                   contains all necessary model parameters 
         *                   for expectation step implementation.
         * @return           The variational parameters for the current
         *                   model, after expectation step is completed.
         */
        std::shared_ptr<parameters::Parameters> doc_e_step(
            const std::shared_ptr<corpus::Document> doc,
            const std::shared_ptr<parameters::Parameters> parameters
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

        // The maximum number of iterations in E-step.
        size_t e_step_iterations_;
        // The convergence tolerance for the maximazation of the ELBO w.r.t.
        // phi and gamma in E-step.
        Scalar e_step_tolerance_;
        // The maximum number of iterations while maximizing phi in E-step.
        size_t fixed_point_iterations_;
};

}  // namespace em
}  // namespace ldaplusplus

#endif  // _FAST_SUPERVISED_E_STEP_HPP_
