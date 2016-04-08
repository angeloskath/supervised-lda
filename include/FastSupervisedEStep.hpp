#ifndef _FAST_SUPERVISED_E_STEP_HPP_
#define _FAST_SUPERVISED_E_STEP_HPP_


#include <Eigen/Core>

#include "IEStep.hpp"


/**
 * FastSupervisedEStep doesn't compute the log likelihood but checks for
 * convergence based on the change of the variational parameters and thus
 * avoids a lot of time consuming computations.
 */
template <typename Scalar>
class FastSupervisedEStep : public IEStep<Scalar>
{
    typedef Eigen::Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Eigen::Matrix<Scalar, Dynamic, 1> VectorX;

    public:
        FastSupervisedEStep(
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

        void e_step() override;

    private:
        bool converged(const VectorX & gamma_old, const VectorX & gamma);

        size_t e_step_iterations_;
        Scalar e_step_tolerance_;
        size_t fixed_point_iterations_;
};


#endif  // _FAST_SUPERVISED_E_STEP_HPP_
