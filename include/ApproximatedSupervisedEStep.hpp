#ifndef _APPROXIMATEDSUPERVISEDESTEP_HPP_
#define _APPROXIMATEDSUPERVISEDESTEP_HPP_

#include "UnsupervisedEStep.hpp"

template<typename Scalar>
class ApproximatedSupervisedEStep : public UnsupervisedEStep<Scalar>
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;

    public:
        enum CWeightType
        {
            Constant = 1,
            ExponentialDecay
        };

        ApproximatedSupervisedEStep(
            size_t e_step_iterations = 10,
            Scalar e_step_tolerance = 1e-2,
            Scalar C = 1,
            CWeightType weight_type = CWeightType::Constant,
            bool compute_likelihood = true
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
         * Store the epochs we 've seen so far.
         */
        void e_step() override;

    private:
        bool converged(const VectorX & gamma_old, const VectorX & gamma);

        /**
         * Get the weight for the supervised part of the e step.
         */
        Scalar get_weight();

        // The maximum number of iterations in E-step
        size_t e_step_iterations_;
        // The convergence tolerance for the maximazation of the ELBO w.r.t.
        // phi and gamma in E-step
        Scalar e_step_tolerance_;
        // A parameter weighting the supervised component in the variational
        // distribution.
        Scalar C_;
        // Compute the exact supervised likelihood to track convergence
        bool compute_likelihood_;
        // Determines how the parameter C updates in time
        CWeightType weight_type_;
        // The epochs seen so far
        int epochs_;
};
#endif   // _APPROXIMATEDSUPERVISEDESTEP_HPP_
