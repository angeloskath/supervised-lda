#ifndef _ONLINE_SUPERVISED_M_STEP_HPP_
#define _ONLINE_SUPERVISED_M_STEP_HPP_

#include "IMStep.hpp"

template <typename Scalar>
class OnlineSupervisedMStep : public IMStep<Scalar>
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;
    
    public:
        OnlineSupervisedMStep(
            VectorX class_weights,
            Scalar regularization_penalty = 1e-2,
            size_t minibatch_size = 128,
            Scalar eta_momentum = 0.9,
            Scalar eta_learning_rate = 0.01,
            Scalar beta_weight = 0.9
        );
        OnlineSupervisedMStep(
            size_t num_classes,
            Scalar regularization_penalty = 1e-2,
            size_t minibatch_size = 128,
            Scalar eta_momentum = 0.9,
            Scalar eta_learning_rate = 0.01,
            Scalar beta_weight = 0.9
        );

        /**
         * Maximize the ELBO.
         *
         * @param parameters       Model parameters, after being updated in m_step
         */
        virtual void m_step(
            std::shared_ptr<Parameters> parameters
        ) override;

        /**
         * This function calculates all necessary parameters, that
         * will be used for the maximazation step. And after seeing
         * `minibatch_size_` documents actually calls the m_step.
         *
         * @param doc              A single document
         * @param v_parameters     The variational parameters used in m-step
         *                         in order to maximize model parameters
         * @param m_parameters     Model parameters, used as output in case of 
         *                         online methods
         */
        virtual void doc_m_step(
            const std::shared_ptr<Document> doc,
            const std::shared_ptr<Parameters> v_parameters,
            std::shared_ptr<Parameters> m_parameters
        ) override;

    private:
        // Number of classes
        VectorX class_weights_;
        size_t num_classes_;

        // Minibatch size and portion (the portion of the corpus)
        size_t minibatch_size_;

        // The regularization penalty for the multinomial logistic regression
        // Mind that it should account for the minibatch size
        Scalar regularization_penalty_;

        // The suff stats and data needed to optimize the ELBO w.r.t. model
        // parameters
        MatrixX b_;
        Scalar beta_weight_;
        MatrixX expected_z_bar_;
        VectorXi y_;
        MatrixX eta_velocity_;
        MatrixX eta_gradient_;
        Scalar eta_momentum_;
        Scalar eta_learning_rate_;

        // The number of document's seen so far
        size_t docs_seen_so_far_;
};

#endif  // _ONLINE_SUPERVISED_M_STEP_HPP_
