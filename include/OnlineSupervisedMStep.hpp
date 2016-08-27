#ifndef _ONLINE_SUPERVISED_M_STEP_HPP_
#define _ONLINE_SUPERVISED_M_STEP_HPP_

#include "IMStep.hpp"

namespace ldaplusplus {


/**
 * OnlineSupervisedMStep is an online implementation of the classical
 * categorical supervised LDA.
 *
 * m_step() is called by doc_m_step() according to the minibatch_size
 * constructor parameter thus the model parameters are updated many times in an
 * EM step.
 *
 * Each m_step() updates the \f$\eta\f$ parameters using an SGD with momentum
 * update and the \f$\beta\f$ using the equation \f$\beta_{n+1} = w_{\beta}
 * \beta_{n} + (1-w_{\beta}) * MLE\f$.
 *
 * In the maximization with respect to \f$\eta\f$ the first order taylor
 * approximation to the expectation of the log normalizer is used as in the
 * SupervisedMStep.
 */
template <typename Scalar>
class OnlineSupervisedMStep : public IMStep<Scalar>
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;

    public:
        /**
         * Create an OnlineSupervisedMStep that accounts for class imbalance by
         * weighting the classes.
         *
         * @param class_weights          Weights to account for class
         *                               imbalance
         * @param regularization_penalty The L2 penalty for the logistic
         *                               regression
         * @param minibatch_size         After that many documents call
         *                               m_step()
         * @param eta_momentum           The momentum for the SGD update
         *                               of \f$\eta\f$
         * @param eta_learning_rate      The learning rate for the SGD
         *                               update of \f$\eta\f$
         * @param beta_weight            The weight for the online update
         *                                   of \f$\beta\f$
         */
        OnlineSupervisedMStep(
            VectorX class_weights,
            Scalar regularization_penalty = 1e-2,
            size_t minibatch_size = 128,
            Scalar eta_momentum = 0.9,
            Scalar eta_learning_rate = 0.01,
            Scalar beta_weight = 0.9
        );
        /**
         * Create an OnlineSupervisedMStep that uses uniform weights for the
         * classes.
         *
         * @param num_classes            The number of classes
         * @param regularization_penalty The L2 penalty for the logistic
         *                               regression
         * @param minibatch_size         After that many documents call
         *                               m_step()
         * @param eta_momentum           The momentum for the SGD update
         *                               of \f$\eta\f$
         * @param eta_learning_rate      The learning rate for the SGD
         *                               update of \f$\eta\f$
         * @param beta_weight            The weight for the online update
         *                                   of \f$\beta\f$
         */
        OnlineSupervisedMStep(
            size_t num_classes,
            Scalar regularization_penalty = 1e-2,
            size_t minibatch_size = 128,
            Scalar eta_momentum = 0.9,
            Scalar eta_learning_rate = 0.01,
            Scalar beta_weight = 0.9
        );

        /**
         * @inheritdoc
         */
        virtual void m_step(
            std::shared_ptr<Parameters> parameters
        ) override;

        /**
         * This function calculates all necessary parameters, that
         * will be used for the maximazation step. And after seeing
         * `minibatch_size` documents actually calls the m_step.
         *
         * @param doc          A single document
         * @param v_parameters The variational parameters used in m-step
         *                     in order to maximize model parameters
         * @param m_parameters Model parameters, used as output in case of 
         *                     online methods
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

}
#endif  // _ONLINE_SUPERVISED_M_STEP_HPP_
