#ifndef _SUPERVISED_LDA_HPP_
#define _SUPERVISED_LDA_HPP_


#include <cstddef>
#include <memory>
#include <vector>

#include <Eigen/Core>

#include "ProgressVisitor.hpp"


using namespace Eigen;  // Should we be using the namespace?


/**
 * SupervisedLDA computes supervised topic models.
 */
template <typename Scalar>
class SupervisedLDA
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;

    public:
        struct LDAState
        {
            std::vector<const VectorX *> vectors;
            std::vector<const MatrixX *> matrices;
        };

        /**
         * @param topics                 The number of topics for this model
         * @param iterations             The maximum number of EM iterations
         * @param e_step_tolerance       The convergence tolerance for the maximazation of
         *                               ELBO w.r.t. phi and gamma in E-step
         * @param m_step_tolerance       The convergence tolerance for the maximazation of
         *                               ELBO w.r.t. eta in M-step
         * @param e_step_iterations      The maximum number of E-step iterations
         * @param m_step_iterations      The maximum number of M-step iterations
         * @param fixed_point_iterations The maximum number of iterations while maximizing
         *                               phi in E-step
         */
        SupervisedLDA(
            size_t topics,
            size_t iterations = 20,
            Scalar e_step_tolerance = 1e-4,
            Scalar m_step_tolerance = 1e-4,
            size_t e_step_iterations = 10,
            size_t m_step_iterations = 20,
            size_t fixed_point_iterations = 20,
            Scalar regularization_penalty = 0.05
        );

        /**
         * @param lda_state All the information required to rebuild the
         *                  SupervisedLDA model as it was
         */
        SupervisedLDA(LDAState lda_state);

        /**
         * Compute a supervised topic model for word counts X and classes y.
         *
         * Perform as many em iterations as configured and stop when reaching
         * max_iter_ or any other stopping criterion.
         *
         * @param X The word counts in column-major order
         * @param y The classes as integers
         */
        void fit(const MatrixXi &X, const VectorXi &y);

        /**
         * Perform a single em iteration.
         *
         * @param X The word counts in column-major order
         * @param y The classes as integers
         */
        void partial_fit(const MatrixXi &X, const VectorXi &y);

        /**
         * Transform the word counts X into topic proportions and return a new
         * matrix.
         */
        MatrixX transform(const MatrixXi &X);

        /**
         * Return the class scores according to the model parameters.
         */
        MatrixX decision_function(const MatrixXi &X);

        /**
         * Use the model to predict the class indexes for the word counts X.
         */
        VectorXi predict(const MatrixXi &X);

        /**
         * Maximize the ELBO w.r.t phi and gamma
         *
         * We use the following update functions until convergence.
         *
         * \phi_n \prop \beta_{w_n} exp(
         *      \Psi(\gamma) +
         *      \frac{1}{N} \eta_y^T +
         *      \frac{h}{h^T \phi_n^{old}}
         *  )
         * \gamma = \alpha + \sum_{n=1}^N \phi_n
         */
        Scalar doc_e_step(
            const VectorXi &X,
            int y,
            const VectorX &alpha,
            const MatrixX &beta,
            const MatrixX &eta,
            Ref<MatrixX> phi,
            Ref<VectorX> gamma,
            size_t fixed_point_iterations,
            size_t e_step_iterations,
            Scalar e_step_tolerance
        );

        /**
         * This function calculates all necessary parameters, that
         * will be used for the maximazation step.
         */
        void doc_m_step(
           const VectorXi &X,
           const MatrixX &phi,
           Ref<MatrixX> b,
           Ref<VectorX> expected_z_bar
        );

        /**
         * Maximize the ELBO w.r.t \beta and \eta.
         *
         * @param expected_Z_bar Is the expected values of Z_bar for every
         *                       document
         * @param b              The unnormalized new betas
         * @param y              The class indexes for every document
         * @param beta           The topic word distributions
         * @param eta            The classification parameters
         * @return               The likelihood of the Multinomial logistic
         *                       regression
         */
        Scalar m_step(
            const MatrixX &expected_z_bar,
            const MatrixX &b,
            const VectorXi &y,
            Ref<MatrixX> beta,
            Ref<MatrixX> eta,
            Scalar L,
            Scalar m_step_iterations,
            Scalar m_step_tolerance
        );

        /**
         * The value of the ELBO.
         */
        Scalar compute_likelihood(
            const VectorXi &X,
            int y,
            const VectorX &alpha,
            const MatrixX &beta,
            const MatrixX &eta,
            const MatrixX &phi,
            const VectorX &gamma,
            const MatrixX &h
        );

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

        /**
         * Set the progress visitor for this lda instance.
         */
        void set_progress_visitor(std::shared_ptr<IProgressVisitor<Scalar> > visitor) {
            visitor_ = visitor;
        }

        /**
         * Get the progress visitor for this lda instance.
         */
        std::shared_ptr<IProgressVisitor<Scalar> > get_progress_visitor() {
            if (visitor_) {
                return visitor_;
            } else {
                // return a noop visitor
                return std::make_shared<FunctionVisitor<Scalar> >(
                    [](Progress<Scalar> p){}
                );
            }
        }

        /**
         * Return an LDAState object containing all that is needed to create
         * another similar lda object.
         */
        LDAState get_state() {
            return {
                {&parameters_, &alpha_},
                {&beta_, &eta_}
            };
        }

    protected:
        /**
         * This function is used for the inisialization of model parameters, namely
         * eta, beta, alpha
         */
        void initialize_model_parameters(
            const MatrixXi &X,
            const VectorXi &y,
            Ref<VectorX> alpha,
            Ref<MatrixX> beta,
            Ref<MatrixX> eta,
            size_t topics
        );

    private:
        // Obviously the total number of topics
        size_t topics_;
        // The maximum number of EM iterations
        size_t iterations_;
        // The convergence tolerance for the maximazation of the ELBO w.r.t.
        // phi and gamma in E-step
        Scalar e_step_tolerance_;
        // The convergence tolerance for the maximazation of the ELBO w.r.t.
        // eta in M-step
        Scalar m_step_tolerance_;
        // The maximum number of iterations in E-step
        size_t e_step_iterations_;
        // The maximum number of iterations in M-step
        size_t m_step_iterations_;
        // The maximum number of iterations while maximizing phi in E-step
        size_t fixed_point_iterations_;
        // The regularization penalty for the multinomial logistic regression
        Scalar regularization_penalty_;

        // The parameters of the model
        VectorX alpha_;
        MatrixX beta_;
        MatrixX eta_;

        // A visitor whom we inform in case of progress etc
        std::shared_ptr<IProgressVisitor<Scalar> > visitor_;

        // A vector containing all the parameters needed to recreate the
        // model's behaviour (tolerances, regularization penalty etc)  which
        // can also be serialized and added to LDAState
        VectorX parameters_;
};


#endif  //_SUPERVISED_LDA_HPP_
