#include "SupervisedMStep.hpp"
#include "MultinomialLogisticRegression.hpp"
#include "GradientDescent.hpp"

template <typename Scalar>
Scalar SupervisedMStep<Scalar>::m_step(
    const MatrixX &expected_z_bar,
    const MatrixX &b,
    const VectorXi &y,
    Ref<MatrixX> beta,
    Ref<MatrixX> eta
) {
    // Maximize w.r.t \beta during
    UnsupervisedMStep<Scalar>::m_step(
        expected_z_bar,
        b,
        y,
        beta,
        eta
    );

    // we need to maximize w.r.t to \eta
    Scalar initial_value = INFINITY;
    MultinomialLogisticRegression<Scalar> mlr(expected_z_bar, y, regularization_penalty_);
    GradientDescent<MultinomialLogisticRegression<Scalar>, MatrixX> minimizer(
        std::make_shared<ArmijoLineSearch<MultinomialLogisticRegression<Scalar>, MatrixX> >(),
        [this, &initial_value](
            Scalar value,
            Scalar gradNorm,
            size_t iterations
        ) {
            //this->get_progress_visitor()->visit(Progress<Scalar>{
            //    ProgressState::Maximization,
            //    value,
            //    iterations,
            //    0
            //});

            Scalar relative_improvement = (initial_value - value) / value;
            initial_value = value;

            return (
                iterations < m_step_iterations_ &&
                relative_improvement > m_step_tolerance_
            );
        }
    );
    minimizer.minimize(mlr, eta);

    return initial_value;

}
