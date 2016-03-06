#include "GradientDescent.hpp"
#include "MultinomialLogisticRegression.hpp"
#include "ProgressEvents.hpp"
#include "SupervisedMStep.hpp"

template <typename Scalar>
void SupervisedMStep<Scalar>::m_step(
    std::shared_ptr<Parameters> model_parameters
) {
    // Maximize w.r.t \beta during
    UnsupervisedMStep<Scalar>::m_step(
        std::shared_ptr<Parameters> model_parameters
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
            this->get_event_dispatcher()->template dispatch<MaximizationProgressEvent<Scalar> >(
                iterations,
                -value  // minus the value to be minimized is the log likelihood
            );

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

// Template instantiation
template class SupervisedMStep<float>;
template class SupervisedMStep<double>;

