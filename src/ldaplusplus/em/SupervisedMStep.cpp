#include "ldaplusplus/optimization/GradientDescent.hpp"
#include "ldaplusplus/optimization/MultinomialLogisticRegression.hpp"
#include "ldaplusplus/events/ProgressEvents.hpp"
#include "ldaplusplus/em/SupervisedMStep.hpp"

namespace ldaplusplus {

using em::SupervisedMStep;
using optimization::ArmijoLineSearch;
using optimization::GradientDescent;
using optimization::MultinomialLogisticRegression;


template <typename Scalar>
void SupervisedMStep<Scalar>::doc_m_step(
    const std::shared_ptr<corpus::Document> doc,
    const std::shared_ptr<parameters::Parameters> v_parameters,
    std::shared_ptr<parameters::Parameters> m_parameters
) {
    UnsupervisedMStep<Scalar>::doc_m_step(
        doc,
        v_parameters,
        m_parameters
    );
    // Cast Parameters to VariationalParameters in order to have access to gamma
    const VectorX &gamma = std::static_pointer_cast<parameters::VariationalParameters<Scalar> >(v_parameters)->gamma;
    // Cast Parameters to SupervisedModelParameters in order to have access to alpha
    const VectorX &alpha = std::static_pointer_cast<parameters::SupervisedModelParameters<Scalar> >(m_parameters)->alpha;
    int num_topics = alpha.rows();

    // Resize properly expected_z_bar_ and y_ every time the current function
    // is being called. In case of y_ simple add one at the end. In case of
    // expected_z_bar add an extra column of size topicsx1
    if (docs_ >= expected_z_bar_.cols()) {
        // Add an extra row in y_
        y_.conservativeResize(docs_+1);
        
        // Add an extra column in expected_z_bar_
        expected_z_bar_.conservativeResize(num_topics, docs_+1);
    }

    y_(docs_) = std::static_pointer_cast<corpus::ClassificationDocument>(doc)->get_class();

    expected_z_bar_.col(docs_) = gamma - alpha;
    // TODO: Maybe move the following normalization to m_step() and call
    //       math_utils::normalize_cols()
    auto words_in_doc = expected_z_bar_.col(docs_).sum();
    if (words_in_doc != 0) {
        expected_z_bar_.col(docs_).array() /= words_in_doc;
    }

    docs_ += 1;
}

template <typename Scalar>
void SupervisedMStep<Scalar>::m_step(
    std::shared_ptr<parameters::Parameters> parameters
) {
    // Maximize w.r.t \beta during
    UnsupervisedMStep<Scalar>::m_step(
        parameters
    );
    MatrixX &eta = std::static_pointer_cast<parameters::SupervisedModelParameters<Scalar> >(parameters)->eta;

    // resize the member variables to fit the documents we 've seen so far in
    // the doc_m_steps
    y_.conservativeResize(docs_);
    expected_z_bar_.conservativeResize(expected_z_bar_.rows(), docs_);
    docs_ = 0;

    // we need to maximize w.r.t to \eta
    Scalar initial_value = INFINITY;
    MultinomialLogisticRegression<Scalar> mlr(expected_z_bar_, y_, regularization_penalty_);
    GradientDescent<MultinomialLogisticRegression<Scalar>, MatrixX> minimizer(
        std::make_shared<ArmijoLineSearch<MultinomialLogisticRegression<Scalar>, MatrixX> >(),
        [this, &initial_value](
            Scalar value,
            Scalar gradNorm,
            size_t iterations
        ) {
            this->get_event_dispatcher()->template dispatch<events::MaximizationProgressEvent<Scalar> >(
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
}

// Template instantiation
template class SupervisedMStep<float>;
template class SupervisedMStep<double>;


}
