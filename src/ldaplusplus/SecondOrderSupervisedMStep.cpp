#include "ldaplusplus/GradientDescent.hpp"
#include "ldaplusplus/ProgressEvents.hpp"
#include "ldaplusplus/SecondOrderLogisticRegressionApproximation.hpp"
#include "ldaplusplus/SecondOrderSupervisedMStep.hpp"

namespace ldaplusplus {


template <typename Scalar>
void SecondOrderSupervisedMStep<Scalar>::doc_m_step(
    const std::shared_ptr<Document> doc,
    const std::shared_ptr<Parameters> v_parameters,
    std::shared_ptr<Parameters> m_parameters
) {
    UnsupervisedMStep<Scalar>::doc_m_step(
        doc,
        v_parameters,
        m_parameters
    );

    // Get the words from the doc
    const VectorXi & X = doc->get_words();
    int N = X.sum();

    // Cast Parameters to VariationalParameters in order to have access to gamma and phi
    const VectorX &gamma = std::static_pointer_cast<VariationalParameters<Scalar> >(v_parameters)->gamma;
    const MatrixX &phi = std::static_pointer_cast<VariationalParameters<Scalar> >(v_parameters)->phi;
    // Cast Parameters to SupervisedModelParameters in order to have access to alpha
    const VectorX &alpha = std::static_pointer_cast<SupervisedModelParameters<Scalar> >(m_parameters)->alpha;
    int num_topics = alpha.rows();

    // Resize properly expected_z_bar_ and y_ every time the current function
    // is being called. In case of y_ simple add one at the end. In case of
    // expected_z_bar add an extra column of size topicsx1
    if (docs_ >= expected_z_bar_.cols()) {
        // Add an extra row in y_
        y_.conservativeResize(docs_+1);
        
        // Add an extra column in expected_z_bar_
        expected_z_bar_.conservativeResize(num_topics, docs_+1);

        // Add an extra matrix in variance_z_bar_
        variance_z_bar_.resize(docs_+1);
    }

    // get the class
    y_(docs_) = std::static_pointer_cast<ClassificationDocument>(doc)->get_class();

    // get the expected_z_bar
    expected_z_bar_.col(docs_) = gamma - alpha;
    expected_z_bar_.col(docs_).array() /= N;

    // get the variance_z_bar
    MatrixX phi_scaled = phi.array().rowwise() * X.cast<Scalar>().array().transpose();
    variance_z_bar_[docs_] = phi_scaled * phi_scaled.transpose();
    variance_z_bar_[docs_] *= 1.0/(N * N);

    docs_ += 1;
}


template <typename Scalar>
void SecondOrderSupervisedMStep<Scalar>::m_step(
    std::shared_ptr<Parameters> parameters
) {
    // Maximize w.r.t \beta during
    UnsupervisedMStep<Scalar>::m_step(
        parameters
    );
    MatrixX &eta = std::static_pointer_cast<SupervisedModelParameters<Scalar> >(parameters)->eta;

    // resize the member variables to fit the documents we 've seen so far in
    // the doc_m_steps
    y_.conservativeResize(docs_);
    expected_z_bar_.conservativeResize(expected_z_bar_.rows(), docs_);
    variance_z_bar_.resize(docs_);
    docs_ = 0;

    // we need to maximize w.r.t to \eta
    Scalar initial_value = INFINITY;
    SecondOrderLogisticRegressionApproximation<Scalar> mlr(
        expected_z_bar_,
        variance_z_bar_,
        y_,
        regularization_penalty_
    );
    GradientDescent<SecondOrderLogisticRegressionApproximation<Scalar>, MatrixX> minimizer(
        std::make_shared<ArmijoLineSearch<SecondOrderLogisticRegressionApproximation<Scalar>, MatrixX> >(),
        [this, &initial_value](
            Scalar value,
            Scalar gradNorm,
            size_t iterations
        ) {
            this->get_event_dispatcher()->template dispatch<MaximizationProgressEvent<Scalar> >(
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
template class SecondOrderSupervisedMStep<float>;
template class SecondOrderSupervisedMStep<double>;

}
