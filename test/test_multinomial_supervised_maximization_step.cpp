
#include <memory>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "test/utils.hpp"

#include "ldaplusplus/Parameters.hpp"
#include "ldaplusplus/ProgressEvents.hpp"
#include "ldaplusplus/MultinomialSupervisedEStep.hpp"
#include "ldaplusplus/MultinomialSupervisedMStep.hpp"

using namespace Eigen;
using namespace ldaplusplus;


// T will be available as TypeParam in TYPED_TEST functions
template <typename T>
class TestMultinomialMaximizationStep : public ParameterizedTest<T> {};

TYPED_TEST_CASE(TestMultinomialMaximizationStep, ForFloatAndDouble);

TYPED_TEST(TestMultinomialMaximizationStep, Maximization) {
    // Build the corpus
    std::mt19937 rng;
    rng.seed(0);
    MatrixXi X(100, 50);
    VectorXi y(50);
    std::uniform_int_distribution<> class_generator(0, 5);
    std::exponential_distribution<> words_generator(0.1);
    for (int d=0; d<50; d++) {
        for (int w=0; w<100; w++) {
            X(w, d) = static_cast<int>(words_generator(rng));
        }
        y(d) = class_generator(rng);
    }

    // Create the corpus and the model
    auto corpus = std::make_shared<EigenClassificationCorpus>(X, y);
    MatrixX<TypeParam> beta = MatrixX<TypeParam>::Random(10, 100);
    beta.array() -= beta.minCoeff();
    beta.array().rowwise() /= beta.array().colwise().sum();
    auto model = std::make_shared<SupervisedModelParameters<TypeParam> >(
        VectorX<TypeParam>::Constant(10, 0.1),
        beta,
        MatrixX<TypeParam>::Constant(10, 6, 1. / 6)
    );

    MultinomialSupervisedEStep<TypeParam> e_step(10, 1e-2, 2);
    MultinomialSupervisedMStep<TypeParam> m_step(2);

    for (size_t i=0; i<corpus->size(); i++) {
        m_step.doc_m_step(
            corpus->at(i),
            e_step.doc_e_step(
                corpus->at(i),
                model
            ),
            model
        );
    }

    std::vector<TypeParam> progress;
    m_step.get_event_dispatcher()->add_listener(
        [&progress](std::shared_ptr<Event> event) {
            if (event->id() == "MaximizationProgressEvent") {
                auto prog_ev = std::static_pointer_cast<MaximizationProgressEvent<TypeParam> >(event);
                progress.push_back(prog_ev->likelihood());
            }
        }
    );

    m_step.m_step(
        model
    );

    ASSERT_EQ(1, progress.size());
    ASSERT_GT(0, progress[0]);
}
