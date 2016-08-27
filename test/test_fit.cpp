#include <random>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "test/utils.hpp"

#include "ldaplusplus/IEStep.hpp"
#include "ldaplusplus/IMStep.hpp"
#include "ldaplusplus/LDABuilder.hpp"
#include "ldaplusplus/LDA.hpp"
#include "ldaplusplus/ProgressEvents.hpp"

using namespace Eigen;
using namespace ldaplusplus;


// T will be available as TypeParam in TYPED_TEST functions
template <typename T>
class TestFit : public ParameterizedTest<T> {};

TYPED_TEST_CASE(TestFit, ForFloatAndDouble);


TYPED_TEST(TestFit, partial_fit) {
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

    LDABuilder<TypeParam> b;
    LDA<TypeParam> lda = b.
            set_e(b.get_supervised_e_step(10, 1e-2, 10)).
            set_supervised_batch_m_step(10, 1e-2).
            initialize_topics("seeded", X, 10).
            initialize_eta("zeros", X, y, 10);

    TypeParam likelihood0, likelihood=0, py0, py;
    lda.get_event_dispatcher()->add_listener(
        [&likelihood, &py](std::shared_ptr<Event> event) {
            //if (event->id() == "ExpectationProgressEvent") {
            //    auto progress = std::static_pointer_cast<ExpectationProgressEvent<TypeParam> >(event);
            //    likelihood += progress->likelihood();
            //}

            if (event->id() == "MaximizationProgressEvent") {
                auto progress = std::static_pointer_cast<MaximizationProgressEvent<TypeParam> >(event);
                py = progress->likelihood();
            }
        }
    );

    lda.partial_fit(X, y);
    likelihood0 = likelihood;
    likelihood = 0;
    py0 = py;
    lda.partial_fit(X, y);

    //EXPECT_GT(likelihood, likelihood0);
    EXPECT_GT(py, py0);
}
