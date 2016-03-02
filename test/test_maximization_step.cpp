
#include <memory>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "test/utils.hpp"

#include "SupervisedMStep.hpp"
#include "ProgressEvents.hpp"

using namespace Eigen;


// T will be available as TypeParam in TYPED_TEST functions
template <typename T>
class TestMaximizationStep : public ParameterizedTest<T> {};

TYPED_TEST_CASE(TestMaximizationStep, ForFloatAndDouble);

TYPED_TEST(TestMaximizationStep, Maximization) {
    std::mt19937 rng;
    rng.seed(0);

    SupervisedMStep<TypeParam> m_step(100, 1e-3, 1e-2);

    std::vector<TypeParam> progress;
    m_step.get_event_dispatcher()->add_listener(
        [&progress](std::shared_ptr<Event> event) {
            if (event->id() == "MaximizationProgressEvent") {
                auto prog_ev = std::static_pointer_cast<MaximizationProgressEvent<TypeParam> >(event);
                progress.push_back(prog_ev->likelihood());
            }
        }
    );

    MatrixX<TypeParam> expected_z_bar = MatrixX<TypeParam>::Random(10, 100);
    MatrixX<TypeParam> b(10, 100);
    VectorXi y(100);
    std::uniform_int_distribution<> dis(0, 5);
    for (int i=0; i<100; i++) {
        y[i] = dis(rng);
    }
    MatrixX<TypeParam> beta(10, 100);
    MatrixX<TypeParam> eta = MatrixX<TypeParam>::Zero(10, 6);

    m_step.m_step(
        expected_z_bar,
        b,
        y,
        beta,
        eta
    );

    ASSERT_GT(progress.size(), 2);
    for (size_t i=1; i<progress.size(); i++) {
        EXPECT_LT(progress[i-1], progress[i]) << "Iteration:" << i;
    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
