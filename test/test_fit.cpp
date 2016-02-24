
#include <random>
#include <vector>
#include <iostream>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "test/utils.hpp"

#include "SupervisedLDA.hpp"
#include "ProgressVisitor.hpp"

using namespace Eigen;


// T will be available as TypeParam in TYPED_TEST functions
template <typename T>
class TestFit : public ParameterizedTest<T> {};

TYPED_TEST_CASE(TestFit, ForFloatAndDouble);


TYPED_TEST(TestFit, partial_fit) {
    std::mt19937 rng;
    rng.seed(0);

    SupervisedLDA<TypeParam> lda(10);

    std::vector<Progress<TypeParam> > progress;
    TypeParam likelihood0, likelihood, py0, py;
    auto visitor = std::make_shared<FunctionVisitor<TypeParam> >(
        [&progress, &likelihood, &py](Progress<TypeParam> p) {
            progress.push_back(p);

            switch (p.state) {
                case Expectation:
                    likelihood = p.value;
                    //if ((p.partial_iteration + 1) % 10) {
                    //    std::cout << "log(Likelihood): " << p.value/(p.partial_iteration + 1) << std::endl;
                    //}
                    break;
                case Maximization:
                    py = -p.value;
                    //std::cout << "log p(y | z_bar, eta): " << -p.value << std::endl;
                    break;
            }
        }
    );

    lda.set_progress_visitor(visitor);

    MatrixXi X(100, 500);
    VectorXi y(500);
    std::uniform_int_distribution<> class_generator(0, 5);
    std::exponential_distribution<> words_generator(0.1);
    for (int d=0; d<500; d++) {
        for (int w=0; w<100; w++) {
            X(w, d) = static_cast<int>(words_generator(rng));
        }
        y(d) = class_generator(rng);
    }

    lda.partial_fit(X, y);
    progress.clear();
    likelihood0 = likelihood;
    py0 = py;

    lda.partial_fit(X, y);
    EXPECT_GT(likelihood, likelihood0);
    EXPECT_GT(py, py0);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
