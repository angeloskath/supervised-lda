
#include <memory>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "ldaplusplus/Document.hpp"

using namespace Eigen;
using namespace ldaplusplus;


TEST(TestCorpus, TestDocument) {
    VectorXi X = VectorXi::Random(10).array().abs().matrix();

    auto doc = std::make_shared<corpus::EigenDocument>(X);

    for (int i=0; i<10; i++) {
        ASSERT_EQ(X[i], doc->get_words()[i]);
    }
}

TEST(TestCorpus, TestClassificationDecorator) {
    VectorXi X = VectorXi::Random(10).array().abs().matrix();

    auto doc = std::make_shared<corpus::ClassificationDecorator>(
        std::make_shared<corpus::EigenDocument>(X),
        13
    );

    for (int i=0; i<10; i++) {
        ASSERT_EQ(X[i], doc->get_words()[i]);
    }
    ASSERT_EQ(13, doc->get_class());
}

TEST(TestCorpus, TestEigenCorpus) {
    MatrixXi X = MatrixXi::Random(10, 100).array().abs().matrix();

    auto corpus = std::make_shared<corpus::EigenCorpus>(X);

    ASSERT_EQ(100, corpus->size());

    for (int i=0; i<100; i++) {
        for (int j=0; j<10; j++) {
            ASSERT_EQ(X(j, i), corpus->at(i)->get_words()[j]);
        }
    }
}
