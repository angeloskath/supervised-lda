#ifndef _TEST_PARAMETERIZED_TEST_HPP_
#define _TEST_PARAMETERIZED_TEST_HPP_


#include <gtest/gtest.h>
#include <Eigen/Core>


typedef ::testing::Types<float, double> ForFloatAndDouble;


template <typename T>
class ParameterizedTest : public ::testing::Test {};

template <typename T>
using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using VectorX = Eigen::Matrix<T, Eigen::Dynamic, 1>;


template <typename T>
struct is_float {
    enum IsFloat { value = false };
};
template <>
struct is_float<float> {
    enum IsFloat { value = true };
};

template <typename T>
struct is_double {
    enum IsDouble { value = false };
};
template <>
struct is_double<double> {
    enum IsDouble { value = true };
};


#endif  // _TEST_PARAMETERIZED_TEST_HPP_
