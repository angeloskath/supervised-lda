#ifndef _TEST_PARAMETERIZED_TEST_HPP_
#define _TEST_PARAMETERIZED_TEST_HPP_


#include <gtest/gtest.h>
#include <eigen3/Eigen/Core>


typedef ::testing::Types<float, double> ForFloatAndDouble;


template <typename T>
class ParameterizedTest : public ::testing::Test {};

template <typename T>
using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using VectorX = Eigen::Matrix<T, Eigen::Dynamic, 1>;


#endif  // _TEST_PARAMETERIZED_TEST_HPP_
