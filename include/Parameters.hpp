#ifndef _PARAMETERS_HPP_
#define _PARAMETERS_HPP_


#include <Eigen/Core>

using namespace Eigen;


/**
 * All the parameter related objects will be extending this empty struct.
 */
struct Parameters
{
};


/**
 * ModelParameters contains the basic LDA model parameters namely the prior for
 * the documents over topics distribution and the topics over words
 * distributions.
 */
template <typename Scalar>
struct ModelParameters : public Parameters
{
    Matrix<Scalar, Dynamic, 1> alpha;
    Matrix<Scalar, Dynamic, Dynamic> beta;
};


/**
 * SupervisedModelParameters adds the linear model's parameters to LDA model.
 */
template <typename Scalar>
struct SupervisedModelParameters : public ModelParameters<Scalar>
{
    Matrix<Scalar, Dynamic, Dynamic> eta;
};


/**
 * The variational parameters are (duh) the variational parameters of the LDA
 * model.
 */
template <typename Scalar>
struct VariationalParameters : public Parameters
{
    Matrix<Scalar, Dynamic, 1> gamma;
    Matrix<Scalar, Dynamic, Dynamic> phi;
};


#endif  // _PARAMETERS_HPP_
