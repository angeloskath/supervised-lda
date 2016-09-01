#ifndef _PARAMETERS_HPP_
#define _PARAMETERS_HPP_


#include <utility>

#include <Eigen/Core>

using namespace Eigen;

namespace ldaplusplus {
namespace parameters {


/**
 * All the parameter related objects will be extending this empty struct.
 */
struct Parameters
{
};


/**
 * ModelParameters contain the basic LDA model parameters namely the prior for
 * the documents over topics distribution and the topics over words
 * distributions.
 */
template <typename Scalar>
struct ModelParameters : public Parameters
{
    ModelParameters() {}
    ModelParameters(
        Matrix<Scalar, Dynamic, 1> a,
        Matrix<Scalar, Dynamic, Dynamic> b
    ) : alpha(std::move(a)),
        beta(std::move(b))
    {}

    Matrix<Scalar, Dynamic, 1> alpha;
    Matrix<Scalar, Dynamic, Dynamic> beta;
};


/**
 * SupervisedModelParameters adds the extra logistic regression parameters to
 * model parameters.
 */
template <typename Scalar>
struct SupervisedModelParameters : public ModelParameters<Scalar>
{
    SupervisedModelParameters() {}
    SupervisedModelParameters(
        Matrix<Scalar, Dynamic, 1> a,
        Matrix<Scalar, Dynamic, Dynamic> b,
        Matrix<Scalar, Dynamic, Dynamic> e
    ) : ModelParameters<Scalar>(a, b),
        eta(std::move(e))
    {}

    Matrix<Scalar, Dynamic, Dynamic> eta;
};


/**
 * The variational parameters are (duh) the variational parameters of the LDA
 * model.
 */
template <typename Scalar>
struct VariationalParameters : public Parameters
{
    VariationalParameters() {}
    VariationalParameters(
        Matrix<Scalar, Dynamic, 1> g,
        Matrix<Scalar, Dynamic, Dynamic> p
    ) : gamma(std::move(g)),
        phi(std::move(p))
    {}

    Matrix<Scalar, Dynamic, 1> gamma;
    Matrix<Scalar, Dynamic, Dynamic> phi;
};


/**
 * The supervised correspondence variational parameters add a variational
 * parameter for sampling a topic assignment to predict the class.
 */
template <typename Scalar>
struct SupervisedCorrespondenceVariationalParameters : public VariationalParameters<Scalar>
{
    SupervisedCorrespondenceVariationalParameters() {}
    SupervisedCorrespondenceVariationalParameters(
        Matrix<Scalar, Dynamic, 1> g,
        Matrix<Scalar, Dynamic, Dynamic> p,
        Matrix<Scalar, Dynamic, 1> t
    ) : VariationalParameters<Scalar>(g, p),
        tau(std::move(t))
    {}

    Matrix<Scalar, Dynamic, 1> tau;
};

}  // namespace parameters
}  // namespace ldaplusplus

#endif  // _PARAMETERS_HPP_
