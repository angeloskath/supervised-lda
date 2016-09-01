#ifndef _PARAMETERS_HPP_
#define _PARAMETERS_HPP_


#include <utility>

#include <Eigen/Core>

//using namespace Eigen;

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
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> a,
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> b
    ) : alpha(std::move(a)),
        beta(std::move(b))
    {}

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> alpha;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> beta;
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
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> a,
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> b,
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> e
    ) : ModelParameters<Scalar>(a, b),
        eta(std::move(e))
    {}

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eta;
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
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> g,
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> p
    ) : gamma(std::move(g)),
        phi(std::move(p))
    {}

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> gamma;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> phi;
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
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> g,
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> p,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> t
    ) : VariationalParameters<Scalar>(g, p),
        tau(std::move(t))
    {}

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> tau;
};

}  // namespace parameters
}  // namespace ldaplusplus

#endif  // _PARAMETERS_HPP_
