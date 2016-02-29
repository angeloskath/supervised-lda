#ifndef _GRADIENT_DESCENT_HPP_
#define _GRADIENT_DESCENT_HPP_

#include <functional>
#include <memory>

#include <Eigen/Core>


using namespace Eigen;


/**
 * LineSearch is an interface that is meant to be used to update the parameter
 * x0 in the direction 'direction' by (probably) performing a line search for
 * the best value.
 */
template <typename ProblemType, typename ParameterType>
class LineSearch
{
    public:
        typedef typename ParameterType::Scalar Scalar;

        virtual Scalar search(
            const ProblemType &problem,
            Ref<ParameterType> x0,
            const ParameterType &grad_x0,
            const ParameterType &direction
        ) = 0;
};


/**
 * ConstantLineSearch simply updates the parameter by a costant factor of the
 * direction performing no line search actually.
 */
template <typename ProblemType, typename ParameterType>
class ConstantLineSearch : public LineSearch<ProblemType, ParameterType>
{
    public:
        typedef typename ParameterType::Scalar Scalar;

        ConstantLineSearch(Scalar alpha) : alpha_(alpha) {}

        Scalar search(
            const ProblemType &problem,
            Ref<ParameterType> x0,
            const ParameterType &grad_x0,
            const ParameterType &direction
        ) {
            x0 -= alpha_ * direction;

            return problem.value(x0);
        }

    private:
        Scalar alpha_;
};

/**
 * Armijo line search is a simple backtracking line search where the Armijo
 * condition is required.
 *
 * The Armijo condition is the following if p_k is the negative direction and
 * g_k the gradient.
 *
 *  f(x_k - a_k p_k) \leq f(x_k) - a_k b g_k^T p_k
 */
template <typename ProblemType, typename ParameterType>
class ArmijoLineSearch : public LineSearch<ProblemType, ParameterType>
{
    public:
        typedef typename ParameterType::Scalar Scalar;

        ArmijoLineSearch(Scalar beta=0.001, Scalar tau=0.5) : beta_(beta),
                                                              tau_(tau)
        {}

        Scalar search(
            const ProblemType &problem,
            Ref<ParameterType> x0,
            const ParameterType &grad_x0,
            const ParameterType &direction
        ) {
            ParameterType x_copy(x0.rows(), x0.cols());
            Scalar value_x0 = problem.value(x0);
            Scalar decrease = beta_ * (grad_x0.array() * direction.array()).sum();
            Scalar value = value_x0;
            Scalar a = 1.0/tau_;

            while (value > value_x0 - a * decrease) {
                a *= tau_;
                x_copy = x0 - a * direction;
                value = problem.value(x_copy);
            }

            x0 -= a * direction;

            return value;
        }

    private:
        Scalar beta_;
        Scalar tau_;
};


/**
 * A very simple implementation of batch gradient descent.
 */
template <typename ProblemType, typename ParameterType>
class GradientDescent
{
    public:
        typedef typename ParameterType::Scalar Scalar;

        GradientDescent(
            std::shared_ptr<LineSearch<ProblemType, ParameterType> > line_search,
            std::function<bool(Scalar, Scalar, size_t)> progress
        ) : line_search_(line_search),
            progress_(progress)
        {}

        void minimize(const ProblemType &problem, Ref<ParameterType> x0) {
            // allocate memory for the gradient
            ParameterType grad(x0.rows(), x0.cols());

            // Keep the value in this variable
            Scalar value = problem.value(x0);

            // And the iterations in this one
            size_t iterations = 0;

            // Whether we stop or not is decided by some one else
            while (progress_(value, grad.template lpNorm<Infinity>(), iterations++)) {
                problem.gradient(x0, grad);
                value = line_search_->search(problem, x0, grad, grad);
            }
        }

    private:
        std::shared_ptr<LineSearch<ProblemType, ParameterType> > line_search_;
        std::function<bool(Scalar, Scalar, size_t)> progress_;
};


#endif // _GRADIENT_DESCENT_HPP_
