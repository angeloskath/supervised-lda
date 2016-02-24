#ifndef _GRADIENT_DESCENT_HPP_
#define _GRADIENT_DESCENT_HPP_

#include <functional>
#include <memory>


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
            ParameterType &x0,
            ParameterType &direction
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
            ParameterType &x0,
            ParameterType &direction
        ) {
            x0 -= alpha_ * direction;

            return problem.value(x0);
        }

    private:
        Scalar alpha_;
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

        void minimize(const ProblemType &problem, ParameterType &x0) {
            // allocate memory for the gradient
            ParameterType grad(x0.rows(), x0.cols());

            // Keep the value in this variable
            Scalar value = problem.value(x0);

            // And the iterations in this one
            size_t iterations = 0;

            // Whether we stop or not is decided by some one else
            while (progress_(value, grad.template lpNorm<Infinity>(), iterations++)) {
                problem.gradient(x0, grad);
                value = line_search_->search(problem, x0, grad);
            }
        }

    private:
        std::shared_ptr<LineSearch<ProblemType, ParameterType> > line_search_;
        std::function<bool(Scalar, Scalar, size_t)> progress_;
};


#endif // _GRADIENT_DESCENT_HPP_
