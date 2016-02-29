#ifndef _PROGRESS_VISITOR_HPP_
#define _PROGRESS_VISITOR_HPP_


template <typename Scalar>
class LDA;


enum ProgressState { Expectation, Maximization, IterationFinished };


template <typename Scalar>
struct Progress
{
    /** Which part of the procedure does this progress refer to */
    ProgressState state;

    /**
     * The actual progress value which for SupervisedLDA is log
     * likelihood both for Expectation and Maximization.
     */
    Scalar value;

    /** The partial iteration value of the E or M step */
    size_t partial_iteration;

    /** The iteration of the EM pair  */
    size_t iteration;

    /** The current lda state */
    typename LDA<Scalar>::LDAState lda_state;
};


template <typename Scalar>
class IProgressVisitor
{
    public:
        virtual void visit(Progress<Scalar> progress) = 0;
};


template <typename Scalar>
class FunctionVisitor : public IProgressVisitor<Scalar>
{
    public:
        FunctionVisitor(std::function<void(Progress<Scalar>)> f) : f_(f) {}

        virtual void visit(Progress<Scalar> progress) {
            return f_(progress);
        }

    private:
        std::function<void(Progress<Scalar>)> f_;
};


#endif // _PROGRESS_VISITOR_HPP_
