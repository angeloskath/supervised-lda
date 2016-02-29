#ifndef _PROGRESS_VISITOR_HPP_
#define _PROGRESS_VISITOR_HPP_


#include "Events.hpp"


template <typename Scalar>
class LDA;


template <typename Scalar>
class ExpectationProgressEvent : public Event
{
    public:
        ExpectationProgressEvent(size_t iteration, Scalar likelihood) :
            Event("ExpectationProgressEvent"),
            iteration_(iteration),
            likelihood_(likelihood)
        {}

        size_t iteration() const { return iteration_; }
        Scalar likelihood() const { return likelihood_; }

    private:
        size_t iteration_;
        Scalar likelihood_;
};


template <typename Scalar>
class MaximizationProgressEvent : public Event
{
    public:
        MaximizationProgressEvent(size_t iteration, Scalar likelihood) :
            Event("MaximizationProgressEvent"),
            iteration_(iteration),
            likelihood_(likelihood)
        {}

        size_t iteration() const { return iteration_; }
        Scalar likelihood() const { return likelihood_; }

    private:
        size_t iteration_;
        Scalar likelihood_;
};


template <typename Scalar>
class EpochProgressEvent : public Event
{
    typedef typename LDA<Scalar>::LDAState LDAState;

    public:
        EpochProgressEvent(LDAState lda_state) :
            Event("EpochProgressEvent"),
            lda_state_(lda_state)
        {}

        const LDAState & lda_state() const { return lda_state_; }

    private:
        LDAState lda_state_;
};


#endif // _PROGRESS_VISITOR_HPP_
