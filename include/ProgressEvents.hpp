#ifndef _PROGRESS_VISITOR_HPP_
#define _PROGRESS_VISITOR_HPP_


#include "Events.hpp"
#include "Parameters.hpp"


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
    public:
        EpochProgressEvent(const std::shared_ptr<Parameters> parameters) :
            Event("EpochProgressEvent"),
            model_parameters_(parameters)
        {}

        const std::shared_ptr<Parameters> model_parameters() const {
            return model_parameters_;
        }

    private:
       std::shared_ptr<Parameters> model_parameters_;
};


#endif // _PROGRESS_VISITOR_HPP_
