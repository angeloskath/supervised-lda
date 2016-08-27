#ifndef _PROGRESS_EVENTS_HPP_
#define _PROGRESS_EVENTS_HPP_


#include "ldaplusplus/events/Events.hpp"
#include "ldaplusplus/Parameters.hpp"

namespace ldaplusplus {

template <typename Scalar>
class LDA;

namespace events {


template <typename Scalar>
class ExpectationProgressEvent : public Event
{
    public:
        ExpectationProgressEvent(Scalar likelihood) :
            Event("ExpectationProgressEvent"),
            likelihood_(likelihood)
        {}

        Scalar likelihood() const { return likelihood_; }

    private:
        Scalar likelihood_;
};


template <typename Scalar>
class MaximizationProgressEvent : public Event
{
    public:
        MaximizationProgressEvent(Scalar likelihood) :
            Event("MaximizationProgressEvent"),
            likelihood_(likelihood)
        {}

        Scalar likelihood() const { return likelihood_; }

    private:
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

}  // namespace events
}  // namespace ldaplusplus

#endif // _PROGRESS_EVENTS_HPP_
