#ifndef _APPLICATIONS_EPOCHPROGRESS_HPP_
#define _APPLICATIONS_EPOCHPROGRESS_HPP_

#include <iostream>

#include "ldaplusplus/events/Events.hpp"

using namespace ldaplusplus;

/**
  * This class is used to keep track of the progress of a complete Expectation
  * - Maximization step.
  */
class EpochProgress : public events::EventListenerInterface
{
    public:
        EpochProgress();

        void on_event(std::shared_ptr<events::Event> event);

    private:
       int em_iterations_;
       int likelihood_;
       int cnt_likelihoods_;
       int is_first_time_;
};

#endif  // _APPLICATIONS_EPOCHPROGRESS_HPP_
