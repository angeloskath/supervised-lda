#ifndef _APPLICATIONS_EXPECTATIONPROGRESS_HPP_
#define _APPLICATIONS_EXPECTATIONPROGRESS_HPP_

#include <iostream>

#include "ldaplusplus/events/Events.hpp"

using namespace ldaplusplus;

/**
  * This class is used to keep track of the progress during the Expectation
  * step.
  */
class ExpectationProgress : public events::EventListenerInterface
{
    public:
        ExpectationProgress(int print_every = 100);

        void on_event(std::shared_ptr<events::Event> event);

    private:
        // The number of completed doc_e_step iterations so far.
        int e_iterations_;
        // A number that indicates after how many documents an output message,
        // concerning the progress in the Expectation step, will be printed
        int print_every_;
        
};

#endif // _APPLICATIONS_EXPECTATIONPROGRESS_HPP_
