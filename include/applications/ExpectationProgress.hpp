#ifndef _EXPECTATIONPROGRESS_HPP_
#define _EXPECTATIONPROGRESS_HPP_

#include <iostream>

#include "ldaplusplus/events/Events.hpp"
#include "ldaplusplus/events/ProgressEvents.hpp"

using namespace ldaplusplus;

/**
  * This class is used to keep track of the progress during the Expecation
  * step.
  */
class ExpectationProgress : public events:EventListenerInterface {
    public:
        ExpectationProgress(int print_every = 100);

        void on_event(std::shared_ptr<events::Event> event);

    private:
        int e_iterations_;
        int print_every_;
        
};

#endif // _EXPECTATIONPROGRESS_HPP_
