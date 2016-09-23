#ifndef _APPLICATIONS_MAXIMIZATIONPROGRESS_HPP_
#define _APPLICATIONS_MAXIMIZATIONPROGRESS_HPP_

#include <iostream>

#include "ldaplusplus/events/Events.hpp"

using namespace ldaplusplus;

/**
  * This class is used to keep track of the progress during the Maximization
  * step.
  */
class MaximizationProgress : public events::EventListenerInterface
{
    public:
        MaximizationProgress();

        void on_event(std::shared_ptr<events::Event> event);

    private:
        // The number of completed doc_m_step iterations
        int m_iterations_;
};

#endif // _APPLICATIONS_MAXIMIZATIONPROGRESS_HPP_
