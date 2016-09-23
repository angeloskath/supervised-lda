#include "ldaplusplus/events/ProgressEvents.hpp"

#include "applications/ExpectationProgress.hpp"

ExpectationProgress::ExpectationProgress(int print_every) {
    e_iterations_ = 0;
    print_every_ = print_every;
}

void ExpectationProgress::on_event(std::shared_ptr<events::Event> event) {
    if (event->id() == "ExpectationProgressEvent") {

        e_iterations_++;
        if (e_iterations_ % print_every_ == 0) {
            std::cout << e_iterations_ << std::endl;
        }
    }
    else if (event->id() == "EpochProgressEvent") {
        // If one Epoch is completed reset the member variables
        e_iterations_ = 0;
    }
}
