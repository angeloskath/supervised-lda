#include "ldaplusplus/events/ProgressEvents.hpp"

#include "applications/MaximizationProgress.hpp"

MaximizationProgress::MaximizationProgress() {
    m_iterations_ = 0;
}

void MaximizationProgress::on_event(std::shared_ptr<events::Event> event) {
    if (event->id() == "MaximizationProgressEvent") {

        auto progress = std::static_pointer_cast<events::MaximizationProgressEvent<double> >(event);
        std::cout << "log p(y | \\bar{z}, eta): " << progress->likelihood() << std::endl;
        m_iterations_++;
    }
    else if (event->id() == "EpochProgressEvent") {
        // If one Epoch is completed reset the member variables
        m_iterations_ = 0;
    }
}
