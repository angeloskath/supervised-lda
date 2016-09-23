#include "ldaplusplus/events/ProgressEvents.hpp"

#include "applications/EpochProgress.hpp"

EpochProgress::EpochProgress() {
    em_iterations_ = 0;
    likelihood_ = 0;
    cnt_likelihoods_ = 0;
    is_first_time_ = true;
}

void EpochProgress::on_event(std::shared_ptr<events::Event> event) {
    if (event->id() == "ExpectationProgressEvent") {
        auto progress = std::static_pointer_cast<events::ExpectationProgressEvent<double> >(event);

        if (is_first_time_) {
            std::cout << "E-M Iteration " << em_iterations_+1 << std::endl;
            is_first_time_ = false;
        }

        // Keep track of the likelihood
        if (std::isfinite(progress->likelihood()) && progress->likelihood() < 0) {
            likelihood_ += progress->likelihood();
            cnt_likelihoods_++;
        }
    }
    else if (event->id() == "EpochProgressEvent") {
        if (likelihood_ < 0) {
            std::cout << "Per document likelihood: " <<
                likelihood_ / cnt_likelihoods_ << std::endl;
        }

        // reset the member variables
        likelihood_ = 0;
        cnt_likelihoods_ = 0;
        em_iterations_ ++;
        is_first_time_ = true;
    }

}
