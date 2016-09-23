#include "ldaplusplus/events/ProgressEvents.hpp"

#include "applications/lda_io.hpp"
#include "applications/SnapshotEvery.hpp"

SnapshotEvery::SnapshotEvery(std::string path, int save_every) {
    seen_so_far_ = 0;
    save_every_ = save_every;
    path_ = std::move(path);
}

void SnapshotEvery::snapshot(
    std::shared_ptr<parameters::Parameters> parameters
) {
    std::stringstream actual_path;
    actual_path << path_ << "_";
    actual_path.fill('0');
    actual_path.width(3);
    actual_path << seen_so_far_;

    io::save_lda(actual_path.str(), parameters);
}

void SnapshotEvery::on_event(std::shared_ptr<events::Event> event) {
    if (event->id() == "EpochProgressEvent") {
        auto progress = std::static_pointer_cast<events::EpochProgressEvent<double> >(event);

        seen_so_far_ ++;
        if (seen_so_far_ % save_every_ == 0) {
            snapshot(progress->model_parameters());
        }
    }
}
