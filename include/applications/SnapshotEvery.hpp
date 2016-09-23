#ifndef _APPLICATIONS_SNAPSHOTEVERY_HPP_
#define _APPLICATIONS_SNAPSHOTEVERY_HPP_

#include <iostream>

#include "ldaplusplus/Parameters.hpp"

#include "ldaplusplus/events/Events.hpp"

using namespace ldaplusplus;

class SnapshotEvery : public events::EventListenerInterface
{
    public:
        SnapshotEvery(std::string path, int save_every=10);

        void on_event(std::shared_ptr<events::Event> event);

        void snapshot(
            std::shared_ptr<parameters::Parameters> parameters
        );
    private:
        std::string path_;
        int save_every_;
        int seen_so_far_;
};

#endif //  _APPLICATIONS_SNAPSHOTEVERY_HPP_
