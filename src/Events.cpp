
#include "Events.hpp"


Event::Event(std::string id) : id_(id) {}


const std::string & Event::id() const {
    return id_;
}


FunctionEventListener::FunctionEventListener(
    std::function<void(std::shared_ptr<Event>)> listener
) : listener_(listener)
{}


void FunctionEventListener::on_event(std::shared_ptr<Event> event) {
    listener_(event);
}


void EventDispatcher::add_listener(std::shared_ptr<IEventListener> listener) {
    listeners_.push_back(listener);
}


void EventDispatcher::remove_listener(std::shared_ptr<IEventListener> listener) {
    for (auto it = listeners_.begin(); it != listeners_.end(); it++) {
        if (*it == listener) {
            listeners_.erase(it);
            break;
        }
    }
}


void EventDispatcher::dispatch(std::shared_ptr<Event> event) {
    for (auto l : listeners_) {
        l->on_event(event);
    }
}


void ThreadSafeEventDispatcher::add_listener(
    std::shared_ptr<IEventListener> listener
) {
    std::lock_guard<std::mutex> l(listeners_mutex_);

    listeners_.push_back(listener);
}

void ThreadSafeEventDispatcher::remove_listener(
    std::shared_ptr<IEventListener> listener
) {
    std::lock_guard<std::mutex> l(deleted_listeners_mutex_);

    deleted_listeners_.insert(listener);
}

void ThreadSafeEventDispatcher::dispatch(std::shared_ptr<Event> event) {
    std::lock_guard<std::mutex> l(events_mutex_);

    events_.push_back(event);
}

void ThreadSafeEventDispatcher::process_events() {
    // first lock the queue and the deleted listeners together and delete the
    // listeners in the deleted list
    {
        std::unique_lock<std::mutex> l1(listeners_mutex_, std::defer_lock);
        std::unique_lock<std::mutex> l2(deleted_listeners_mutex_, std::defer_lock);

        std::lock(l1, l2);

        for (auto l=listeners_.begin(); l!=listeners_.end(); l++) {
            if (deleted_listeners_.find(*l) != deleted_listeners_.end()) {
                l = listeners_.erase(l);
                l--;
            }
        }

        deleted_listeners_.clear();
    }

    // copy the event list
    std::list<std::shared_ptr<Event> > events;
    {
        std::lock_guard<std::mutex> l(events_mutex_);
        events = events_;
        events_.clear();
    }

    // copy the listeners list
    std::list<std::shared_ptr<IEventListener> > listeners;
    {
        std::lock_guard<std::mutex> l(listeners_mutex_);
        listeners = listeners_;
    }

    // dispatch events without worry
    for (auto ev : events) {
        for (auto l : listeners) {
            l->on_event(ev);
        }
    }
}


SameThreadEventDispatcher::SameThreadEventDispatcher()
    : thread_id_(std::this_thread::get_id())
{}

void SameThreadEventDispatcher::dispatch(std::shared_ptr<Event> event) {
    ThreadSafeEventDispatcher::dispatch(event);

    if (std::this_thread::get_id() == thread_id_) {
        ThreadSafeEventDispatcher::process_events();
    }
}
