
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
