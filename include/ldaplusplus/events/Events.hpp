#ifndef _LDAPLUSPLUS_EVENTS_EVENTS_HPP_
#define _LDAPLUSPLUS_EVENTS_EVENTS_HPP_


#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>

namespace ldaplusplus {
namespace events {


/**
 * A base event object that will be dispatched and received.
 */
class Event
{
    public:
        /**
         * @param id A string that identifies the event
         */
        Event(std::string id);

        /**
         * @return A string that identifies the event (passed in the
         *         constructor)
         */
        const std::string & id() const;

    private:
        std::string id_;
};


/**
 * A simple event listener interface.
 */
class EventListenerInterface
{
    public:
        /**
         * @param event The received event
         */
        virtual void on_event(std::shared_ptr<Event> event) = 0;

        virtual ~EventListenerInterface(){};
};


/**
 * Creates an event listener from a function with the same signature.
 */
class FunctionEventListener : public EventListenerInterface
{
    public:
        FunctionEventListener(std::function<void(std::shared_ptr<Event>)> listener);

        void on_event(std::shared_ptr<Event> event) override;

    private:
        std::function<void(std::shared_ptr<Event>)> listener_;
};


/**
 * EventDispatcher is the interface of a very simple event dispatcher.
 */
class EventDispatcherInterface
{
    public:
        /**
         * Add a listener to this dispatcher so that in subsequent calls to
         * dispatch this listener will be notified.
         *
         * The interface doesn't define handling of double adds or double
         * removes.
         *
         * @param listener The listener to be added
         */
        virtual void add_listener(std::shared_ptr<EventListenerInterface> listener) = 0;

        /**
         * Remove a listener from this dispatcher so that in subsequent calls
         * to dispatch this listener will not be notified.
         *
         * The interface doesn't define handling of double adds or double
         * removes.
         *
         * @param listener The listener to be removed
         */
        virtual void remove_listener(std::shared_ptr<EventListenerInterface> listener) = 0;

        /**
         * Call the on_event() function of every listener passing this event as
         * a parameter.
         *
         * The interface doesn't define when the on_event function will be
         * called and it is not guaranteed that upon return of the dispatch
         * function all listeners will be notified.
         *
         * @param event The event to be sent to the listeners
         */
        virtual void dispatch(std::shared_ptr<Event> event) = 0;

        /**
         * Create on the fly a listener from the function and add it to the
         * dispatcher.
         *
         * The created object is returned so that the listener can be removed
         * later.
         *
         * @param listener A function implementing the EventListener interface.
         */
        std::shared_ptr<EventListenerInterface> add_listener(
            std::function<void(std::shared_ptr<Event>)> listener
        ) {
            auto l = std::make_shared<FunctionEventListener>(listener);

            add_listener(l);

            return l;
        }

        /**
         * Create on the fly a listener of type ListenerType and add it to the
         * dispatcher.
         *
         * The created object is returned so that the listener can be removed
         * later.
         *
         * @param args Variadic template arguments to be expanded as
         *             constructor parameters for the listener
         */
        template <class ListenerType, typename... Args>
        std::shared_ptr<EventListenerInterface> add_listener(Args... args) {
            auto l = std::make_shared<ListenerType>(args...);

            add_listener(l);

            return l;
        }

        /**
         * Create on the fly an event object and dispatch it.
         *
         * @param args Variadic template arguments to be expanded as
         *             constructor parameters for the event
         */
        template <class EventType, typename... Args>
        void dispatch(Args... args) {
            auto event = std::make_shared<EventType>(args...);

            dispatch(event);
        }

        virtual ~EventDispatcherInterface(){};
};


/**
 * EventDispatcher is a simple implementation of an EventDispatcherInterface. It can be
 * copied, passed by value, reference whatever. It is **not** thread safe.
 */
class EventDispatcher : public EventDispatcherInterface
{
    public:
        void add_listener(std::shared_ptr<EventListenerInterface> listener) override;
        void remove_listener(std::shared_ptr<EventListenerInterface> listener) override;
        void dispatch(std::shared_ptr<Event> event) override;

    private:
        std::list<std::shared_ptr<EventListenerInterface> > listeners_;
};


/**
 * This helper class allows us to have an event dispatcher composition just by
 * extending from it.
 */
class EventDispatcherComposition
{
    public:
        EventDispatcherComposition() :
            event_dispatcher_(std::make_shared<EventDispatcher>())
        {}

        std::shared_ptr<EventDispatcherInterface> get_event_dispatcher() {
            return event_dispatcher_;
        }

        void set_event_dispatcher(std::shared_ptr<EventDispatcherInterface> dispatcher) {
            event_dispatcher_ = dispatcher;
        }

    private:
        std::shared_ptr<EventDispatcherInterface> event_dispatcher_;
};


/**
 * A thread safe event dispatcher that dispatches the events when its process_events
 * method is called and on the thread that the process_events method is called.
 */
class ThreadSafeEventDispatcher : public EventDispatcherInterface
{
    public:
        virtual void add_listener(std::shared_ptr<EventListenerInterface> listener) override;
        virtual void remove_listener(std::shared_ptr<EventListenerInterface> listener) override;
        virtual void dispatch(std::shared_ptr<Event> event) override;

        /**
         * Traverse the listener queue and notify them of any events that
         * occured since the last time.
         *
         * The listeners will be called on the thread that this method is
         * called.
         */
        void process_events();

    private:
        std::mutex listeners_mutex_;
        std::list<std::shared_ptr<EventListenerInterface> > listeners_;

        std::mutex deleted_listeners_mutex_;
        std::unordered_set<std::shared_ptr<EventListenerInterface> > deleted_listeners_;

        std::mutex events_mutex_;
        std::list<std::shared_ptr<Event> > events_;
};


/**
 * SameThreadEventDispatcher dispatches immediately any events that are
 * dispatched from the thread in which it was created and only allows
 * process_events to be called from that thread.
 * 
 * It is thread safe.
 */
class SameThreadEventDispatcher : public ThreadSafeEventDispatcher
{
    public:
        SameThreadEventDispatcher();

        virtual void dispatch(std::shared_ptr<Event> event) override;

    private:
        std::thread::id thread_id_;
};

}  // namespace events
}  // namespace ldaplusplus

#endif  // _LDAPLUSPLUS_EVENTS_EVENTS_HPP_
