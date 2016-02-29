#ifndef _EVENTS_HPP_
#define _EVENTS_HPP_


#include <memory>
#include <list>


/**
 * A base event object that will be dispatched and received.
 */
class Event
{
    public:
        Event(std::string id);

        const std::string & id() const;

    private:
        std::string id_;
};


/**
 * A simple event listener interface.
 */
class IEventListener
{
    public:
        virtual void on_event(std::shared_ptr<Event> event) = 0;
};


/**
 * Creates an event listener from a function with the same signature.
 */
class FunctionEventListener : public IEventListener
{
    public:
        FunctionEventListener(std::function<void(std::shared_ptr<Event>)> listener);

        void on_event(std::shared_ptr<Event> event) override;

    private:
        std::function<void(std::shared_ptr<Event>)> listener_;
};


/**
 * IEventDispatcher is the interface of a very simple event dispatcher.
 */
class IEventDispatcher
{
    public:
        
        virtual void add_listener(std::shared_ptr<IEventListener> listener) = 0;
        virtual void remove_listener(std::shared_ptr<IEventListener> listener) = 0;
        virtual void dispatch(std::shared_ptr<Event> event) = 0;

        /**
         * The created object is returned so that the listener can be removed.
         */
        std::shared_ptr<IEventListener> add_listener(
            std::function<void(std::shared_ptr<Event>)> listener
        ) {
            auto l = std::make_shared<FunctionEventListener>(listener);

            add_listener(l);

            return l;
        }

        /**
         * The created object is returned so that the listener can be removed.
         */
        template <class ListenerType, typename... Args>
        std::shared_ptr<IEventListener> add_listener(Args... args) {
            auto l = std::make_shared<ListenerType>(args...);

            add_listener(l);

            return l;
        }

        template <class EventType, typename... Args>
        void dispatch(Args... args) {
            auto event = std::make_shared<EventType>(args...);

            dispatch(event);
        }
};


/**
 * EventDispatcher is a simple implementation of an IEventDispatcher. It can be
 * copied, passed by value, reference whatever. It is not thread safe.
 */
class EventDispatcher : public IEventDispatcher
{
    public:
        void add_listener(std::shared_ptr<IEventListener> listener);
        void remove_listener(std::shared_ptr<IEventListener> listener);
        void dispatch(std::shared_ptr<Event> event);

    private:
        std::list<std::shared_ptr<IEventListener> > listeners_;
};


#endif  // _EVENTS_HPP_
