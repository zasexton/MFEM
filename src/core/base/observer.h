#pragma once

#ifndef BASE_OBSERVER_H
#define BASE_OBSERVER_H

#include <functional>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <typeindex>
#include <string>
#include <string_view>
#include <concepts>
#include <chrono>
#include <atomic>
#include <thread>

namespace fem::core::base {

// Forward declarations
    class Event;
    class EventDispatcher;

/**
 * @brief Base interface for all events in the system
 *
 * Events represent something that happened in the system and contain
 * information about what occurred. They are immutable after creation.
 */
    class Event {
    public:
        /**
         * @brief Constructor with event type name
         */
        explicit Event(std::string_view event_type);

        /**
         * @brief Virtual destructor
         */
        virtual ~Event() = default;

        /**
         * @brief Get event type information
         */
        [[nodiscard]] virtual std::type_index get_type() const = 0;

        /**
         * @brief Get human-readable event type name
         */
        [[nodiscard]] const std::string& get_type_name() const { return type_name_; }

        /**
         * @brief Get event timestamp
         */
        [[nodiscard]] auto get_timestamp() const { return timestamp_; }

        /**
         * @brief Get event source (optional identifier of what generated this event)
         */
        [[nodiscard]] const std::string& get_source() const { return source_; }

        /**
         * @brief Set event source
         */
        void set_source(std::string_view source) { source_ = source; }

        /**
         * @brief Check if event has been handled
         */
        [[nodiscard]] bool is_handled() const { return handled_; }

        /**
         * @brief Mark event as handled (stops further propagation)
         */
        void set_handled(bool handled = true) { handled_ = handled; }

        /**
         * @brief Get string representation of event
         */
        virtual std::string to_string() const;

    protected:
        std::string type_name_;
        std::string source_;
        std::chrono::steady_clock::time_point timestamp_;
        mutable bool handled_{false};
    };

/**
 * @brief CRTP helper for typed events
 */
    template<typename Derived>
    class TypedEvent : public Event {
    public:
        explicit TypedEvent(std::string_view event_type = "")
                : Event(event_type.empty() ? typeid(Derived).name() : event_type) {}

        /**
         * @brief Get event type (automatically derived)
         */
        [[nodiscard]] std::type_index get_type() const override {
            return std::type_index(typeid(Derived));
        }

        /**
         * @brief Safe cast to derived type
         */
        template<typename T>
        [[nodiscard]] const T* as() const {
            return dynamic_cast<const T*>(this);
        }
    };

/**
 * @brief Observer interface for receiving events
 */
    template<typename EventType>
    class Observer {
    public:
        /**
         * @brief Virtual destructor
         */
        virtual ~Observer() = default;

        /**
         * @brief Handle an event of the specified type
         */
        virtual void on_event(const EventType& event) = 0;

        /**
         * @brief Optional: Handle any event (for logging, debugging)
         */
        virtual void on_any_event(const Event& event) {}

    protected:
        Observer() = default;
    };

/**
 * @brief Subject that can notify observers about events
 */
    template<typename EventType>
    class Subject {
    public:
        using observer_ptr = std::weak_ptr<Observer<EventType>>;
        using observer_list = std::vector<observer_ptr>;

        /**
         * @brief Add an observer
         */
        void add_observer(std::shared_ptr<Observer<EventType>> observer) {
            std::lock_guard lock(observers_mutex_);
            observers_.push_back(observer);
        }

        /**
         * @brief Remove an observer
         */
        void remove_observer(std::shared_ptr<Observer<EventType>> observer) {
            std::lock_guard lock(observers_mutex_);
            observers_.erase(
                    std::remove_if(observers_.begin(), observers_.end(),
                                   [&observer](const observer_ptr& weak_obs) {
                                       return weak_obs.expired() || weak_obs.lock() == observer;
                                   }),
                    observers_.end()
            );
        }

        /**
         * @brief Notify all observers about an event
         */
        void notify_observers(const EventType& event) {
            // Clean up expired observers and notify valid ones
            std::lock_guard lock(observers_mutex_);

            auto it = observers_.begin();
            while (it != observers_.end()) {
                if (auto observer = it->lock()) {
                    observer->on_event(event);
                    ++it;
                } else {
                    it = observers_.erase(it); // Remove expired observer
                }
            }
        }

        /**
         * @brief Get number of active observers
         */
        [[nodiscard]] size_t get_observer_count() const {
            std::lock_guard lock(observers_mutex_);
            size_t count = 0;
            for (const auto& weak_obs : observers_) {
                if (!weak_obs.expired()) ++count;
            }
            return count;
        }

        /**
         * @brief Clear all observers
         */
        void clear_observers() {
            std::lock_guard lock(observers_mutex_);
            observers_.clear();
        }

    protected:
        mutable std::mutex observers_mutex_;
        observer_list observers_;
    };

/**
 * @brief Event handler function type
 */
    template<typename EventType>
    using EventHandler = std::function<void(const EventType&)>;

/**
 * @brief Universal event handler that can handle any event
 */
    using UniversalEventHandler = std::function<void(const Event&)>;

/**
 * @brief Event subscription handle for managing subscriptions
 */
    class EventSubscription {
    public:
        using unsubscribe_func = std::function<void()>;

        explicit EventSubscription(unsubscribe_func unsubscriber)
                : unsubscriber_(std::move(unsubscriber)) {}

        ~EventSubscription() {
            unsubscribe();
        }

        // Non-copyable but movable
        EventSubscription(const EventSubscription&) = delete;
        EventSubscription& operator=(const EventSubscription&) = delete;

        EventSubscription(EventSubscription&& other) noexcept
                : unsubscriber_(std::move(other.unsubscriber_)) {
                other.unsubscriber_ = nullptr;
        }

        EventSubscription& operator=(EventSubscription&& other) noexcept {
            if (this != &other) {
                unsubscribe();
                unsubscriber_ = std::move(other.unsubscriber_);
                other.unsubscriber_ = nullptr;
            }
            return *this;
        }

        /**
         * @brief Manually unsubscribe from events
         */
        void unsubscribe() {
            if (unsubscriber_) {
                unsubscriber_();
                unsubscriber_ = nullptr;
            }
        }

        /**
         * @brief Check if subscription is still active
         */
        [[nodiscard]] bool is_active() const {
            return unsubscriber_ != nullptr;
        }

    private:
        unsubscribe_func unsubscriber_;
    };

/**
 * @brief Central event dispatcher for type-safe event handling
 *
 * Provides a centralized way to register event handlers and dispatch events.
 * Thread-safe and supports both typed and universal event handlers.
 */
    class EventDispatcher {
    public:
        /**
         * @brief Subscribe to events of a specific type
         */
        template<typename EventType>
        [[nodiscard]] std::unique_ptr<EventSubscription> subscribe(EventHandler<EventType> handler) {
            std::lock_guard outer_lock(handlers_mutex_);

            std::type_index type_id(typeid(EventType));
            auto wrapper = [handler](const Event& event) {
                if (const auto* typed_event = dynamic_cast<const EventType*>(&event)) {
                    handler(*typed_event);
                }
            };

            auto& handler_list = typed_handlers_[type_id];
            auto id = next_handler_id_++;
            handler_list[id] = wrapper;

            auto unsubscriber = [this, type_id, id]() {
                std::lock_guard guard(handlers_mutex_);
                if (auto it = typed_handlers_.find(type_id); it != typed_handlers_.end()) {
                    it->second.erase(id);
                    if (it->second.empty()) {
                        typed_handlers_.erase(it);
                    }
                }
            };

            return std::make_unique<EventSubscription>(unsubscriber);
        }

        /**
         * @brief Subscribe to all events (useful for logging, debugging)
         */
        [[nodiscard]] std::unique_ptr<EventSubscription> subscribe_all(UniversalEventHandler handler) {
            std::lock_guard outer_lock(handlers_mutex_);

            auto id = next_handler_id_++;
            universal_handlers_[id] = handler;

            auto unsubscriber = [this, id]() {
                std::lock_guard guard(handlers_mutex_);
                universal_handlers_.erase(id);
            };

            return std::make_unique<EventSubscription>(unsubscriber);
        }

        /**
         * @brief Dispatch an event to all registered handlers
         */
        void dispatch(const Event& event) {
            std::shared_lock lock(handlers_mutex_);

            // Call universal handlers first
            for (const auto& [id, handler] : universal_handlers_) {
                if (!event.is_handled()) {
                    handler(event);
                }
            }

            // Call typed handlers
            auto type_id = event.get_type();
            if (auto it = typed_handlers_.find(type_id); it != typed_handlers_.end()) {
                for (const auto& [id, handler] : it->second) {
                    if (!event.is_handled()) {
                        handler(event);
                    }
                }
            }
        }

        /**
         * @brief Create and dispatch an event
         */
        template<typename EventType, typename... Args>
        void emit(Args&&... args) {
            EventType event(std::forward<Args>(args)...);
            dispatch(event);
        }

        /**
         * @brief Get statistics about registered handlers
         */
        struct Statistics {
            size_t typed_handler_types{0};
            size_t total_typed_handlers{0};
            size_t universal_handlers{0};
        };

        [[nodiscard]] Statistics get_statistics() const {
            std::shared_lock lock(handlers_mutex_);

            Statistics stats;
            stats.typed_handler_types = typed_handlers_.size();
            stats.universal_handlers = universal_handlers_.size();

            for (const auto& [type, handlers] : typed_handlers_) {
                stats.total_typed_handlers += handlers.size();
            }

            return stats;
        }

        /**
         * @brief Clear all handlers
         */
        void clear() {
            std::lock_guard lock(handlers_mutex_);
            typed_handlers_.clear();
            universal_handlers_.clear();
        }

    private:
        using handler_id = uint64_t;
        using wrapped_handler = std::function<void(const Event&)>;

        mutable std::shared_mutex handlers_mutex_;
        std::unordered_map<std::type_index, std::unordered_map<handler_id, wrapped_handler>> typed_handlers_;
        std::unordered_map<handler_id, UniversalEventHandler> universal_handlers_;
        std::atomic<handler_id> next_handler_id_{1};
    };

/**
 * @brief Global event bus singleton for application-wide event communication
 */
    class EventBus {
    public:
        /**
         * @brief Get the global event bus instance
         */
        static EventBus& instance() {
            static EventBus bus;
            return bus;
        }

        /**
         * @brief Get the underlying event dispatcher
         */
        EventDispatcher& get_dispatcher() { return dispatcher_; }
        const EventDispatcher& get_dispatcher() const { return dispatcher_; }

        /**
         * @brief Subscribe to events (forwarded to dispatcher)
         */
        template<typename EventType>
        [[nodiscard]] std::unique_ptr<EventSubscription> subscribe(EventHandler<EventType> handler) {
            return dispatcher_.subscribe<EventType>(std::move(handler));
        }

        /**
         * @brief Subscribe to all events (forwarded to dispatcher)
         */
        [[nodiscard]] std::unique_ptr<EventSubscription> subscribe_all(UniversalEventHandler handler) {
            return dispatcher_.subscribe_all(std::move(handler));
        }

        /**
         * @brief Dispatch event (forwarded to dispatcher)
         */
        void dispatch(const Event& event) {
            dispatcher_.dispatch(event);
        }

        /**
         * @brief Create and dispatch event (forwarded to dispatcher)
         */
        template<typename EventType, typename... Args>
        void emit(Args&&... args) {
            dispatcher_.emit<EventType>(std::forward<Args>(args)...);
        }

    private:
        EventDispatcher dispatcher_;
        EventBus() = default;
    };

// === Convenience Functions ===

/**
 * @brief Emit event to global event bus
 */
    template<typename EventType, typename... Args>
    void emit_event(Args&&... args) {
        EventBus::instance().emit<EventType>(std::forward<Args>(args)...);
    }

/**
 * @brief Subscribe to events on global event bus
 */
    template<typename EventType>
    [[nodiscard]] std::unique_ptr<EventSubscription> subscribe_to_events(EventHandler<EventType> handler) {
        return EventBus::instance().subscribe<EventType>(std::move(handler));
    }

/**
 * @brief Subscribe to all events on global event bus
 */
    [[nodiscard]] inline std::unique_ptr<EventSubscription> subscribe_to_all_events(UniversalEventHandler handler) {
        return EventBus::instance().subscribe_all(std::move(handler));
    }

// === RAII Event Emitter Helper ===

/**
 * @brief RAII helper for emitting start/end events
 */
    template<typename StartEvent, typename EndEvent>
    class ScopedEventEmitter {
    public:
        template<typename... StartArgs>
        explicit ScopedEventEmitter(StartArgs&&... start_args)
                : start_time_(std::chrono::steady_clock::now()) {
            emit_event<StartEvent>(std::forward<StartArgs>(start_args)...);
        }

        ~ScopedEventEmitter() {
            auto duration = std::chrono::steady_clock::now() - start_time_;
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
            // Could add duration to end event if needed
            emit_event<EndEvent>();
        }

    private:
        std::chrono::steady_clock::time_point start_time_;
    };

} // namespace fem::core

#endif //BASE_OBSERVER_H
