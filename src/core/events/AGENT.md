# Core Events - AGENT.md

## Purpose
The `events/` layer provides a comprehensive event-driven architecture supporting synchronous and asynchronous event handling, publish-subscribe patterns, event queuing, filtering, and priority-based dispatch. It enables loose coupling between components through type-safe event communication with minimal overhead and maximum flexibility.

## Architecture Philosophy
- **Type safety**: Compile-time event type checking
- **Zero allocation**: Event pooling and in-place construction
- **Lock-free options**: For high-performance scenarios
- **Flexible dispatch**: Immediate, queued, or async delivery
- **Composable filters**: Chain event processors and transformers

## Files Overview

### Core Components
```cpp
event.hpp            // Base event class and traits
event_bus.hpp        // Central event dispatcher
event_queue.hpp      // Event queuing with priorities
event_handler.hpp    // Handler interface and wrappers
event_emitter.hpp    // Event source interface
event_types.hpp      // Common event type definitions
```

### Event Management
```cpp
subscription.hpp     // Subscription lifecycle management
event_filter.hpp     // Event filtering framework
event_router.hpp     // Event routing rules
event_aggregator.hpp // Event batching and aggregation
event_transformer.hpp // Event transformation pipeline
priority_queue.hpp   // Priority-based event queuing
```

### Signal/Slot System
```cpp
signal.hpp          // Qt-style signals
slot.hpp            // Slot connections
connection.hpp      // Connection management
signal_mapper.hpp   // Signal mapping and forwarding
auto_connection.hpp // Automatic disconnection
```

### Async Events
```cpp
async_event_bus.hpp // Asynchronous event dispatch
event_loop.hpp      // Event loop implementation
event_dispatcher.hpp // Thread-safe event dispatching
deferred_events.hpp // Deferred event execution
event_scheduler.hpp // Scheduled event delivery
```

### Advanced Features
```cpp
event_store.hpp     // Event persistence and replay
event_history.hpp   // Event history tracking
event_metrics.hpp   // Event system metrics
event_debugger.hpp  // Event flow debugging
weak_handler.hpp    // Weak reference handlers
```

### Utilities
```cpp
delegate.hpp        // Fast delegate implementation
function_traits.hpp // Function type introspection
type_erasure.hpp   // Type-erased event storage
event_macros.hpp   // Convenience macros
timer_events.hpp   // Timer-based events
```

## Detailed Component Specifications

### `event.hpp`
```cpp
// Base event class
class Event {
public:
    using EventId = uint64_t;
    using Timestamp = std::chrono::steady_clock::time_point;
    
    enum class Priority {
        Lowest = 0,
        Low = 25,
        Normal = 50,
        High = 75,
        Highest = 100,
        Immediate = 255
    };
    
    enum class PropagationStatus {
        Continue,     // Continue to next handler
        Stop,        // Stop propagation
        StopImmediate // Stop immediately (don't finish current handler)
    };
    
protected:
    EventId id_;
    Timestamp timestamp_;
    Priority priority_;
    bool consumed_ = false;
    PropagationStatus propagation_ = PropagationStatus::Continue;
    std::any user_data_;
    
public:
    Event(Priority priority = Priority::Normal)
        : id_(generate_id())
        , timestamp_(std::chrono::steady_clock::now())
        , priority_(priority) {}
    
    virtual ~Event() = default;
    
    // Event identification
    EventId id() const { return id_; }
    virtual std::type_index type() const = 0;
    virtual std::string name() const = 0;
    virtual std::size_t hash() const { return std::hash<EventId>{}(id_); }
    
    // Event metadata
    Timestamp timestamp() const { return timestamp_; }
    Priority priority() const { return priority_; }
    
    // Event control
    void consume() { consumed_ = true; }
    bool is_consumed() const { return consumed_; }
    
    void stop_propagation() { propagation_ = PropagationStatus::Stop; }
    void stop_immediate_propagation() { propagation_ = PropagationStatus::StopImmediate; }
    PropagationStatus propagation_status() const { return propagation_; }
    
    // User data attachment
    template<typename T>
    void set_user_data(T&& data) { user_data_ = std::forward<T>(data); }
    
    template<typename T>
    T get_user_data() const { return std::any_cast<T>(user_data_); }
    
    // Cloning support
    virtual std::unique_ptr<Event> clone() const = 0;
    
protected:
    static EventId generate_id() {
        static std::atomic<EventId> next_id{1};
        return next_id.fetch_add(1, std::memory_order_relaxed);
    }
};

// Typed event template
template<typename Derived>
class TypedEvent : public Event {
public:
    using Event::Event;
    
    std::type_index type() const override {
        return std::type_index(typeid(Derived));
    }
    
    std::string name() const override {
        return typeid(Derived).name();
    }
    
    std::unique_ptr<Event> clone() const override {
        return std::make_unique<Derived>(static_cast<const Derived&>(*this));
    }
    
    // Static type info
    static std::type_index static_type() {
        return std::type_index(typeid(Derived));
    }
};

// Event with payload
template<typename T>
class PayloadEvent : public TypedEvent<PayloadEvent<T>> {
    T payload_;
    
public:
    explicit PayloadEvent(T payload, Event::Priority priority = Event::Priority::Normal)
        : TypedEvent<PayloadEvent<T>>(priority)
        , payload_(std::move(payload)) {}
    
    const T& payload() const { return payload_; }
    T& payload() { return payload_; }
};

// Cancellable event
class CancellableEvent : public Event {
    bool cancelled_ = false;
    std::string cancellation_reason_;
    
public:
    void cancel(const std::string& reason = "") {
        cancelled_ = true;
        cancellation_reason_ = reason;
        stop_immediate_propagation();
    }
    
    bool is_cancelled() const { return cancelled_; }
    const std::string& cancellation_reason() const { return cancellation_reason_; }
};

// Event traits
template<typename T>
struct is_event : std::is_base_of<Event, T> {};

template<typename T>
inline constexpr bool is_event_v = is_event<T>::value;

// Event concepts (C++20)
template<typename T>
concept EventType = std::is_base_of_v<Event, T>;
```
**Why necessary**: Base event infrastructure, type safety, event metadata and control.
**Usage**: All event types inherit from this, provides common functionality.

### `event_bus.hpp`
```cpp
class EventBus {
public:
    using HandlerId = uint64_t;
    using HandlerFunc = std::function<void(const Event&)>;
    
    template<typename EventType>
    using TypedHandlerFunc = std::function<void(const EventType&)>;
    
private:
    struct HandlerInfo {
        HandlerId id;
        HandlerFunc handler;
        Event::Priority priority;
        std::weak_ptr<void> lifetime_tracker;
        bool once = false;
        std::optional<std::function<bool(const Event&)>> filter;
    };
    
    using HandlerList = std::vector<HandlerInfo>;
    std::unordered_map<std::type_index, HandlerList> handlers_;
    mutable std::shared_mutex handlers_mutex_;
    
    std::atomic<HandlerId> next_handler_id_{1};
    std::atomic<bool> enabled_{true};
    
    // Event metrics
    mutable std::atomic<uint64_t> events_published_{0};
    mutable std::atomic<uint64_t> events_handled_{0};
    
public:
    // Singleton access (optional)
    static EventBus& global() {
        static EventBus instance;
        return instance;
    }
    
    // Subscribe to events
    template<typename EventType>
    HandlerId subscribe(TypedHandlerFunc<EventType> handler,
                       Event::Priority priority = Event::Priority::Normal) {
        static_assert(is_event_v<EventType>, "Type must derive from Event");
        
        return subscribe_impl(
            EventType::static_type(),
            [handler = std::move(handler)](const Event& e) {
                handler(static_cast<const EventType&>(e));
            },
            priority
        );
    }
    
    // Subscribe with lifetime tracking
    template<typename EventType>
    HandlerId subscribe(TypedHandlerFunc<EventType> handler,
                       std::weak_ptr<void> lifetime,
                       Event::Priority priority = Event::Priority::Normal) {
        auto handler_id = subscribe<EventType>(std::move(handler), priority);
        
        std::unique_lock lock(handlers_mutex_);
        if (auto it = find_handler(EventType::static_type(), handler_id)) {
            it->lifetime_tracker = lifetime;
        }
        
        return handler_id;
    }
    
    // Subscribe for single event
    template<typename EventType>
    HandlerId subscribe_once(TypedHandlerFunc<EventType> handler,
                            Event::Priority priority = Event::Priority::Normal) {
        auto handler_id = subscribe<EventType>(std::move(handler), priority);
        
        std::unique_lock lock(handlers_mutex_);
        if (auto it = find_handler(EventType::static_type(), handler_id)) {
            it->once = true;
        }
        
        return handler_id;
    }
    
    // Subscribe with filter
    template<typename EventType>
    HandlerId subscribe_filtered(TypedHandlerFunc<EventType> handler,
                                 std::function<bool(const EventType&)> filter,
                                 Event::Priority priority = Event::Priority::Normal) {
        auto handler_id = subscribe<EventType>(std::move(handler), priority);
        
        std::unique_lock lock(handlers_mutex_);
        if (auto it = find_handler(EventType::static_type(), handler_id)) {
            it->filter = [filter](const Event& e) {
                return filter(static_cast<const EventType&>(e));
            };
        }
        
        return handler_id;
    }
    
    // Unsubscribe
    void unsubscribe(HandlerId handler_id) {
        std::unique_lock lock(handlers_mutex_);
        
        for (auto& [type, handlers] : handlers_) {
            handlers.erase(
                std::remove_if(handlers.begin(), handlers.end(),
                    [handler_id](const HandlerInfo& info) {
                        return info.id == handler_id;
                    }),
                handlers.end()
            );
        }
    }
    
    // Publish event
    template<typename EventType>
    void publish(const EventType& event) {
        static_assert(is_event_v<EventType>, "Type must derive from Event");
        
        if (!enabled_.load(std::memory_order_relaxed)) {
            return;
        }
        
        events_published_.fetch_add(1, std::memory_order_relaxed);
        
        dispatch_event(event);
    }
    
    // Emit event (convenience)
    template<typename EventType, typename... Args>
    void emit(Args&&... args) {
        publish(EventType(std::forward<Args>(args)...));
    }
    
    // Queue event for later dispatch
    template<typename EventType>
    void queue(const EventType& event);
    
    // Process queued events
    void process_queue();
    
    // Clear all handlers
    void clear() {
        std::unique_lock lock(handlers_mutex_);
        handlers_.clear();
    }
    
    // Enable/disable
    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }
    bool is_enabled() const { return enabled_; }
    
    // Statistics
    uint64_t events_published() const { return events_published_; }
    uint64_t events_handled() const { return events_handled_; }
    
    std::size_t handler_count() const {
        std::shared_lock lock(handlers_mutex_);
        std::size_t count = 0;
        for (const auto& [_, handlers] : handlers_) {
            count += handlers.size();
        }
        return count;
    }
    
private:
    HandlerId subscribe_impl(std::type_index type,
                            HandlerFunc handler,
                            Event::Priority priority) {
        std::unique_lock lock(handlers_mutex_);
        
        HandlerId id = next_handler_id_.fetch_add(1, std::memory_order_relaxed);
        
        auto& handler_list = handlers_[type];
        handler_list.push_back({id, std::move(handler), priority});
        
        // Sort by priority
        std::sort(handler_list.begin(), handler_list.end(),
            [](const HandlerInfo& a, const HandlerInfo& b) {
                return a.priority > b.priority;
            });
        
        return id;
    }
    
    void dispatch_event(const Event& event) {
        std::shared_lock lock(handlers_mutex_);
        
        auto it = handlers_.find(event.type());
        if (it == handlers_.end()) {
            return;
        }
        
        // Copy handlers to avoid issues with modifications during dispatch
        HandlerList handlers_copy = it->second;
        lock.unlock();
        
        for (auto& handler_info : handlers_copy) {
            // Check lifetime
            if (auto tracker = handler_info.lifetime_tracker.lock(); 
                !tracker && !handler_info.lifetime_tracker.expired()) {
                continue;  // Object destroyed
            }
            
            // Apply filter
            if (handler_info.filter && !handler_info.filter.value()(event)) {
                continue;
            }
            
            // Call handler
            handler_info.handler(event);
            events_handled_.fetch_add(1, std::memory_order_relaxed);
            
            // Check propagation
            if (event.propagation_status() == Event::PropagationStatus::StopImmediate) {
                break;
            }
            
            // Remove if once
            if (handler_info.once) {
                unsubscribe(handler_info.id);
            }
            
            if (event.propagation_status() == Event::PropagationStatus::Stop) {
                break;
            }
        }
    }
    
    HandlerList::iterator find_handler(std::type_index type, HandlerId id) {
        auto& handlers = handlers_[type];
        return std::find_if(handlers.begin(), handlers.end(),
            [id](const HandlerInfo& info) { return info.id == id; });
    }
};

// Scoped subscription
class ScopedSubscription {
    EventBus* bus_;
    EventBus::HandlerId handler_id_;
    
public:
    ScopedSubscription(EventBus* bus, EventBus::HandlerId id)
        : bus_(bus), handler_id_(id) {}
    
    ~ScopedSubscription() {
        if (bus_ && handler_id_ != 0) {
            bus_->unsubscribe(handler_id_);
        }
    }
    
    // Move-only
    ScopedSubscription(ScopedSubscription&& other) noexcept
        : bus_(other.bus_), handler_id_(other.handler_id_) {
        other.bus_ = nullptr;
        other.handler_id_ = 0;
    }
    
    ScopedSubscription& operator=(ScopedSubscription&& other) noexcept {
        if (this != &other) {
            if (bus_ && handler_id_ != 0) {
                bus_->unsubscribe(handler_id_);
            }
            bus_ = other.bus_;
            handler_id_ = other.handler_id_;
            other.bus_ = nullptr;
            other.handler_id_ = 0;
        }
        return *this;
    }
    
    void release() {
        bus_ = nullptr;
        handler_id_ = 0;
    }
};
```
**Why necessary**: Central event routing, publish-subscribe pattern, decoupled communication.
**Usage**: Application-wide event system, module communication, UI updates.

### `event_queue.hpp`
```cpp
template<typename EventType = Event>
class EventQueue {
    struct QueuedEvent {
        std::unique_ptr<EventType> event;
        std::chrono::steady_clock::time_point enqueue_time;
        
        bool operator<(const QueuedEvent& other) const {
            // Higher priority first, then earlier timestamp
            if (event->priority() != other.event->priority()) {
                return event->priority() < other.event->priority();
            }
            return enqueue_time > other.enqueue_time;
        }
    };
    
    std::priority_queue<QueuedEvent> queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    std::atomic<std::size_t> max_size_{10000};
    std::atomic<bool> blocking_{false};
    
    // Statistics
    std::atomic<uint64_t> events_queued_{0};
    std::atomic<uint64_t> events_processed_{0};
    std::atomic<uint64_t> events_dropped_{0};
    
public:
    EventQueue(std::size_t max_size = 10000, bool blocking = false)
        : max_size_(max_size), blocking_(blocking) {}
    
    // Queue event
    bool push(std::unique_ptr<EventType> event) {
        std::unique_lock lock(queue_mutex_);
        
        if (!blocking_ && queue_.size() >= max_size_) {
            events_dropped_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        
        if (blocking_) {
            queue_cv_.wait(lock, [this] {
                return queue_.size() < max_size_;
            });
        }
        
        queue_.push({std::move(event), std::chrono::steady_clock::now()});
        events_queued_.fetch_add(1, std::memory_order_relaxed);
        
        queue_cv_.notify_one();
        return true;
    }
    
    // Queue event with timeout
    bool push_for(std::unique_ptr<EventType> event, std::chrono::milliseconds timeout) {
        std::unique_lock lock(queue_mutex_);
        
        if (!queue_cv_.wait_for(lock, timeout, [this] {
            return queue_.size() < max_size_;
        })) {
            events_dropped_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        
        queue_.push({std::move(event), std::chrono::steady_clock::now()});
        events_queued_.fetch_add(1, std::memory_order_relaxed);
        
        queue_cv_.notify_one();
        return true;
    }
    
    // Dequeue event
    std::unique_ptr<EventType> pop() {
        std::unique_lock lock(queue_mutex_);
        
        if (queue_.empty()) {
            return nullptr;
        }
        
        auto event = std::move(const_cast<QueuedEvent&>(queue_.top()).event);
        queue_.pop();
        events_processed_.fetch_add(1, std::memory_order_relaxed);
        
        queue_cv_.notify_all();
        return event;
    }
    
    // Blocking dequeue
    std::unique_ptr<EventType> pop_wait() {
        std::unique_lock lock(queue_mutex_);
        
        queue_cv_.wait(lock, [this] { return !queue_.empty(); });
        
        auto event = std::move(const_cast<QueuedEvent&>(queue_.top()).event);
        queue_.pop();
        events_processed_.fetch_add(1, std::memory_order_relaxed);
        
        queue_cv_.notify_all();
        return event;
    }
    
    // Dequeue with timeout
    std::unique_ptr<EventType> pop_wait_for(std::chrono::milliseconds timeout) {
        std::unique_lock lock(queue_mutex_);
        
        if (!queue_cv_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) {
            return nullptr;
        }
        
        auto event = std::move(const_cast<QueuedEvent&>(queue_.top()).event);
        queue_.pop();
        events_processed_.fetch_add(1, std::memory_order_relaxed);
        
        queue_cv_.notify_all();
        return event;
    }
    
    // Process events with handler
    template<typename Handler>
    std::size_t process(Handler&& handler, std::size_t max_events = std::numeric_limits<std::size_t>::max()) {
        std::size_t processed = 0;
        
        while (processed < max_events) {
            auto event = pop();
            if (!event) {
                break;
            }
            
            handler(*event);
            ++processed;
        }
        
        return processed;
    }
    
    // Process all events
    template<typename Handler>
    void process_all(Handler&& handler) {
        while (!empty()) {
            if (auto event = pop()) {
                handler(*event);
            }
        }
    }
    
    // Queue state
    std::size_t size() const {
        std::lock_guard lock(queue_mutex_);
        return queue_.size();
    }
    
    bool empty() const {
        std::lock_guard lock(queue_mutex_);
        return queue_.empty();
    }
    
    void clear() {
        std::lock_guard lock(queue_mutex_);
        while (!queue_.empty()) {
            queue_.pop();
        }
        queue_cv_.notify_all();
    }
    
    // Configuration
    void set_max_size(std::size_t size) { max_size_ = size; }
    std::size_t max_size() const { return max_size_; }
    
    void set_blocking(bool blocking) { blocking_ = blocking; }
    bool is_blocking() const { return blocking_; }
    
    // Statistics
    uint64_t events_queued() const { return events_queued_; }
    uint64_t events_processed() const { return events_processed_; }
    uint64_t events_dropped() const { return events_dropped_; }
};

// Lock-free event queue using ring buffer
template<typename EventType, std::size_t Size>
class LockFreeEventQueue {
    struct Slot {
        std::atomic<bool> occupied{false};
        alignas(EventType) std::byte storage[sizeof(EventType)];
    };
    
    std::array<Slot, Size> buffer_;
    alignas(64) std::atomic<std::size_t> write_index_{0};
    alignas(64) std::atomic<std::size_t> read_index_{0};
    
public:
    bool push(const EventType& event) {
        std::size_t write_idx = write_index_.load(std::memory_order_relaxed);
        std::size_t next_idx = (write_idx + 1) % Size;
        
        if (next_idx == read_index_.load(std::memory_order_acquire)) {
            return false;  // Queue full
        }
        
        new (buffer_[write_idx].storage) EventType(event);
        buffer_[write_idx].occupied.store(true, std::memory_order_release);
        write_index_.store(next_idx, std::memory_order_release);
        
        return true;
    }
    
    std::optional<EventType> pop() {
        std::size_t read_idx = read_index_.load(std::memory_order_relaxed);
        
        if (read_idx == write_index_.load(std::memory_order_acquire)) {
            return std::nullopt;  // Queue empty
        }
        
        while (!buffer_[read_idx].occupied.load(std::memory_order_acquire)) {
            // Wait for write to complete
            std::this_thread::yield();
        }
        
        EventType* event = reinterpret_cast<EventType*>(buffer_[read_idx].storage);
        EventType result = std::move(*event);
        event->~EventType();
        
        buffer_[read_idx].occupied.store(false, std::memory_order_release);
        read_index_.store((read_idx + 1) % Size, std::memory_order_release);
        
        return result;
    }
    
    bool empty() const {
        return read_index_.load(std::memory_order_relaxed) == 
               write_index_.load(std::memory_order_relaxed);
    }
};
```
**Why necessary**: Event buffering, priority handling, async processing, backpressure management.
**Usage**: Worker threads, event loops, deferred processing.

### `signal.hpp`
```cpp
template<typename... Args>
class Signal {
public:
    using Slot = std::function<void(Args...)>;
    using ConnectionId = uint64_t;
    
    class Connection {
        friend class Signal;
        
        Signal* signal_ = nullptr;
        ConnectionId id_ = 0;
        bool blocked_ = false;
        
    public:
        Connection() = default;
        Connection(Signal* signal, ConnectionId id) : signal_(signal), id_(id) {}
        
        void disconnect() {
            if (signal_) {
                signal_->disconnect(id_);
                signal_ = nullptr;
                id_ = 0;
            }
        }
        
        void block() { blocked_ = true; }
        void unblock() { blocked_ = false; }
        bool is_blocked() const { return blocked_; }
        
        bool connected() const { return signal_ != nullptr && id_ != 0; }
        
        ConnectionId id() const { return id_; }
    };
    
private:
    struct SlotInfo {
        ConnectionId id;
        Slot slot;
        std::weak_ptr<void> tracker;
        bool once = false;
        bool blocked = false;
        int priority = 0;
    };
    
    mutable std::shared_mutex mutex_;
    std::vector<SlotInfo> slots_;
    std::atomic<ConnectionId> next_connection_id_{1};
    std::atomic<bool> emitting_{false};
    
public:
    Signal() = default;
    
    // Connect slot
    Connection connect(Slot slot, int priority = 0) {
        std::unique_lock lock(mutex_);
        
        ConnectionId id = next_connection_id_.fetch_add(1);
        slots_.push_back({id, std::move(slot), {}, false, false, priority});
        
        // Sort by priority
        std::sort(slots_.begin(), slots_.end(),
            [](const SlotInfo& a, const SlotInfo& b) {
                return a.priority > b.priority;
            });
        
        return Connection(this, id);
    }
    
    // Connect with automatic disconnection
    Connection connect_scoped(Slot slot, std::weak_ptr<void> tracker, int priority = 0) {
        std::unique_lock lock(mutex_);
        
        ConnectionId id = next_connection_id_.fetch_add(1);
        slots_.push_back({id, std::move(slot), tracker, false, false, priority});
        
        std::sort(slots_.begin(), slots_.end(),
            [](const SlotInfo& a, const SlotInfo& b) {
                return a.priority > b.priority;
            });
        
        return Connection(this, id);
    }
    
    // Connect for single emission
    Connection connect_once(Slot slot, int priority = 0) {
        auto conn = connect(std::move(slot), priority);
        
        std::unique_lock lock(mutex_);
        if (auto it = find_slot(conn.id()); it != slots_.end()) {
            it->once = true;
        }
        
        return conn;
    }
    
    // Emit signal
    void emit(Args... args) {
        // Prevent nested emissions
        bool expected = false;
        if (!emitting_.compare_exchange_strong(expected, true)) {
            // Queue for later emission or handle nested case
            return;
        }
        
        // Copy slots to avoid modification during emission
        std::vector<SlotInfo> slots_copy;
        {
            std::shared_lock lock(mutex_);
            slots_copy = slots_;
        }
        
        // Emit to all slots
        std::vector<ConnectionId> to_remove;
        
        for (auto& slot_info : slots_copy) {
            // Check lifetime
            if (auto tracker = slot_info.tracker.lock(); 
                !tracker && !slot_info.tracker.expired()) {
                to_remove.push_back(slot_info.id);
                continue;
            }
            
            // Check if blocked
            if (slot_info.blocked) {
                continue;
            }
            
            // Call slot
            slot_info.slot(args...);
            
            // Remove if once
            if (slot_info.once) {
                to_remove.push_back(slot_info.id);
            }
        }
        
        // Remove dead connections
        if (!to_remove.empty()) {
            std::unique_lock lock(mutex_);
            for (auto id : to_remove) {
                disconnect_impl(id);
            }
        }
        
        emitting_ = false;
    }
    
    // Operator() as emit
    void operator()(Args... args) {
        emit(std::forward<Args>(args)...);
    }
    
    // Disconnect
    void disconnect(ConnectionId id) {
        std::unique_lock lock(mutex_);
        disconnect_impl(id);
    }
    
    void disconnect_all() {
        std::unique_lock lock(mutex_);
        slots_.clear();
    }
    
    // Connection management
    std::size_t connection_count() const {
        std::shared_lock lock(mutex_);
        return slots_.size();
    }
    
    bool has_connections() const {
        std::shared_lock lock(mutex_);
        return !slots_.empty();
    }
    
    // Block/unblock connections
    void block_connection(ConnectionId id) {
        std::unique_lock lock(mutex_);
        if (auto it = find_slot(id); it != slots_.end()) {
            it->blocked = true;
        }
    }
    
    void unblock_connection(ConnectionId id) {
        std::unique_lock lock(mutex_);
        if (auto it = find_slot(id); it != slots_.end()) {
            it->blocked = false;
        }
    }
    
private:
    void disconnect_impl(ConnectionId id) {
        slots_.erase(
            std::remove_if(slots_.begin(), slots_.end(),
                [id](const SlotInfo& info) { return info.id == id; }),
            slots_.end()
        );
    }
    
    typename std::vector<SlotInfo>::iterator find_slot(ConnectionId id) {
        return std::find_if(slots_.begin(), slots_.end(),
            [id](const SlotInfo& info) { return info.id == id; });
    }
};

// Automatic connection management
template<typename... Args>
class AutoConnection {
    using Connection = typename Signal<Args...>::Connection;
    Connection connection_;
    
public:
    AutoConnection() = default;
    
    AutoConnection(Connection conn) : connection_(std::move(conn)) {}
    
    ~AutoConnection() {
        if (connection_.connected()) {
            connection_.disconnect();
        }
    }
    
    // Move-only
    AutoConnection(AutoConnection&& other) noexcept
        : connection_(std::move(other.connection_)) {}
    
    AutoConnection& operator=(AutoConnection&& other) noexcept {
        if (this != &other) {
            if (connection_.connected()) {
                connection_.disconnect();
            }
            connection_ = std::move(other.connection_);
        }
        return *this;
    }
    
    Connection& get() { return connection_; }
    const Connection& get() const { return connection_; }
    
    void reset(Connection conn = Connection()) {
        if (connection_.connected()) {
            connection_.disconnect();
        }
        connection_ = std::move(conn);
    }
};
```
**Why necessary**: Qt-style signal/slot connections, automatic disconnection, type-safe callbacks.
**Usage**: UI events, property changes, observable patterns.

### `event_loop.hpp`
```cpp
class EventLoop {
    using Task = std::function<void()>;
    using TimerId = uint64_t;
    
    struct TimerInfo {
        TimerId id;
        std::chrono::steady_clock::time_point next_fire;
        std::chrono::milliseconds interval;
        std::function<void()> callback;
        bool repeating;
        
        bool operator>(const TimerInfo& other) const {
            return next_fire > other.next_fire;
        }
    };
    
    EventQueue<Event> event_queue_;
    std::queue<Task> task_queue_;
    std::priority_queue<TimerInfo, std::vector<TimerInfo>, std::greater<>> timer_queue_;
    
    mutable std::mutex queues_mutex_;
    std::condition_variable wake_cv_;
    
    std::atomic<bool> running_{false};
    std::atomic<bool> quit_requested_{false};
    std::thread::id thread_id_;
    
    std::atomic<TimerId> next_timer_id_{1};
    
public:
    EventLoop() = default;
    
    // Run event loop
    void run() {
        running_ = true;
        quit_requested_ = false;
        thread_id_ = std::this_thread::get_id();
        
        while (!quit_requested_) {
            process_one_iteration();
        }
        
        running_ = false;
    }
    
    // Run for duration
    void run_for(std::chrono::milliseconds duration) {
        auto end_time = std::chrono::steady_clock::now() + duration;
        
        running_ = true;
        quit_requested_ = false;
        thread_id_ = std::this_thread::get_id();
        
        while (!quit_requested_ && std::chrono::steady_clock::now() < end_time) {
            auto remaining = end_time - std::chrono::steady_clock::now();
            process_one_iteration(remaining);
        }
        
        running_ = false;
    }
    
    // Process single iteration
    void process_one() {
        process_one_iteration(std::chrono::milliseconds(0));
    }
    
    // Queue event
    void post_event(std::unique_ptr<Event> event) {
        event_queue_.push(std::move(event));
        wake_cv_.notify_one();
    }
    
    // Queue task
    void post_task(Task task) {
        {
            std::lock_guard lock(queues_mutex_);
            task_queue_.push(std::move(task));
        }
        wake_cv_.notify_one();
    }
    
    // Timer management
    TimerId set_timeout(std::function<void()> callback, std::chrono::milliseconds delay) {
        return add_timer(std::move(callback), delay, false);
    }
    
    TimerId set_interval(std::function<void()> callback, std::chrono::milliseconds interval) {
        return add_timer(std::move(callback), interval, true);
    }
    
    void clear_timer(TimerId id) {
        std::lock_guard lock(queues_mutex_);
        // Mark timer as cancelled (would need additional tracking)
    }
    
    // Defer execution to next iteration
    void defer(Task task) {
        post_task(std::move(task));
    }
    
    // Quit event loop
    void quit() {
        quit_requested_ = true;
        wake_cv_.notify_one();
    }
    
    // State queries
    bool is_running() const { return running_; }
    std::thread::id thread_id() const { return thread_id_; }
    
    bool in_event_loop() const {
        return std::this_thread::get_id() == thread_id_;
    }
    
private:
    void process_one_iteration(std::chrono::milliseconds max_wait = std::chrono::milliseconds::max()) {
        auto now = std::chrono::steady_clock::now();
        
        // Process timers
        process_timers(now);
        
        // Process events
        process_events();
        
        // Process tasks
        process_tasks();
        
        // Calculate wait time
        auto next_timer = get_next_timer_time();
        auto wait_time = max_wait;
        
        if (next_timer) {
            auto timer_wait = std::chrono::duration_cast<std::chrono::milliseconds>(
                *next_timer - std::chrono::steady_clock::now()
            );
            wait_time = std::min(wait_time, std::max(timer_wait, std::chrono::milliseconds(0)));
        }
        
        // Wait for new events/tasks/timers
        if (wait_time > std::chrono::milliseconds(0)) {
            std::unique_lock lock(queues_mutex_);
            wake_cv_.wait_for(lock, wait_time, [this] {
                return quit_requested_ || !event_queue_.empty() || !task_queue_.empty();
            });
        }
    }
    
    void process_timers(std::chrono::steady_clock::time_point now) {
        std::lock_guard lock(queues_mutex_);
        
        while (!timer_queue_.empty() && timer_queue_.top().next_fire <= now) {
            TimerInfo timer = timer_queue_.top();
            timer_queue_.pop();
            
            // Execute callback
            timer.callback();
            
            // Re-queue if repeating
            if (timer.repeating) {
                timer.next_fire = now + timer.interval;
                timer_queue_.push(timer);
            }
        }
    }
    
    void process_events() {
        // Process all pending events
        event_queue_.process_all([](const Event& event) {
            // Dispatch to global event bus or custom handler
            EventBus::global().publish(event);
        });
    }
    
    void process_tasks() {
        std::queue<Task> tasks;
        {
            std::lock_guard lock(queues_mutex_);
            tasks.swap(task_queue_);
        }
        
        while (!tasks.empty()) {
            tasks.front()();
            tasks.pop();
        }
    }
    
    TimerId add_timer(std::function<void()> callback,
                     std::chrono::milliseconds delay,
                     bool repeating) {
        std::lock_guard lock(queues_mutex_);
        
        TimerId id = next_timer_id_.fetch_add(1);
        auto next_fire = std::chrono::steady_clock::now() + delay;
        
        timer_queue_.push({id, next_fire, delay, std::move(callback), repeating});
        wake_cv_.notify_one();
        
        return id;
    }
    
    std::optional<std::chrono::steady_clock::time_point> get_next_timer_time() const {
        std::lock_guard lock(queues_mutex_);
        
        if (timer_queue_.empty()) {
            return std::nullopt;
        }
        
        return timer_queue_.top().next_fire;
    }
};

// Global event loop for current thread
thread_local EventLoop* current_event_loop = nullptr;

EventLoop* get_current_event_loop() {
    return current_event_loop;
}

void set_current_event_loop(EventLoop* loop) {
    current_event_loop = loop;
}
```
**Why necessary**: Event-driven programming, timer management, task scheduling, async execution.
**Usage**: UI main loops, server event loops, async programming.

### `event_macros.hpp`
```cpp
// Event definition macros
#define DEFINE_EVENT(EventName, ...) \
    class EventName : public TypedEvent<EventName> { \
    public: \
        __VA_ARGS__ \
    }

#define DEFINE_SIMPLE_EVENT(EventName) \
    class EventName : public TypedEvent<EventName> { \
    public: \
        using TypedEvent::TypedEvent; \
    }

// Event publishing macros
#define PUBLISH_EVENT(bus, EventType, ...) \
    (bus).publish(EventType(__VA_ARGS__))

#define EMIT_EVENT(bus, EventType, ...) \
    (bus).emit<EventType>(__VA_ARGS__)

#define PUBLISH_GLOBAL(EventType, ...) \
    EventBus::global().publish(EventType(__VA_ARGS__))

// Subscription macros
#define SUBSCRIBE(bus, EventType, handler) \
    (bus).subscribe<EventType>(handler)

#define SUBSCRIBE_METHOD(bus, EventType, object, method) \
    (bus).subscribe<EventType>( \
        [&object](const EventType& e) { (object).method(e); } \
    )

#define SUBSCRIBE_GLOBAL(EventType, handler) \
    EventBus::global().subscribe<EventType>(handler)

// Scoped subscription
#define AUTO_SUBSCRIBE(var, bus, EventType, handler) \
    ScopedSubscription var(&(bus), (bus).subscribe<EventType>(handler))

// Signal/slot macros
#define SIGNAL(name, ...) \
    Signal<__VA_ARGS__> name

#define CONNECT(signal, slot) \
    (signal).connect(slot)

#define CONNECT_METHOD(signal, object, method) \
    (signal).connect([&object](auto&&... args) { \
        (object).method(std::forward<decltype(args)>(args)...); \
    })

#define EMIT(signal, ...) \
    (signal).emit(__VA_ARGS__)

// Event handler macros
#define EVENT_HANDLER(EventType) \
    void handle_##EventType(const EventType& event)

#define BEGIN_EVENT_MAP(ClassName) \
    void handle_event(const Event& event) override { \
        using _class_type = ClassName;

#define HANDLE_EVENT(EventType, method) \
    if (event.type() == EventType::static_type()) { \
        method(static_cast<const EventType&>(event)); \
        return; \
    }

#define END_EVENT_MAP() \
    }

// Timer macros
#define SET_TIMEOUT(loop, callback, delay_ms) \
    (loop).set_timeout(callback, std::chrono::milliseconds(delay_ms))

#define SET_INTERVAL(loop, callback, interval_ms) \
    (loop).set_interval(callback, std::chrono::milliseconds(interval_ms))

// Async event macros
#define POST_EVENT(loop, EventType, ...) \
    (loop).post_event(std::make_unique<EventType>(__VA_ARGS__))

#define POST_TASK(loop, task) \
    (loop).post_task(task)

#define DEFER(loop, code) \
    (loop).defer([&]() { code; })
```
**Why necessary**: Reduce boilerplate, consistent event handling, convenient macros.
**Usage**: Throughout application for event definition and handling.

## Event System Patterns

### Basic Event Publishing
```cpp
// Define custom event
DEFINE_EVENT(DataUpdatedEvent,
    std::string data_id;
    std::any new_value;
    std::any old_value;
    
    DataUpdatedEvent(std::string id, std::any new_val, std::any old_val)
        : data_id(std::move(id))
        , new_value(std::move(new_val))
        , old_value(std::move(old_val)) {}
);

// Subscribe to event
auto subscription = EventBus::global().subscribe<DataUpdatedEvent>(
    [](const DataUpdatedEvent& e) {
        std::cout << "Data " << e.data_id << " updated\n";
    }
);

// Publish event
EventBus::global().publish(
    DataUpdatedEvent("user_123", 42, 41)
);
```

### Signal/Slot Pattern
```cpp
class Model {
public:
    Signal<int> value_changed;
    
private:
    int value_ = 0;
    
public:
    void set_value(int v) {
        if (value_ != v) {
            value_ = v;
            value_changed.emit(value_);
        }
    }
};

class View {
public:
    void on_value_changed(int new_value) {
        std::cout << "Value changed to: " << new_value << "\n";
    }
};

// Connect
Model model;
View view;
auto connection = model.value_changed.connect(
    [&view](int v) { view.on_value_changed(v); }
);
```

### Event Loop Usage
```cpp
class Application {
    EventLoop main_loop_;
    
public:
    void run() {
        // Set up timer
        main_loop_.set_interval([]() {
            std::cout << "Timer tick\n";
        }, std::chrono::seconds(1));
        
        // Post initial task
        main_loop_.post_task([]() {
            std::cout << "Application started\n";
        });
        
        // Run event loop
        main_loop_.run();
    }
    
    void handle_user_input(const std::string& input) {
        // Post event to main loop
        main_loop_.post_event(
            std::make_unique<UserInputEvent>(input)
        );
    }
};
```

### Priority-Based Event Handling
```cpp
// Subscribe with different priorities
EventBus& bus = EventBus::global();

// High priority - processes first
bus.subscribe<ErrorEvent>(
    [](const ErrorEvent& e) {
        log_error(e);
    },
    Event::Priority::High
);

// Normal priority
bus.subscribe<ErrorEvent>(
    [](const ErrorEvent& e) {
        update_ui_error_count();
    },
    Event::Priority::Normal
);

// Low priority - processes last
bus.subscribe<ErrorEvent>(
    [](const ErrorEvent& e) {
        send_telemetry(e);
    },
    Event::Priority::Low
);
```

### Event Filtering and Transformation
```cpp
// Subscribe with filter
bus.subscribe_filtered<MouseEvent>(
    [](const MouseEvent& e) {
        handle_click(e);
    },
    [](const MouseEvent& e) {
        return e.button() == MouseButton::Left;
    }
);

// Event aggregation
class EventAggregator {
    std::vector<Event> buffered_events_;
    std::chrono::milliseconds flush_interval_{100};
    
public:
    void aggregate(const Event& e) {
        buffered_events_.push_back(e.clone());
        
        if (should_flush()) {
            flush();
        }
    }
    
    void flush() {
        process_batch(buffered_events_);
        buffered_events_.clear();
    }
};
```

## Performance Considerations

- **Event pooling**: ~10-20ns for pooled event allocation
- **Handler dispatch**: ~50-100ns per handler call
- **Lock-free queue**: ~100ns push/pop operations
- **Signal emission**: ~30-50ns per slot call
- **Event cloning**: Varies by event size and complexity

## Testing Strategy

- **Handler execution**: Verify correct handler calling
- **Priority ordering**: Test priority-based dispatch
- **Thread safety**: Concurrent publishing and subscribing
- **Memory leaks**: Track handler lifecycle
- **Performance**: Benchmark event throughput

## Usage Guidelines

1. **Choose appropriate patterns**: EventBus for global, Signal for local
2. **Use priorities wisely**: Reserve high priority for critical handlers
3. **Avoid blocking**: Keep handlers fast or defer work
4. **Clean up subscriptions**: Use RAII or weak references
5. **Consider batching**: Aggregate high-frequency events

## Anti-patterns to Avoid

- Synchronous long-running handlers
- Circular event dependencies
- Modifying event bus during dispatch
- Forgetting to unsubscribe
- Excessive event granularity

## Dependencies
- `base/` - For Object, Observer patterns
- `concurrency/` - For thread safety
- `memory/` - For event pooling
- Standard library (C++20)

## Future Enhancements
- Event sourcing support
- Distributed events
- Event replay and debugging
- Pattern matching on events
- Compile-time event registration