/**
 * @file test_observer.cpp
 * @brief Comprehensive unit tests for the Observer pattern implementations
 *
 * Tests cover:
 * - Event base class and TypedEvent CRTP helper
 * - Observer<T> interface and Subject<T> implementation
 * - EventDispatcher with typed and universal handlers
 * - EventSubscription RAII management
 * - EventBus global singleton functionality
 * - ScopedEventEmitter for start/end events
 * - Thread safety for concurrent observers and dispatching
 * - Performance characteristics and memory management
 * - Error handling and edge cases
 * - Integration scenarios and complex workflows
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <future>
#include <barrier>

#include "../../../base/observer.h"

using namespace fem::core::base;
using namespace testing;

// ============================================================================
// Test Event Classes
// ============================================================================

// Basic test event
class TestEvent : public TypedEvent<TestEvent> {
public:
    explicit TestEvent(std::string message = "test", int value = 42)
        : TypedEvent<TestEvent>("TestEvent"), message_(std::move(message)), value_(value) {}

    const std::string& get_message() const { return message_; }
    int get_value() const { return value_; }

    std::string to_string() const override {
        return Event::to_string() + ", message=" + message_ + ", value=" + std::to_string(value_);
    }

private:
    std::string message_;
    int value_;
};

// Another event type for testing different handlers
class ProgressEvent : public TypedEvent<ProgressEvent> {
public:
    explicit ProgressEvent(double percentage = 0.0, std::string operation = "")
        : TypedEvent<ProgressEvent>("ProgressEvent"), percentage_(percentage), operation_(std::move(operation)) {}

    double get_percentage() const { return percentage_; }
    const std::string& get_operation() const { return operation_; }

private:
    double percentage_;
    std::string operation_;
};

// Start/End events for scoped testing
class OperationStartEvent : public TypedEvent<OperationStartEvent> {
public:
    explicit OperationStartEvent(std::string operation_name = "test_operation")
        : TypedEvent<OperationStartEvent>("OperationStartEvent"), operation_name_(std::move(operation_name)) {}

    const std::string& get_operation_name() const { return operation_name_; }

private:
    std::string operation_name_;
};

class OperationEndEvent : public TypedEvent<OperationEndEvent> {
public:
    explicit OperationEndEvent(std::string operation_name = "test_operation")
        : TypedEvent<OperationEndEvent>("OperationEndEvent"), operation_name_(std::move(operation_name)) {}

    const std::string& get_operation_name() const { return operation_name_; }

private:
    std::string operation_name_;
};

// Error event for testing error scenarios
class ErrorEvent : public TypedEvent<ErrorEvent> {
public:
    explicit ErrorEvent(std::string error_message = "", int error_code = 0)
        : TypedEvent<ErrorEvent>("ErrorEvent"), error_message_(std::move(error_message)), error_code_(error_code) {}

    const std::string& get_error_message() const { return error_message_; }
    int get_error_code() const { return error_code_; }

private:
    std::string error_message_;
    int error_code_;
};

// ============================================================================
// Test Observer Classes
// ============================================================================

// Basic test observer for TestEvent
class TestObserver : public Observer<TestEvent> {
public:
    explicit TestObserver(std::string name = "TestObserver") : name_(std::move(name)) {}

    void on_event(const TestEvent& event) override {
        events_received_.push_back({event.get_message(), event.get_value()});
        last_event_timestamp_ = event.get_timestamp();
        total_events_++;
    }

    void on_any_event(const Event& event) override {
        any_events_received_++;
        last_any_event_type_ = event.get_type_name();
    }

    // Test accessors
    const std::string& get_name() const { return name_; }
    size_t get_event_count() const { return events_received_.size(); }
    size_t get_any_event_count() const { return any_events_received_; }
    size_t get_total_event_count() const { return total_events_; }
    const std::string& get_last_any_event_type() const { return last_any_event_type_; }
    auto get_last_event_timestamp() const { return last_event_timestamp_; }

    const auto& get_received_events() const { return events_received_; }

    void reset() {
        events_received_.clear();
        any_events_received_ = 0;
        total_events_ = 0;
        last_any_event_type_.clear();
    }

private:
    std::string name_;
    std::vector<std::pair<std::string, int>> events_received_;
    size_t any_events_received_ = 0;
    size_t total_events_ = 0;
    std::string last_any_event_type_;
    std::chrono::steady_clock::time_point last_event_timestamp_;
};

// Progress observer for ProgressEvent
class ProgressObserver : public Observer<ProgressEvent> {
public:
    void on_event(const ProgressEvent& event) override {
        progress_updates_.push_back({event.get_percentage(), event.get_operation()});
        if (event.get_percentage() >= 100.0) {
            completed_operations_++;
        }
    }

    const auto& get_progress_updates() const { return progress_updates_; }
    size_t get_completed_operations() const { return completed_operations_; }
    void reset() { progress_updates_.clear(); completed_operations_ = 0; }

private:
    std::vector<std::pair<double, std::string>> progress_updates_;
    size_t completed_operations_ = 0;
};

// Thread-safe observer for concurrency testing
class ThreadSafeObserver : public Observer<TestEvent> {
public:
    void on_event(const TestEvent& event) override {
        std::lock_guard lock(mutex_);
        event_count_++;
        thread_ids_.insert(std::this_thread::get_id());
        messages_.push_back(event.get_message());
    }

    size_t get_event_count() const {
        std::lock_guard lock(mutex_);
        return event_count_;
    }

    size_t get_unique_thread_count() const {
        std::lock_guard lock(mutex_);
        return thread_ids_.size();
    }

    std::vector<std::string> get_messages() const {
        std::lock_guard lock(mutex_);
        return messages_;
    }

private:
    mutable std::mutex mutex_;
    size_t event_count_ = 0;
    std::unordered_set<std::thread::id> thread_ids_;
    std::vector<std::string> messages_;
};

// Mock observer for testing
class MockTestObserver : public Observer<TestEvent> {
public:
    MOCK_METHOD(void, on_event, (const TestEvent& event), (override));
    MOCK_METHOD(void, on_any_event, (const Event& event), (override));
};

// ============================================================================
// Test Fixtures
// ============================================================================

class ObserverTest : public ::testing::Test {
protected:
    void SetUp() override {
        observer1 = std::make_shared<TestObserver>("Observer1");
        observer2 = std::make_shared<TestObserver>("Observer2");
        progress_observer = std::make_shared<ProgressObserver>();
        thread_safe_observer = std::make_shared<ThreadSafeObserver>();
    }

    void TearDown() override {
        // Reset observers
        observer1->reset();
        observer2->reset();
        progress_observer->reset();
    }

    std::shared_ptr<TestObserver> observer1;
    std::shared_ptr<TestObserver> observer2;
    std::shared_ptr<ProgressObserver> progress_observer;
    std::shared_ptr<ThreadSafeObserver> thread_safe_observer;
};

class EventDispatcherTest : public ::testing::Test {
protected:
    void SetUp() override {
        dispatcher = std::make_unique<EventDispatcher>();
    }

    void TearDown() override {
        dispatcher->clear();
    }

    std::unique_ptr<EventDispatcher> dispatcher;
};

// ============================================================================
// Event Tests
// ============================================================================

TEST(EventBasicTest, EventCreationAndProperties) {
    TestEvent event("test_message", 123);

    EXPECT_EQ(event.get_type_name(), "TestEvent");
    EXPECT_EQ(event.get_message(), "test_message");
    EXPECT_EQ(event.get_value(), 123);
    EXPECT_EQ(event.get_type(), std::type_index(typeid(TestEvent)));
    EXPECT_FALSE(event.is_handled());
    EXPECT_TRUE(event.get_source().empty());

    // Timestamp should be recent
    auto now = std::chrono::steady_clock::now();
    auto event_time = event.get_timestamp();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(now - event_time);
    EXPECT_LT(diff.count(), 100); // Should be within 100ms
}

TEST(EventBasicTest, EventSourceAndHandling) {
    TestEvent event("test", 42);

    event.set_source("TestSource");
    EXPECT_EQ(event.get_source(), "TestSource");

    EXPECT_FALSE(event.is_handled());
    event.set_handled(true);
    EXPECT_TRUE(event.is_handled());

    event.set_handled(false);
    EXPECT_FALSE(event.is_handled());
}

TEST(EventBasicTest, EventToString) {
    TestEvent event("hello", 100);
    event.set_source("TestApp");

    std::string str = event.to_string();
    EXPECT_THAT(str, HasSubstr("TestEvent"));
    EXPECT_THAT(str, HasSubstr("TestApp"));
    EXPECT_THAT(str, HasSubstr("hello"));
    EXPECT_THAT(str, HasSubstr("100"));
    EXPECT_THAT(str, HasSubstr("handled=false"));
}

TEST(EventBasicTest, TypedEventCasting) {
    TestEvent test_event("test", 42);
    Event& base_ref = test_event;
    (void)base_ref; // Suppress unused variable warning

    // Test safe casting
    const TestEvent* cast_result = test_event.as<TestEvent>();
    ASSERT_NE(cast_result, nullptr);
    EXPECT_EQ(cast_result->get_message(), "test");
    EXPECT_EQ(cast_result->get_value(), 42);

    // Test failed cast
    const ProgressEvent* failed_cast = test_event.as<ProgressEvent>();
    EXPECT_EQ(failed_cast, nullptr);
}

TEST(EventBasicTest, TypedEventDefaultName) {
    TestEvent event_with_default;
    EXPECT_EQ(event_with_default.get_type_name(), "TestEvent");

    TestEvent event_with_custom("CustomName");
    EXPECT_EQ(event_with_custom.get_type_name(), "TestEvent");
}

// ============================================================================
// Observer and Subject Tests
// ============================================================================

TEST_F(ObserverTest, BasicObserverSubjectInteraction) {
    Subject<TestEvent> subject;

    // Add observers
    subject.add_observer(observer1);
    subject.add_observer(observer2);
    EXPECT_EQ(subject.get_observer_count(), 2);

    // Notify observers
    TestEvent event("hello", 123);
    subject.notify_observers(event);

    EXPECT_EQ(observer1->get_event_count(), 1);
    EXPECT_EQ(observer2->get_event_count(), 1);

    auto obs1_events = observer1->get_received_events();
    EXPECT_EQ(obs1_events[0].first, "hello");
    EXPECT_EQ(obs1_events[0].second, 123);
}

TEST_F(ObserverTest, ObserverRemoval) {
    Subject<TestEvent> subject;

    subject.add_observer(observer1);
    subject.add_observer(observer2);
    EXPECT_EQ(subject.get_observer_count(), 2);

    // Remove one observer
    subject.remove_observer(observer1);
    EXPECT_EQ(subject.get_observer_count(), 1);

    // Notify remaining observer
    TestEvent event("after_removal", 456);
    subject.notify_observers(event);

    EXPECT_EQ(observer1->get_event_count(), 0);
    EXPECT_EQ(observer2->get_event_count(), 1);
}

TEST_F(ObserverTest, WeakReferenceManagement) {
    Subject<TestEvent> subject;

    {
        auto temp_observer = std::make_shared<TestObserver>("TempObserver");
        subject.add_observer(temp_observer);
        EXPECT_EQ(subject.get_observer_count(), 1);
    } // temp_observer goes out of scope

    // Notify should clean up expired weak references
    TestEvent event("cleanup_test", 789);
    subject.notify_observers(event);
    EXPECT_EQ(subject.get_observer_count(), 0);
}

TEST_F(ObserverTest, MultipleEventsToSameObserver) {
    Subject<TestEvent> subject;
    subject.add_observer(observer1);

    // Send multiple events
    for (int i = 0; i < 5; ++i) {
        TestEvent event("event_" + std::to_string(i), i * 10);
        subject.notify_observers(event);
    }

    EXPECT_EQ(observer1->get_event_count(), 5);
    auto events = observer1->get_received_events();
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(events[i].first, "event_" + std::to_string(i));
        EXPECT_EQ(events[i].second, i * 10);
    }
}

TEST_F(ObserverTest, ClearObservers) {
    Subject<TestEvent> subject;
    subject.add_observer(observer1);
    subject.add_observer(observer2);
    EXPECT_EQ(subject.get_observer_count(), 2);

    subject.clear_observers();
    EXPECT_EQ(subject.get_observer_count(), 0);

    TestEvent event("after_clear", 999);
    subject.notify_observers(event);
    EXPECT_EQ(observer1->get_event_count(), 0);
    EXPECT_EQ(observer2->get_event_count(), 0);
}

TEST_F(ObserverTest, SubjectWithDifferentEventTypes) {
    Subject<TestEvent> test_subject;
    Subject<ProgressEvent> progress_subject;

    test_subject.add_observer(observer1);
    progress_subject.add_observer(progress_observer);

    TestEvent test_event("test", 42);
    ProgressEvent progress_event(50.0, "loading");

    test_subject.notify_observers(test_event);
    progress_subject.notify_observers(progress_event);

    EXPECT_EQ(observer1->get_event_count(), 1);
    EXPECT_EQ(progress_observer->get_progress_updates().size(), 1);
    EXPECT_EQ(progress_observer->get_progress_updates()[0].first, 50.0);
}

// ============================================================================
// Event Dispatcher Tests
// ============================================================================

TEST_F(EventDispatcherTest, BasicEventSubscriptionAndDispatch) {
    std::vector<TestEvent> received_events;

    auto subscription = dispatcher->subscribe<TestEvent>(
        [&received_events](const TestEvent& event) {
            received_events.push_back(event);
        }
    );

    EXPECT_TRUE(subscription->is_active());

    TestEvent event1("first", 1);
    TestEvent event2("second", 2);

    dispatcher->dispatch(event1);
    dispatcher->dispatch(event2);

    EXPECT_EQ(received_events.size(), 2);
    EXPECT_EQ(received_events[0].get_message(), "first");
    EXPECT_EQ(received_events[1].get_message(), "second");
}

TEST_F(EventDispatcherTest, MultipleHandlersForSameEventType) {
    std::vector<std::string> handler1_messages;
    std::vector<std::string> handler2_messages;

    auto sub1 = dispatcher->subscribe<TestEvent>(
        [&handler1_messages](const TestEvent& event) {
            handler1_messages.push_back("H1:" + event.get_message());
        }
    );

    auto sub2 = dispatcher->subscribe<TestEvent>(
        [&handler2_messages](const TestEvent& event) {
            handler2_messages.push_back("H2:" + event.get_message());
        }
    );

    TestEvent event("broadcast", 42);
    dispatcher->dispatch(event);

    EXPECT_EQ(handler1_messages.size(), 1);
    EXPECT_EQ(handler2_messages.size(), 1);
    EXPECT_EQ(handler1_messages[0], "H1:broadcast");
    EXPECT_EQ(handler2_messages[0], "H2:broadcast");
}

TEST_F(EventDispatcherTest, UniversalEventHandler) {
    std::vector<std::string> all_events;

    auto universal_sub = dispatcher->subscribe_all(
        [&all_events](const Event& event) {
            all_events.push_back(event.get_type_name());
        }
    );

    TestEvent test_event("test", 1);
    ProgressEvent progress_event(25.0, "task");
    ErrorEvent error_event("Something went wrong", 404);

    dispatcher->dispatch(test_event);
    dispatcher->dispatch(progress_event);
    dispatcher->dispatch(error_event);

    EXPECT_EQ(all_events.size(), 3);
    EXPECT_THAT(all_events, ElementsAre("TestEvent", "ProgressEvent", "ErrorEvent"));
}

TEST_F(EventDispatcherTest, EventHandling) {
    bool test_handler_called = false;
    bool universal_handler_called = false;

    auto test_sub = dispatcher->subscribe<TestEvent>(
        [&test_handler_called](const TestEvent& event) {
            test_handler_called = true;
            event.set_handled(true); // Mark as handled
        }
    );

    auto universal_sub = dispatcher->subscribe_all(
        [&universal_handler_called](const Event& /*event*/) {
            universal_handler_called = true;
        }
    );

    TestEvent event("handled_test", 42);
    dispatcher->dispatch(event);

    EXPECT_TRUE(universal_handler_called); // Universal handlers called first
    EXPECT_TRUE(test_handler_called);      // Typed handler also called
    EXPECT_TRUE(event.is_handled());
}

TEST_F(EventDispatcherTest, EventEmit) {
    std::vector<TestEvent> received_events;

    auto subscription = dispatcher->subscribe<TestEvent>(
        [&received_events](const TestEvent& event) {
            received_events.push_back(event);
        }
    );

    // Test emit with arguments
    dispatcher->emit<TestEvent>("emitted", 999);

    EXPECT_EQ(received_events.size(), 1);
    EXPECT_EQ(received_events[0].get_message(), "emitted");
    EXPECT_EQ(received_events[0].get_value(), 999);
}

TEST_F(EventDispatcherTest, DispatcherStatistics) {
    auto stats_initial = dispatcher->get_statistics();
    EXPECT_EQ(stats_initial.typed_handler_types, 0);
    EXPECT_EQ(stats_initial.total_typed_handlers, 0);
    EXPECT_EQ(stats_initial.universal_handlers, 0);

    auto test_sub1 = dispatcher->subscribe<TestEvent>([](const TestEvent&) {});
    auto test_sub2 = dispatcher->subscribe<TestEvent>([](const TestEvent&) {});
    auto progress_sub = dispatcher->subscribe<ProgressEvent>([](const ProgressEvent&) {});
    auto universal_sub = dispatcher->subscribe_all([](const Event&) {});

    auto stats_after = dispatcher->get_statistics();
    EXPECT_EQ(stats_after.typed_handler_types, 2);  // TestEvent, ProgressEvent
    EXPECT_EQ(stats_after.total_typed_handlers, 3); // 2 TestEvent + 1 ProgressEvent
    EXPECT_EQ(stats_after.universal_handlers, 1);
}

TEST_F(EventDispatcherTest, DispatcherClear) {
    auto sub1 = dispatcher->subscribe<TestEvent>([](const TestEvent&) {});
    auto sub2 = dispatcher->subscribe_all([](const Event&) {});

    auto stats_before = dispatcher->get_statistics();
    EXPECT_GT(stats_before.total_typed_handlers + stats_before.universal_handlers, 0);

    dispatcher->clear();

    auto stats_after = dispatcher->get_statistics();
    EXPECT_EQ(stats_after.typed_handler_types, 0);
    EXPECT_EQ(stats_after.total_typed_handlers, 0);
    EXPECT_EQ(stats_after.universal_handlers, 0);
}

// ============================================================================
// Event Subscription Tests
// ============================================================================

TEST_F(EventDispatcherTest, SubscriptionRAII) {
    std::vector<TestEvent> received_events;

    {
        auto subscription = dispatcher->subscribe<TestEvent>(
            [&received_events](const TestEvent& event) {
                received_events.push_back(event);
            }
        );

        TestEvent event1("before_destruction", 1);
        dispatcher->dispatch(event1);
        EXPECT_EQ(received_events.size(), 1);

    } // subscription goes out of scope

    TestEvent event2("after_destruction", 2);
    dispatcher->dispatch(event2);
    EXPECT_EQ(received_events.size(), 1); // Should not receive second event
}

TEST_F(EventDispatcherTest, ManualUnsubscribe) {
    std::vector<TestEvent> received_events;

    auto subscription = dispatcher->subscribe<TestEvent>(
        [&received_events](const TestEvent& event) {
            received_events.push_back(event);
        }
    );

    TestEvent event1("before_unsubscribe", 1);
    dispatcher->dispatch(event1);
    EXPECT_EQ(received_events.size(), 1);
    EXPECT_TRUE(subscription->is_active());

    subscription->unsubscribe();
    EXPECT_FALSE(subscription->is_active());

    TestEvent event2("after_unsubscribe", 2);
    dispatcher->dispatch(event2);
    EXPECT_EQ(received_events.size(), 1); // Should not receive second event
}

TEST_F(EventDispatcherTest, SubscriptionMove) {
    std::vector<TestEvent> received_events;

    auto subscription1 = dispatcher->subscribe<TestEvent>(
        [&received_events](const TestEvent& event) {
            received_events.push_back(event);
        }
    );

    // Move subscription
    auto subscription2 = std::move(subscription1);
    // Note: subscription1 is now in a moved-from state and should not be accessed
    EXPECT_TRUE(subscription2->is_active());

    TestEvent event("move_test", 42);
    dispatcher->dispatch(event);
    EXPECT_EQ(received_events.size(), 1);
}

TEST_F(EventDispatcherTest, MultipleUnsubscribeCalls) {
    auto subscription = dispatcher->subscribe<TestEvent>([](const TestEvent&) {});

    EXPECT_TRUE(subscription->is_active());
    subscription->unsubscribe();
    EXPECT_FALSE(subscription->is_active());

    // Multiple unsubscribe calls should be safe
    subscription->unsubscribe();
    EXPECT_FALSE(subscription->is_active());
}

// ============================================================================
// Event Bus Tests
// ============================================================================

TEST(EventBusTest, GlobalEventBusSingleton) {
    auto& bus1 = EventBus::instance();
    auto& bus2 = EventBus::instance();

    EXPECT_EQ(&bus1, &bus2); // Should be the same instance
}

TEST(EventBusTest, EventBusBasicFunctionality) {
    std::vector<TestEvent> received_events;

    auto subscription = EventBus::instance().subscribe<TestEvent>(
        [&received_events](const TestEvent& event) {
            received_events.push_back(event);
        }
    );

    EventBus::instance().emit<TestEvent>("global_test", 777);

    EXPECT_EQ(received_events.size(), 1);
    EXPECT_EQ(received_events[0].get_message(), "global_test");
    EXPECT_EQ(received_events[0].get_value(), 777);

    subscription->unsubscribe(); // Clean up
}

TEST(EventBusTest, ConvenienceFunctions) {
    std::vector<TestEvent> received_events;

    auto subscription = subscribe_to_events<TestEvent>(
        [&received_events](const TestEvent& event) {
            received_events.push_back(event);
        }
    );

    emit_event<TestEvent>("convenience_test", 888);

    EXPECT_EQ(received_events.size(), 1);
    EXPECT_EQ(received_events[0].get_message(), "convenience_test");

    subscription->unsubscribe();
}

TEST(EventBusTest, UniversalSubscription) {
    std::vector<std::string> event_types;

    auto subscription = subscribe_to_all_events(
        [&event_types](const Event& event) {
            event_types.push_back(event.get_type_name());
        }
    );

    emit_event<TestEvent>("test", 1);
    emit_event<ProgressEvent>(50.0, "loading");

    EXPECT_EQ(event_types.size(), 2);
    EXPECT_THAT(event_types, ElementsAre("TestEvent", "ProgressEvent"));

    subscription->unsubscribe();
}

// ============================================================================
// Scoped Event Emitter Tests
// ============================================================================

TEST(ScopedEventEmitterTest, StartEndEventEmission) {
    std::vector<std::string> event_sequence;

    auto start_sub = subscribe_to_events<OperationStartEvent>(
        [&event_sequence](const OperationStartEvent& event) {
            event_sequence.push_back("START:" + event.get_operation_name());
        }
    );

    auto end_sub = subscribe_to_events<OperationEndEvent>(
        [&event_sequence](const OperationEndEvent& event) {
            event_sequence.push_back("END:" + event.get_operation_name());
        }
    );

    {
        ScopedEventEmitter<OperationStartEvent, OperationEndEvent> emitter("test_operation");
        // Do some work...
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } // End event emitted here

    EXPECT_EQ(event_sequence.size(), 2);
    EXPECT_EQ(event_sequence[0], "START:test_operation");
    EXPECT_EQ(event_sequence[1], "END:test_operation");

    start_sub->unsubscribe();
    end_sub->unsubscribe();
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST_F(ObserverTest, ConcurrentObserverNotification) {
    Subject<TestEvent> subject;
    subject.add_observer(thread_safe_observer);

    const int num_threads = 10;
    const int events_per_thread = 100;
    std::vector<std::thread> threads;

    // Launch threads that send events concurrently
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&subject, t, events_per_thread]() {
            for (int i = 0; i < events_per_thread; ++i) {
                TestEvent event("thread_" + std::to_string(t) + "_event_" + std::to_string(i), i);
                subject.notify_observers(event);
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(thread_safe_observer->get_event_count(), num_threads * events_per_thread);
    EXPECT_EQ(thread_safe_observer->get_unique_thread_count(), num_threads);
}

TEST_F(EventDispatcherTest, ConcurrentEventDispatch) {
    std::atomic<int> events_received{0};

    auto subscription = dispatcher->subscribe<TestEvent>(
        [&events_received](const TestEvent& /*event*/) {
            events_received.fetch_add(1, std::memory_order_relaxed);
        }
    );

    const int num_threads = 8;
    const int events_per_thread = 50;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this, t, events_per_thread]() {
            for (int i = 0; i < events_per_thread; ++i) {
                TestEvent event("concurrent_" + std::to_string(t), i);
                dispatcher->dispatch(event);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(events_received.load(), num_threads * events_per_thread);
}

TEST_F(EventDispatcherTest, ConcurrentSubscriptionAndDispatch) {
    std::atomic<int> total_events{0};
    std::vector<std::unique_ptr<EventSubscription>> subscriptions;
    std::mutex subscriptions_mutex;

    const int num_subscriber_threads = 4;
    const int num_dispatcher_threads = 4;
    const int events_per_thread = 25;

    std::vector<std::thread> threads;

    // Threads that create subscriptions
    for (int t = 0; t < num_subscriber_threads; ++t) {
        threads.emplace_back([this, &total_events, &subscriptions, &subscriptions_mutex]() {
            auto sub = dispatcher->subscribe<TestEvent>(
                [&total_events](const TestEvent&) {
                    total_events.fetch_add(1, std::memory_order_relaxed);
                }
            );

            std::lock_guard lock(subscriptions_mutex);
            subscriptions.push_back(std::move(sub));
        });
    }

    // Wait for subscriptions to be created
    for (auto& thread : threads) {
        thread.join();
    }
    threads.clear();

    // Threads that dispatch events
    for (int t = 0; t < num_dispatcher_threads; ++t) {
        threads.emplace_back([this, t, events_per_thread]() {
            for (int i = 0; i < events_per_thread; ++i) {
                TestEvent event("dispatcher_" + std::to_string(t), i);
                dispatcher->dispatch(event);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Each event should be received by all subscribers
    EXPECT_EQ(total_events.load(), num_subscriber_threads * num_dispatcher_threads * events_per_thread);
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST_F(EventDispatcherTest, EventDispatchPerformance) {
    std::atomic<int> events_processed{0};

    auto subscription = dispatcher->subscribe<TestEvent>(
        [&events_processed](const TestEvent&) {
            events_processed.fetch_add(1, std::memory_order_relaxed);
        }
    );

    const int num_events = 100000;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_events; ++i) {
        TestEvent event("perf_test_" + std::to_string(i), i);
        dispatcher->dispatch(event);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    EXPECT_EQ(events_processed.load(), num_events);
    // Should dispatch 100K events in reasonable time (less than 1 second)
    EXPECT_LT(duration.count(), 1000000);
}

TEST_F(ObserverTest, SubjectNotificationPerformance) {
    Subject<TestEvent> subject;

    // Add multiple observers
    std::vector<std::shared_ptr<TestObserver>> observers;
    for (int i = 0; i < 100; ++i) {
        auto observer = std::make_shared<TestObserver>("Observer_" + std::to_string(i));
        observers.push_back(observer);
        subject.add_observer(observer);
    }

    const int num_events = 1000;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_events; ++i) {
        TestEvent event("perf_event_" + std::to_string(i), i);
        subject.notify_observers(event);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Each observer should receive all events
    for (const auto& observer : observers) {
        EXPECT_EQ(observer->get_event_count(), num_events);
    }

    // Should complete in reasonable time
    EXPECT_LT(duration.count(), 1000);
}

// ============================================================================
// Error Handling and Edge Cases
// ============================================================================

TEST_F(ObserverTest, EmptySubjectNotification) {
    Subject<TestEvent> empty_subject;
    EXPECT_EQ(empty_subject.get_observer_count(), 0);

    // Should handle empty observer list gracefully
    TestEvent event("empty_test", 42);
    EXPECT_NO_THROW(empty_subject.notify_observers(event));
}

TEST_F(EventDispatcherTest, DispatchWithNoHandlers) {
    TestEvent event("no_handlers", 42);
    EXPECT_NO_THROW(dispatcher->dispatch(event));

    auto stats = dispatcher->get_statistics();
    EXPECT_EQ(stats.total_typed_handlers, 0);
    EXPECT_EQ(stats.universal_handlers, 0);
}

TEST_F(EventDispatcherTest, HandlerExceptionHandling) {
    bool handler1_called = false;
    bool handler2_called = false;

    auto sub1 = dispatcher->subscribe<TestEvent>(
        [&handler1_called](const TestEvent&) {
            handler1_called = true;
            throw std::runtime_error("Handler 1 exception");
        }
    );

    auto sub2 = dispatcher->subscribe<TestEvent>(
        [&handler2_called](const TestEvent&) {
            handler2_called = true;
        }
    );

    TestEvent event("exception_test", 42);

    // Dispatcher does not catch exceptions, so they propagate
    EXPECT_THROW(dispatcher->dispatch(event), std::runtime_error);

    EXPECT_TRUE(handler1_called);
    EXPECT_TRUE(handler2_called); // Called because handler order in unordered_map is unspecified
}

TEST_F(ObserverTest, NullObserverHandling) {
    Subject<TestEvent> subject;

    // Add null observer (should be handled gracefully)
    std::shared_ptr<Observer<TestEvent>> null_observer;
    EXPECT_NO_THROW(subject.add_observer(null_observer));

    TestEvent event("null_test", 42);
    EXPECT_NO_THROW(subject.notify_observers(event));
}

TEST_F(EventDispatcherTest, MultipleSubscriptionsFromSameHandler) {
    int call_count = 0;
    auto handler = [&call_count](const TestEvent&) { call_count++; };

    auto sub1 = dispatcher->subscribe<TestEvent>(handler);
    auto sub2 = dispatcher->subscribe<TestEvent>(handler);

    TestEvent event("duplicate_handler", 42);
    dispatcher->dispatch(event);

    // Both subscriptions should trigger
    EXPECT_EQ(call_count, 2);
}

// ============================================================================
// Memory Management Tests
// ============================================================================

TEST_F(EventDispatcherTest, MemoryLeakPrevention) {
    // Create many subscriptions and let them go out of scope
    for (int i = 0; i < 1000; ++i) {
        auto subscription = dispatcher->subscribe<TestEvent>([](const TestEvent&) {});
        // subscription automatically unsubscribes when destroyed
    }

    auto stats = dispatcher->get_statistics();
    EXPECT_EQ(stats.total_typed_handlers, 0);
    EXPECT_EQ(stats.universal_handlers, 0);
}

TEST_F(ObserverTest, WeakReferenceCleanup) {
    Subject<TestEvent> subject;

    // Add observers that will go out of scope
    for (int i = 0; i < 10; ++i) {
        auto temp_observer = std::make_shared<TestObserver>("Temp_" + std::to_string(i));
        subject.add_observer(temp_observer);
    } // All temp observers destroyed here

    EXPECT_EQ(subject.get_observer_count(), 0); // Expired references are not counted

    // Notification should clean up expired references
    TestEvent event("cleanup", 42);
    subject.notify_observers(event);

    EXPECT_EQ(subject.get_observer_count(), 0); // Should be cleaned up now
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(ObserverIntegrationTest, ComplexEventWorkflow) {
    // Create a complex workflow with multiple event types and handlers
    std::vector<std::string> workflow_log;

    auto start_sub = subscribe_to_events<OperationStartEvent>(
        [&workflow_log](const OperationStartEvent& event) {
            workflow_log.push_back("Started: " + event.get_operation_name());
        }
    );

    auto progress_sub = subscribe_to_events<ProgressEvent>(
        [&workflow_log](const ProgressEvent& event) {
            workflow_log.push_back("Progress: " + std::to_string(static_cast<int>(event.get_percentage())) + "%");
        }
    );

    auto error_sub = subscribe_to_events<ErrorEvent>(
        [&workflow_log](const ErrorEvent& event) {
            workflow_log.push_back("Error: " + event.get_error_message());
        }
    );

    auto end_sub = subscribe_to_events<OperationEndEvent>(
        [&workflow_log](const OperationEndEvent& event) {
            workflow_log.push_back("Completed: " + event.get_operation_name());
        }
    );

    // Simulate a workflow
    emit_event<OperationStartEvent>("file_processing");
    emit_event<ProgressEvent>(25.0, "reading");
    emit_event<ProgressEvent>(50.0, "processing");
    emit_event<ProgressEvent>(75.0, "writing");
    emit_event<ProgressEvent>(100.0, "done");
    emit_event<OperationEndEvent>("file_processing");

    EXPECT_EQ(workflow_log.size(), 6);
    EXPECT_EQ(workflow_log[0], "Started: file_processing");
    EXPECT_EQ(workflow_log[1], "Progress: 25%");
    EXPECT_EQ(workflow_log[2], "Progress: 50%");
    EXPECT_EQ(workflow_log[3], "Progress: 75%");
    EXPECT_EQ(workflow_log[4], "Progress: 100%");
    EXPECT_EQ(workflow_log[5], "Completed: file_processing");

    // Clean up subscriptions
    start_sub->unsubscribe();
    progress_sub->unsubscribe();
    error_sub->unsubscribe();
    end_sub->unsubscribe();
}

TEST(ObserverIntegrationTest, MixedDispatcherAndSubject) {
    // Test interaction between EventDispatcher and Subject
    EventDispatcher dispatcher;
    Subject<TestEvent> subject;

    std::vector<std::string> dispatcher_events;
    std::vector<std::string> subject_events;

    auto dispatcher_sub = dispatcher.subscribe<TestEvent>(
        [&dispatcher_events](const TestEvent& event) {
            dispatcher_events.push_back("D:" + event.get_message());
        }
    );

    auto observer = std::make_shared<TestObserver>("SubjectObserver");
    subject.add_observer(observer);

    // Send events through both systems
    TestEvent event1("via_dispatcher", 1);
    TestEvent event2("via_subject", 2);

    dispatcher.dispatch(event1);
    subject.notify_observers(event2);

    EXPECT_EQ(dispatcher_events.size(), 1);
    EXPECT_EQ(dispatcher_events[0], "D:via_dispatcher");

    EXPECT_EQ(observer->get_event_count(), 1);
    EXPECT_EQ(observer->get_received_events()[0].first, "via_subject");
}

// ============================================================================
// Mock Observer Integration Tests
// ============================================================================

TEST(ObserverMockTest, MockObserverExpectations) {
    auto mock_observer = std::make_shared<MockTestObserver>();
    Subject<TestEvent> subject;

    EXPECT_CALL(*mock_observer, on_event(_))
        .Times(3)
        .WillRepeatedly([](const TestEvent& /*event*/) {
            // Verify event properties in mock
        });

    subject.add_observer(mock_observer);

    TestEvent event1("mock_test_1", 1);
    TestEvent event2("mock_test_2", 2);
    TestEvent event3("mock_test_3", 3);

    subject.notify_observers(event1);
    subject.notify_observers(event2);
    subject.notify_observers(event3);
}

TEST(ObserverMockTest, MockObserverWithEventHandling) {
    auto mock_observer = std::make_shared<MockTestObserver>();

    EXPECT_CALL(*mock_observer, on_event(_))
        .WillOnce([](const TestEvent& event) {
            event.set_handled(true);
        });

    Subject<TestEvent> subject;
    subject.add_observer(mock_observer);

    TestEvent event("handled_by_mock", 42);
    subject.notify_observers(event);

    EXPECT_TRUE(event.is_handled());
}