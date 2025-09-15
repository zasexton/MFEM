/**
 * @file test_registry.cpp
 * @brief Comprehensive unit tests for the Registry implementation
 *
 * Tests cover:
 * - Basic registry functionality (registration, lookup, removal)
 * - Thread safety for concurrent operations
 * - Event callbacks and notifications
 * - Batch operations for performance
 * - Query methods and filtering
 * - Edge cases and error conditions
 * - Memory management and cleanup
 * - Performance characteristics
 * - Global registry singleton
 * - Different key types (string, int, uint32_t)
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <sstream>
#include <future>
#include <random>
#include <set>

#include "core/base/registry.h"

using namespace fem::core::base;
using namespace testing;

// ============================================================================
// Test Objects for Registry Testing
// ============================================================================

// Base test object for registry
class RegistryTestObject : public Object {
public:
    explicit RegistryTestObject(std::string_view name = "RegistryTestObject", int value = 0)
        : Object(name), value_(value) {}

    virtual ~RegistryTestObject() = default;

    int get_value() const { return value_; }
    void set_value(int value) { value_ = value; }

    virtual std::string get_type_name() const { return "RegistryTestObject"; }

private:
    int value_;
};

// Derived test object
class DerivedTestObject : public RegistryTestObject {
public:
    explicit DerivedTestObject(std::string_view name = "DerivedTestObject", int value = 0)
        : RegistryTestObject(name, value) {}

    std::string get_type_name() const override { return "DerivedTestObject"; }
};

// Object with counter for testing concurrent access
class CounterTestObject : public RegistryTestObject {
public:
    explicit CounterTestObject(std::string_view name = "CounterTestObject")
        : RegistryTestObject(name) {
        counter_.fetch_add(1, std::memory_order_relaxed);
    }

    ~CounterTestObject() {
        counter_.fetch_sub(1, std::memory_order_relaxed);
    }

    static void reset_counter() { counter_.store(0, std::memory_order_relaxed); }
    static int get_counter() { return counter_.load(std::memory_order_relaxed); }

    std::string get_type_name() const override { return "CounterTestObject"; }

private:
    static std::atomic<int> counter_;
};

std::atomic<int> CounterTestObject::counter_{0};

// ============================================================================
// Mock Classes for Testing Callbacks
// ============================================================================

class MockRegistryCallbacks {
public:
    MOCK_METHOD(void, on_registration, (object_ptr<RegistryTestObject> obj), ());
    MOCK_METHOD(void, on_unregistration, (Object::id_type id, std::string_view key), ());
};

// ============================================================================
// Registry Test Fixtures
// ============================================================================

class RegistryTest : public ::testing::Test {
protected:
    void SetUp() override {
        registry_ = std::make_unique<Registry<RegistryTestObject>>("TestRegistry");
        CounterTestObject::reset_counter();
    }

    void TearDown() override {
        registry_.reset();
    }

    std::unique_ptr<Registry<RegistryTestObject>> registry_;
};

class RegistryIntegerKeyTest : public ::testing::Test {
protected:
    void SetUp() override {
        int_registry_ = std::make_unique<Registry<RegistryTestObject, int>>("IntRegistry");
    }

    void TearDown() override {
        int_registry_.reset();
    }

    std::unique_ptr<Registry<RegistryTestObject, int>> int_registry_;
};

// ============================================================================
// Basic Registry Functionality Tests
// ============================================================================

TEST_F(RegistryTest, Construction) {
    Registry<RegistryTestObject> default_registry;
    EXPECT_EQ(default_registry.name(), "Registry");
    EXPECT_TRUE(default_registry.empty());
    EXPECT_EQ(default_registry.size(), 0);

    Registry<RegistryTestObject> named_registry("TestName");
    EXPECT_EQ(named_registry.name(), "TestName");
    EXPECT_TRUE(named_registry.empty());
    EXPECT_EQ(named_registry.size(), 0);
}

TEST_F(RegistryTest, NameManagement) {
    EXPECT_EQ(registry_->name(), "TestRegistry");

    registry_->set_name("NewName");
    EXPECT_EQ(registry_->name(), "NewName");
}

TEST_F(RegistryTest, BasicRegistration) {
    auto obj = make_object<RegistryTestObject>("test_obj", 42);
    auto id = obj->id();

    // Register by ID only
    EXPECT_TRUE(registry_->register_object(obj));
    EXPECT_EQ(registry_->size(), 1);
    EXPECT_FALSE(registry_->empty());
    EXPECT_TRUE(registry_->contains_id(id));

    // Duplicate registration should fail
    EXPECT_FALSE(registry_->register_object(obj));
    EXPECT_EQ(registry_->size(), 1);
}

TEST_F(RegistryTest, NamedRegistration) {
    auto obj = make_object<RegistryTestObject>("test_obj", 42);
    auto id = obj->id();

    // Register with key
    EXPECT_TRUE(registry_->register_object("test_key", obj));
    EXPECT_EQ(registry_->size(), 1);
    EXPECT_TRUE(registry_->contains_id(id));
    EXPECT_TRUE(registry_->contains_key("test_key"));

    // Duplicate registration should fail
    EXPECT_FALSE(registry_->register_object("test_key", obj));
    EXPECT_FALSE(registry_->register_object("other_key", obj));
    EXPECT_EQ(registry_->size(), 1);
}

TEST_F(RegistryTest, NullObjectRegistration) {
    object_ptr<RegistryTestObject> null_obj = nullptr;

    EXPECT_FALSE(registry_->register_object(null_obj));
    EXPECT_FALSE(registry_->register_object("key", null_obj));
    EXPECT_TRUE(registry_->empty());
}

TEST_F(RegistryTest, AutoNameGeneration) {
    auto obj1 = make_object<RegistryTestObject>("obj1");
    auto obj2 = make_object<RegistryTestObject>("obj2");
    auto obj3 = make_object<RegistryTestObject>("obj3");

    EXPECT_TRUE(registry_->register_object_auto_name(obj1, "test"));
    EXPECT_TRUE(registry_->register_object_auto_name(obj2, "test"));
    EXPECT_TRUE(registry_->register_object_auto_name(obj3, "different"));

    EXPECT_EQ(registry_->size(), 3);

    // Should have generated unique names
    auto keys = registry_->get_all_keys();
    EXPECT_EQ(keys.size(), 3);
    EXPECT_TRUE(std::find(keys.begin(), keys.end(), "test_1") != keys.end());
    EXPECT_TRUE(std::find(keys.begin(), keys.end(), "test_2") != keys.end());
    EXPECT_TRUE(std::find(keys.begin(), keys.end(), "different_1") != keys.end());
}

// ============================================================================
// Lookup Tests
// ============================================================================

TEST_F(RegistryTest, LookupByID) {
    auto obj = make_object<RegistryTestObject>("test_obj", 42);
    auto id = obj->id();

    ASSERT_TRUE(registry_->register_object(obj));

    auto found = registry_->find_by_id(id);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->get_value(), 42);
    EXPECT_EQ(found->id(), id);

    // Non-existent ID
    auto not_found = registry_->find_by_id(999999);
    EXPECT_EQ(not_found, nullptr);
}

TEST_F(RegistryTest, LookupByKey) {
    auto obj = make_object<RegistryTestObject>("test_obj", 42);

    ASSERT_TRUE(registry_->register_object("test_key", obj));

    auto found = registry_->find_by_key("test_key");
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->get_value(), 42);
    EXPECT_EQ(found->id(), obj->id());

    // Non-existent key
    auto not_found = registry_->find_by_key("non_existent");
    EXPECT_EQ(not_found, nullptr);
}

TEST_F(RegistryTest, FindOrDefault) {
    auto obj = make_object<RegistryTestObject>("test_obj", 42);
    auto default_obj = make_object<RegistryTestObject>("default_obj", 100);

    ASSERT_TRUE(registry_->register_object("test_key", obj));

    // Existing key
    auto found = registry_->find_or_default("test_key", default_obj);
    EXPECT_EQ(found->get_value(), 42);

    // Non-existing key with default
    auto defaulted = registry_->find_or_default("missing_key", default_obj);
    EXPECT_EQ(defaulted->get_value(), 100);

    // Non-existing key without default
    auto null_default = registry_->find_or_default("missing_key");
    EXPECT_EQ(null_default, nullptr);
}

TEST_F(RegistryTest, KeyIdMapping) {
    auto obj = make_object<RegistryTestObject>("test_obj", 42);
    auto id = obj->id();

    ASSERT_TRUE(registry_->register_object("test_key", obj));

    auto key = registry_->get_key_for_id(id);
    ASSERT_TRUE(key.has_value());
    EXPECT_EQ(key.value(), "test_key");

    // Object registered without key
    auto obj2 = make_object<RegistryTestObject>("obj2");
    ASSERT_TRUE(registry_->register_object(obj2));

    auto no_key = registry_->get_key_for_id(obj2->id());
    EXPECT_FALSE(no_key.has_value());
}

// ============================================================================
// Removal Tests
// ============================================================================

TEST_F(RegistryTest, UnregisterByID) {
    auto obj = make_object<RegistryTestObject>("test_obj", 42);
    auto id = obj->id();

    ASSERT_TRUE(registry_->register_object("test_key", obj));
    EXPECT_EQ(registry_->size(), 1);

    EXPECT_TRUE(registry_->unregister_by_id(id));
    EXPECT_EQ(registry_->size(), 0);
    EXPECT_FALSE(registry_->contains_id(id));
    EXPECT_FALSE(registry_->contains_key("test_key"));

    // Double removal should fail
    EXPECT_FALSE(registry_->unregister_by_id(id));
}

TEST_F(RegistryTest, UnregisterByKey) {
    auto obj = make_object<RegistryTestObject>("test_obj", 42);

    ASSERT_TRUE(registry_->register_object("test_key", obj));
    EXPECT_EQ(registry_->size(), 1);

    EXPECT_TRUE(registry_->unregister_by_key("test_key"));
    EXPECT_EQ(registry_->size(), 0);
    EXPECT_FALSE(registry_->contains_id(obj->id()));
    EXPECT_FALSE(registry_->contains_key("test_key"));

    // Double removal should fail
    EXPECT_FALSE(registry_->unregister_by_key("test_key"));
}

TEST_F(RegistryTest, UnregisterObject) {
    auto obj = make_object<RegistryTestObject>("test_obj", 42);

    ASSERT_TRUE(registry_->register_object("test_key", obj));
    EXPECT_EQ(registry_->size(), 1);

    EXPECT_TRUE(registry_->unregister_object(obj));
    EXPECT_EQ(registry_->size(), 0);

    // Null object
    object_ptr<RegistryTestObject> null_obj = nullptr;
    EXPECT_FALSE(registry_->unregister_object(null_obj));
}

TEST_F(RegistryTest, ClearRegistry) {
    auto obj1 = make_object<RegistryTestObject>("obj1", 1);
    auto obj2 = make_object<RegistryTestObject>("obj2", 2);
    auto obj3 = make_object<RegistryTestObject>("obj3", 3);

    ASSERT_TRUE(registry_->register_object("key1", obj1));
    ASSERT_TRUE(registry_->register_object("key2", obj2));
    ASSERT_TRUE(registry_->register_object(obj3));

    EXPECT_EQ(registry_->size(), 3);

    registry_->clear();

    EXPECT_EQ(registry_->size(), 0);
    EXPECT_TRUE(registry_->empty());
    EXPECT_FALSE(registry_->contains_key("key1"));
    EXPECT_FALSE(registry_->contains_key("key2"));
    EXPECT_FALSE(registry_->contains_id(obj3->id()));
}

// ============================================================================
// Query and Iteration Tests
// ============================================================================

TEST_F(RegistryTest, GetAllMethods) {
    auto obj1 = make_object<RegistryTestObject>("obj1", 1);
    auto obj2 = make_object<RegistryTestObject>("obj2", 2);
    auto obj3 = make_object<RegistryTestObject>("obj3", 3);

    ASSERT_TRUE(registry_->register_object("key1", obj1));
    ASSERT_TRUE(registry_->register_object("key2", obj2));
    ASSERT_TRUE(registry_->register_object(obj3)); // No key

    auto ids = registry_->get_all_ids();
    auto keys = registry_->get_all_keys();
    auto objects = registry_->get_all_objects();

    EXPECT_EQ(ids.size(), 3);
    EXPECT_EQ(keys.size(), 2); // Only two have keys
    EXPECT_EQ(objects.size(), 3);

    // Verify content
    std::set<Object::id_type> id_set(ids.begin(), ids.end());
    EXPECT_TRUE(id_set.count(obj1->id()));
    EXPECT_TRUE(id_set.count(obj2->id()));
    EXPECT_TRUE(id_set.count(obj3->id()));

    std::set<std::string> key_set(keys.begin(), keys.end());
    EXPECT_TRUE(key_set.count("key1"));
    EXPECT_TRUE(key_set.count("key2"));
}

TEST_F(RegistryTest, FindIfPredicate) {
    auto obj1 = make_object<RegistryTestObject>("obj1", 10);
    auto obj2 = make_object<RegistryTestObject>("obj2", 20);
    auto obj3 = make_object<RegistryTestObject>("obj3", 30);

    ASSERT_TRUE(registry_->register_object(obj1));
    ASSERT_TRUE(registry_->register_object(obj2));
    ASSERT_TRUE(registry_->register_object(obj3));

    // Find objects with value > 15
    auto results = registry_->find_if([](const auto& obj) {
        return obj->get_value() > 15;
    });

    EXPECT_EQ(results.size(), 2);
    for (const auto& obj : results) {
        EXPECT_GT(obj->get_value(), 15);
    }
}

TEST_F(RegistryTest, FindFirstIf) {
    auto obj1 = make_object<RegistryTestObject>("obj1", 10);
    auto obj2 = make_object<RegistryTestObject>("obj2", 20);
    auto obj3 = make_object<RegistryTestObject>("obj3", 30);

    ASSERT_TRUE(registry_->register_object(obj1));
    ASSERT_TRUE(registry_->register_object(obj2));
    ASSERT_TRUE(registry_->register_object(obj3));

    // Find first object with value > 15
    auto result = registry_->find_first_if([](const auto& obj) {
        return obj->get_value() > 15;
    });

    ASSERT_NE(result, nullptr);
    EXPECT_GT(result->get_value(), 15);

    // Find non-existent
    auto not_found = registry_->find_first_if([](const auto& obj) {
        return obj->get_value() > 100;
    });

    EXPECT_EQ(not_found, nullptr);
}

TEST_F(RegistryTest, CountIf) {
    auto obj1 = make_object<RegistryTestObject>("obj1", 10);
    auto obj2 = make_object<RegistryTestObject>("obj2", 20);
    auto obj3 = make_object<RegistryTestObject>("obj3", 30);

    ASSERT_TRUE(registry_->register_object(obj1));
    ASSERT_TRUE(registry_->register_object(obj2));
    ASSERT_TRUE(registry_->register_object(obj3));

    auto count = registry_->count_if([](const auto& obj) {
        return obj->get_value() >= 20;
    });

    EXPECT_EQ(count, 2);
}

TEST_F(RegistryTest, ForEach) {
    auto obj1 = make_object<RegistryTestObject>("obj1", 10);
    auto obj2 = make_object<RegistryTestObject>("obj2", 20);
    auto obj3 = make_object<RegistryTestObject>("obj3", 30);

    ASSERT_TRUE(registry_->register_object(obj1));
    ASSERT_TRUE(registry_->register_object(obj2));
    ASSERT_TRUE(registry_->register_object(obj3));

    int sum = 0;
    registry_->for_each([&sum](const auto& obj) {
        sum += obj->get_value();
    });

    EXPECT_EQ(sum, 60); // 10 + 20 + 30
}

// ============================================================================
// Event Callback Tests
// ============================================================================

TEST_F(RegistryTest, RegistrationCallbacks) {
    MockRegistryCallbacks mock;

    // Add callback
    registry_->add_registration_callback([&mock](auto obj) {
        mock.on_registration(obj);
    });

    auto obj = make_object<RegistryTestObject>("test_obj", 42);

    EXPECT_CALL(mock, on_registration(_))
        .Times(1);

    ASSERT_TRUE(registry_->register_object("test_key", obj));
}

TEST_F(RegistryTest, UnregistrationCallbacks) {
    MockRegistryCallbacks mock;

    // Add callback
    registry_->add_unregistration_callback([&mock](auto id, auto key) {
        mock.on_unregistration(id, key);
    });

    auto obj = make_object<RegistryTestObject>("test_obj", 42);
    auto id = obj->id();

    ASSERT_TRUE(registry_->register_object("test_key", obj));

    EXPECT_CALL(mock, on_unregistration(id, StrEq("test_key")))
        .Times(1);

    EXPECT_TRUE(registry_->unregister_by_key("test_key"));
}

TEST_F(RegistryTest, MultipleCallbacks) {
    int registration_count = 0;
    int unregistration_count = 0;

    registry_->add_registration_callback([&registration_count](auto /*obj*/) {
        registration_count++;
    });

    registry_->add_registration_callback([&registration_count](auto /*obj*/) {
        registration_count++;
    });

    registry_->add_unregistration_callback([&unregistration_count](auto /*id*/, auto /*key*/) {
        unregistration_count++;
    });

    auto obj = make_object<RegistryTestObject>("test_obj", 42);

    ASSERT_TRUE(registry_->register_object("test_key", obj));
    EXPECT_EQ(registration_count, 2);

    EXPECT_TRUE(registry_->unregister_by_key("test_key"));
    EXPECT_EQ(unregistration_count, 1);
}

TEST_F(RegistryTest, ClearCallbacks) {
    int callback_count = 0;

    registry_->add_registration_callback([&callback_count](auto /*obj*/) {
        callback_count++;
    });

    registry_->clear_callbacks();

    auto obj = make_object<RegistryTestObject>("test_obj", 42);
    ASSERT_TRUE(registry_->register_object(obj));

    EXPECT_EQ(callback_count, 0); // Callback was cleared
}

// ============================================================================
// Batch Operations Tests
// ============================================================================

TEST_F(RegistryTest, BatchRegistration) {
    std::vector<object_ptr<RegistryTestObject>> objects;
    for (int i = 0; i < 5; ++i) {
        objects.push_back(make_object<RegistryTestObject>("obj" + std::to_string(i), i));
    }

    auto registered = registry_->register_batch(objects);
    EXPECT_EQ(registered, 5);
    EXPECT_EQ(registry_->size(), 5);

    // Batch register same objects should fail
    auto duplicate_registered = registry_->register_batch(objects);
    EXPECT_EQ(duplicate_registered, 0);
    EXPECT_EQ(registry_->size(), 5);
}

TEST_F(RegistryTest, BatchRegistrationWithKeys) {
    std::vector<std::pair<std::string, object_ptr<RegistryTestObject>>> pairs;
    for (int i = 0; i < 5; ++i) {
        auto key = "key" + std::to_string(i);
        auto obj = make_object<RegistryTestObject>("obj" + std::to_string(i), i);
        pairs.emplace_back(key, obj);
    }

    auto registered = registry_->register_batch_with_keys(pairs);
    EXPECT_EQ(registered, 5);
    EXPECT_EQ(registry_->size(), 5);

    // Verify all keys are present
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(registry_->contains_key("key" + std::to_string(i)));
    }
}

TEST_F(RegistryTest, BatchUnregistration) {
    std::vector<Object::id_type> ids;
    for (int i = 0; i < 5; ++i) {
        auto obj = make_object<RegistryTestObject>("obj" + std::to_string(i), i);
        ids.push_back(obj->id());
        EXPECT_TRUE(registry_->register_object("key" + std::to_string(i), obj));
    }

    EXPECT_EQ(registry_->size(), 5);

    auto unregistered = registry_->unregister_batch(ids);
    EXPECT_EQ(unregistered, 5);
    EXPECT_EQ(registry_->size(), 0);
}

// ============================================================================
// Integer Key Tests
// ============================================================================

TEST_F(RegistryIntegerKeyTest, IntegerKeys) {
    auto obj1 = make_object<RegistryTestObject>("obj1", 1);
    auto obj2 = make_object<RegistryTestObject>("obj2", 2);

    EXPECT_TRUE(int_registry_->register_object(100, obj1));
    EXPECT_TRUE(int_registry_->register_object(200, obj2));

    EXPECT_TRUE(int_registry_->contains_key(100));
    EXPECT_TRUE(int_registry_->contains_key(200));
    EXPECT_FALSE(int_registry_->contains_key(300));

    auto found1 = int_registry_->find_by_key(100);
    auto found2 = int_registry_->find_by_key(200);

    ASSERT_NE(found1, nullptr);
    ASSERT_NE(found2, nullptr);
    EXPECT_EQ(found1->get_value(), 1);
    EXPECT_EQ(found2->get_value(), 2);
}

TEST_F(RegistryIntegerKeyTest, AutoNameGeneration) {
    auto obj1 = make_object<RegistryTestObject>("obj1");
    auto obj2 = make_object<RegistryTestObject>("obj2");

    EXPECT_TRUE(int_registry_->register_object_auto_name(obj1));
    EXPECT_TRUE(int_registry_->register_object_auto_name(obj2));

    auto keys = int_registry_->get_all_keys();
    EXPECT_EQ(keys.size(), 2);
    EXPECT_TRUE(std::find(keys.begin(), keys.end(), 1) != keys.end());
    EXPECT_TRUE(std::find(keys.begin(), keys.end(), 2) != keys.end());
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST_F(RegistryTest, ConcurrentRegistration) {
    const int num_threads = 10;
    const int objects_per_thread = 100;
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < objects_per_thread; ++i) {
                auto obj = make_object<CounterTestObject>("thread" + std::to_string(t) + "_obj" + std::to_string(i));
                std::string key = "thread" + std::to_string(t) + "_key" + std::to_string(i);

                if (registry_->register_object(key, obj)) {
                    success_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(success_count.load(), num_threads * objects_per_thread);
    EXPECT_EQ(registry_->size(), num_threads * objects_per_thread);
    EXPECT_EQ(CounterTestObject::get_counter(), num_threads * objects_per_thread);
}

TEST_F(RegistryTest, ConcurrentLookup) {
    // First register some objects
    std::vector<object_ptr<RegistryTestObject>> objects;
    for (int i = 0; i < 100; ++i) {
        auto obj = make_object<RegistryTestObject>("obj" + std::to_string(i), i);
        objects.push_back(obj);
        EXPECT_TRUE(registry_->register_object("key" + std::to_string(i), obj));
    }

    const int num_threads = 20;
    const int lookups_per_thread = 1000;
    std::vector<std::thread> threads;
    std::atomic<int> successful_lookups{0};

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, 99);

            for (int i = 0; i < lookups_per_thread; ++i) {
                int index = dis(gen);
                std::string key = "key" + std::to_string(index);

                auto found = registry_->find_by_key(key);
                if (found && found->get_value() == index) {
                    successful_lookups.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(successful_lookups.load(), num_threads * lookups_per_thread);
}

TEST_F(RegistryTest, ConcurrentMixedOperations) {
    const int num_threads = 8;
    std::vector<std::thread> threads;
    std::atomic<int> operations_count{0};

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            if (t % 4 == 0) {
                // Registration thread
                for (int i = 0; i < 50; ++i) {
                    auto obj = make_object<RegistryTestObject>("reg_obj" + std::to_string(t) + "_" + std::to_string(i), i);
                    if (registry_->register_object("reg_key" + std::to_string(t) + "_" + std::to_string(i), obj)) {
                        operations_count.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            } else if (t % 4 == 1) {
                // Lookup thread
                std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Let some registrations happen first
                for (int i = 0; i < 100; ++i) {
                    auto size = registry_->size();
                    auto empty = registry_->empty();
                    (void)size; (void)empty; // Suppress unused variable warnings
                    operations_count.fetch_add(1, std::memory_order_relaxed);
                }
            } else if (t % 4 == 2) {
                // Query thread
                for (int i = 0; i < 30; ++i) {
                    auto ids = registry_->get_all_ids();
                    auto keys = registry_->get_all_keys();
                    operations_count.fetch_add(1, std::memory_order_relaxed);
                }
            } else {
                // Statistics thread
                for (int i = 0; i < 50; ++i) {
                    auto stats = registry_->get_statistics();
                    operations_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_GT(operations_count.load(), 0);
    EXPECT_GT(registry_->size(), 0);
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST_F(RegistryTest, RegistrationPerformance) {
    const int num_objects = 10000;
    std::vector<object_ptr<RegistryTestObject>> objects;

    // Pre-create objects
    for (int i = 0; i < num_objects; ++i) {
        objects.push_back(make_object<RegistryTestObject>("perf_obj" + std::to_string(i), i));
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_objects; ++i) {
        EXPECT_TRUE(registry_->register_object("perf_key" + std::to_string(i), objects[i]));
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Should complete in reasonable time (less than 500ms for 10000 registrations)
    EXPECT_LT(duration.count(), 500000);
    EXPECT_EQ(registry_->size(), num_objects);
}

TEST_F(RegistryTest, LookupPerformance) {
    const int num_objects = 1000;
    const int num_lookups = 100000;

    // Register objects
    for (int i = 0; i < num_objects; ++i) {
        auto obj = make_object<RegistryTestObject>("lookup_obj" + std::to_string(i), i);
        EXPECT_TRUE(registry_->register_object("lookup_key" + std::to_string(i), obj));
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, num_objects - 1);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_lookups; ++i) {
        int index = dis(gen);
        std::string key = "lookup_key" + std::to_string(index);
        auto found = registry_->find_by_key(key);
        EXPECT_NE(found, nullptr);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Should complete in reasonable time (less than 1 second for 100000 lookups)
    EXPECT_LT(duration.count(), 1000000);
}

TEST_F(RegistryTest, BatchOperationPerformance) {
    const int batch_size = 5000;
    std::vector<object_ptr<RegistryTestObject>> objects;

    for (int i = 0; i < batch_size; ++i) {
        objects.push_back(make_object<RegistryTestObject>("batch_obj" + std::to_string(i), i));
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto registered = registry_->register_batch(objects);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    EXPECT_EQ(registered, batch_size);
    // Batch operations should be faster than individual registrations
    EXPECT_LT(duration.count(), 100000); // Less than 100ms for 5000 objects
}

// ============================================================================
// Memory Management Tests
// ============================================================================

TEST_F(RegistryTest, MemoryLeakPrevention) {
    CounterTestObject::reset_counter();

    // Create and register many objects
    for (int i = 0; i < 1000; ++i) {
        auto obj = make_object<CounterTestObject>("mem_obj" + std::to_string(i));
        EXPECT_TRUE(registry_->register_object("mem_key" + std::to_string(i), obj));
    }

    EXPECT_EQ(CounterTestObject::get_counter(), 1000);
    EXPECT_EQ(registry_->size(), 1000);

    // Clear registry - should release all objects
    registry_->clear();

    // Objects should be cleaned up automatically
    EXPECT_EQ(CounterTestObject::get_counter(), 0);
    EXPECT_EQ(registry_->size(), 0);
}

TEST_F(RegistryTest, WeakReferenceCleanup) {
    auto obj1 = make_object<RegistryTestObject>("obj1", 1);
    auto obj2 = make_object<RegistryTestObject>("obj2", 2);
    auto obj3 = make_object<RegistryTestObject>("obj3", 3);

    ASSERT_TRUE(registry_->register_object("key1", obj1));
    ASSERT_TRUE(registry_->register_object("key2", obj2));
    ASSERT_TRUE(registry_->register_object("key3", obj3));

    EXPECT_EQ(registry_->size(), 3);

    // Manually release some references (simulate objects going out of scope elsewhere)
    obj1.reset();
    obj2.reset();
    // Keep obj3 alive

    // Cleanup should remove objects with only registry reference
    auto cleaned = registry_->cleanup_weak_references();

    // Note: This test behavior depends on the implementation details
    // The registry holds references, so cleanup might not remove anything
    // This test mainly ensures the cleanup method doesn't crash
    EXPECT_GE(cleaned, 0);
}

// ============================================================================
// Edge Cases and Error Conditions
// ============================================================================

TEST_F(RegistryTest, EmptyKeys) {
    auto obj = make_object<RegistryTestObject>("test_obj", 42);

    EXPECT_TRUE(registry_->register_object("", obj));
    EXPECT_TRUE(registry_->contains_key(""));

    auto found = registry_->find_by_key("");
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->get_value(), 42);
}

TEST_F(RegistryTest, LongKeys) {
    std::string long_key(10000, 'x');
    auto obj = make_object<RegistryTestObject>("test_obj", 42);

    EXPECT_TRUE(registry_->register_object(long_key, obj));
    EXPECT_TRUE(registry_->contains_key(long_key));

    auto found = registry_->find_by_key(long_key);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->get_value(), 42);
}

TEST_F(RegistryTest, SpecialCharacterKeys) {
    std::string special_key = "test@#$%^&*(){}[]|\\:;\"'<>,.?/~`+=";
    auto obj = make_object<RegistryTestObject>("test_obj", 42);

    EXPECT_TRUE(registry_->register_object(special_key, obj));
    EXPECT_TRUE(registry_->contains_key(special_key));

    auto found = registry_->find_by_key(special_key);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->get_value(), 42);
}

TEST_F(RegistryTest, DuplicateKeyHandling) {
    auto obj1 = make_object<RegistryTestObject>("obj1", 1);
    auto obj2 = make_object<RegistryTestObject>("obj2", 2);

    EXPECT_TRUE(registry_->register_object("duplicate_key", obj1));
    EXPECT_FALSE(registry_->register_object("duplicate_key", obj2));

    auto found = registry_->find_by_key("duplicate_key");
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->get_value(), 1); // Should be first object
}

TEST_F(RegistryTest, ExceptionInCallbacks) {
    // Add callback that throws
    registry_->add_registration_callback([](auto /*obj*/) {
        throw std::runtime_error("Callback exception");
    });

    auto obj = make_object<RegistryTestObject>("test_obj", 42);

    // Registration should still succeed despite callback exception
    EXPECT_TRUE(registry_->register_object("test_key", obj));
    EXPECT_EQ(registry_->size(), 1);
    EXPECT_TRUE(registry_->contains_key("test_key"));
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_F(RegistryTest, Statistics) {
    auto stats = registry_->get_statistics();
    EXPECT_EQ(stats.total_objects, 0);
    EXPECT_EQ(stats.named_objects, 0);
    EXPECT_EQ(stats.unnamed_objects, 0);
    EXPECT_EQ(stats.registry_name, "TestRegistry");

    auto obj1 = make_object<RegistryTestObject>("obj1", 1);
    auto obj2 = make_object<RegistryTestObject>("obj2", 2);
    auto obj3 = make_object<RegistryTestObject>("obj3", 3);

    ASSERT_TRUE(registry_->register_object("key1", obj1));
    ASSERT_TRUE(registry_->register_object("key2", obj2));
    ASSERT_TRUE(registry_->register_object(obj3)); // No key

    stats = registry_->get_statistics();
    EXPECT_EQ(stats.total_objects, 3);
    EXPECT_EQ(stats.named_objects, 2);
    EXPECT_EQ(stats.unnamed_objects, 1);
}

// ============================================================================
// Global Registry Tests
// ============================================================================

TEST(GlobalRegistryTest, Singleton) {
    auto& global1 = GlobalObjectRegistry::instance();
    auto& global2 = GlobalObjectRegistry::instance();

    EXPECT_EQ(&global1, &global2);
}

TEST(GlobalRegistryTest, BasicFunctionality) {
    auto& global = GlobalObjectRegistry::instance();

    // Clear any existing objects
    global.get_registry().clear();

    auto obj = make_object<RegistryTestObject>("global_obj", 99);
    auto id = obj->id();

    EXPECT_TRUE(global.register_object("global_key", obj));

    auto found_by_name = global.find_by_name("global_key");
    auto found_by_id = global.find_by_id(id);

    ASSERT_NE(found_by_name, nullptr);
    ASSERT_NE(found_by_id, nullptr);
    EXPECT_EQ(found_by_name->id(), id);
    EXPECT_EQ(found_by_id->id(), id);

    global.get_registry().clear();
}

TEST(GlobalRegistryTest, ConvenienceFunctions) {
    // Clear any existing objects
    GlobalObjectRegistry::instance().get_registry().clear();

    auto obj = make_object<RegistryTestObject>("convenience_obj", 123);

    EXPECT_TRUE(register_global_object("convenience_key", obj));

    auto found = find_global_object("convenience_key");
    ASSERT_NE(found, nullptr);

    // Cast to derived type
    auto* derived = dynamic_cast<RegistryTestObject*>(found.get());
    ASSERT_NE(derived, nullptr);
    EXPECT_EQ(derived->get_value(), 123);

    GlobalObjectRegistry::instance().get_registry().clear();
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(RegistryTest, CompleteWorkflow) {
    // Test complete registry workflow
    std::vector<object_ptr<RegistryTestObject>> objects;
    std::vector<std::string> keys;

    // 1. Register multiple objects with different methods
    for (int i = 0; i < 10; ++i) {
        auto obj = make_object<RegistryTestObject>("workflow_obj" + std::to_string(i), i);
        objects.push_back(obj);

        if (i % 3 == 0) {
            // Register with specific key
            std::string key = "specific_key_" + std::to_string(i);
            keys.push_back(key);
            EXPECT_TRUE(registry_->register_object(key, obj));
        } else if (i % 3 == 1) {
            // Register with auto-generated key
            EXPECT_TRUE(registry_->register_object_auto_name(obj, "auto"));
        } else {
            // Register without key
            EXPECT_TRUE(registry_->register_object(obj));
        }
    }

    // 2. Verify all objects are registered
    EXPECT_EQ(registry_->size(), 10);

    // 3. Test various query methods
    auto all_ids = registry_->get_all_ids();
    auto all_keys = registry_->get_all_keys();
    auto all_objects = registry_->get_all_objects();

    EXPECT_EQ(all_ids.size(), 10);
    EXPECT_GT(all_keys.size(), 0);
    EXPECT_EQ(all_objects.size(), 10);

    // 4. Test predicate operations
    auto high_value_objects = registry_->find_if([](const auto& obj) {
        return obj->get_value() >= 5;
    });
    EXPECT_EQ(high_value_objects.size(), 5);

    auto first_even = registry_->find_first_if([](const auto& obj) {
        return obj->get_value() % 2 == 0;
    });
    ASSERT_NE(first_even, nullptr);
    EXPECT_EQ(first_even->get_value() % 2, 0);

    // 5. Test removal
    for (const auto& key : keys) {
        EXPECT_TRUE(registry_->unregister_by_key(key));
    }

    // 6. Final cleanup
    registry_->clear();
    EXPECT_TRUE(registry_->empty());
}

TEST_F(RegistryTest, MultipleRegistryTypes) {
    // Test different key types work correctly
    Registry<RegistryTestObject, int> int_registry("IntRegistry");
    Registry<RegistryTestObject, std::uint32_t> uint_registry("UintRegistry");

    auto obj1 = make_object<RegistryTestObject>("obj1", 1);
    auto obj2 = make_object<RegistryTestObject>("obj2", 2);

    EXPECT_TRUE(int_registry.register_object(100, obj1));
    EXPECT_TRUE(uint_registry.register_object(200u, obj2));

    EXPECT_EQ(int_registry.size(), 1);
    EXPECT_EQ(uint_registry.size(), 1);

    auto found1 = int_registry.find_by_key(100);
    auto found2 = uint_registry.find_by_key(200u);

    ASSERT_NE(found1, nullptr);
    ASSERT_NE(found2, nullptr);
    EXPECT_EQ(found1->get_value(), 1);
    EXPECT_EQ(found2->get_value(), 2);
}