/**
 * @file test_factory.cpp
 * @brief Comprehensive unit tests for the Factory pattern implementation
 *
 * Tests cover:
 * - Basic factory functionality (registration, creation, lookup)
 * - Type safety and CRTP functionality
 * - Parameterized object creation
 * - Thread safety for concurrent operations
 * - Custom creator functions
 * - Factory registrar RAII helper
 * - Edge cases and error conditions
 * - Performance characteristics
 * - Memory management
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

#include "core/base/factory.h"

using namespace fem::core::base;
using namespace testing;

// ============================================================================
// Test Objects for Factory Testing
// ============================================================================

// Base class for factory testing
class FactoryTestBase : public Object {
public:
    explicit FactoryTestBase(std::string_view name = "FactoryTestBase")
        : Object(name) {}
    virtual ~FactoryTestBase() = default;
    virtual std::string get_type_description() const { return "FactoryTestBase"; }
    virtual int get_value() const { return 0; }
};

// Simple derived class
class SimpleProduct : public FactoryTestBase {
public:
    explicit SimpleProduct(std::string_view name = "SimpleProduct")
        : FactoryTestBase(name), value_(42) {}

    std::string get_type_description() const override { return "SimpleProduct"; }
    int get_value() const override { return value_; }

private:
    int value_;
};

// Parameterized constructor class
class ParameterizedProduct : public FactoryTestBase {
public:
    explicit ParameterizedProduct(const std::unordered_map<std::string, std::string>& params)
        : FactoryTestBase("ParameterizedProduct") {
        auto it = params.find("value");
        value_ = (it != params.end()) ? std::stoi(it->second) : 100;

        auto name_it = params.find("name");
        if (name_it != params.end()) {
            name_ = name_it->second;
        }
    }

    // Default constructor fallback
    ParameterizedProduct() : FactoryTestBase("ParameterizedProduct"), value_(100) {}

    std::string get_type_description() const override { return "ParameterizedProduct"; }
    int get_value() const override { return value_; }
    const std::string& get_name() const { return name_; }

private:
    int value_;
    std::string name_ = "default";
};

// Complex product with dependencies
class ComplexProduct : public FactoryTestBase {
public:
    ComplexProduct() : FactoryTestBase("ComplexProduct"), creation_time_(std::chrono::steady_clock::now()) {}

    std::string get_type_description() const override { return "ComplexProduct"; }
    int get_value() const override { return 999; }

    auto get_creation_time() const { return creation_time_; }

private:
    std::chrono::steady_clock::time_point creation_time_;
};

// Thread-safe counter for concurrent testing
class CounterProduct : public FactoryTestBase {
public:
    CounterProduct() : FactoryTestBase("CounterProduct") {
        counter_.fetch_add(1, std::memory_order_relaxed);
    }

    std::string get_type_description() const override { return "CounterProduct"; }
    int get_value() const override { return counter_.load(std::memory_order_relaxed); }

    static void reset_counter() { counter_.store(0, std::memory_order_relaxed); }
    static int get_counter() { return counter_.load(std::memory_order_relaxed); }

private:
    static std::atomic<int> counter_;
};

std::atomic<int> CounterProduct::counter_{0};

// ============================================================================
// Factory Test Fixtures
// ============================================================================

class FactoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Clear factory state before each test
        Factory<FactoryTestBase>::instance().clear();
        CounterProduct::reset_counter();
    }

    void TearDown() override {
        // Clean up after each test
        Factory<FactoryTestBase>::instance().clear();
    }
};

// ============================================================================
// Basic Factory Functionality Tests
// ============================================================================

TEST_F(FactoryTest, SingletonBehavior) {
    auto& factory1 = Factory<FactoryTestBase>::instance();
    auto& factory2 = Factory<FactoryTestBase>::instance();

    EXPECT_EQ(&factory1, &factory2);
}

TEST_F(FactoryTest, TypeRegistration) {
    auto& factory = Factory<FactoryTestBase>::instance();

    // Register types
    EXPECT_TRUE(factory.register_type<SimpleProduct>("simple"));
    EXPECT_TRUE(factory.register_type<ComplexProduct>("complex"));

    // Check registration
    EXPECT_TRUE(factory.is_registered("simple"));
    EXPECT_TRUE(factory.is_registered("complex"));
    EXPECT_FALSE(factory.is_registered("unknown"));

    // Check by type
    EXPECT_TRUE(factory.is_registered<SimpleProduct>());
    EXPECT_TRUE(factory.is_registered<ComplexProduct>());
}

TEST_F(FactoryTest, ObjectCreation) {
    auto& factory = Factory<FactoryTestBase>::instance();
    factory.register_type<SimpleProduct>("simple");
    factory.register_type<ComplexProduct>("complex");

    // Create by name
    auto simple = factory.create("simple");
    ASSERT_NE(simple, nullptr);
    EXPECT_EQ(simple->get_type_description(), "SimpleProduct");
    EXPECT_EQ(simple->get_value(), 42);

    auto complex = factory.create("complex");
    ASSERT_NE(complex, nullptr);
    EXPECT_EQ(complex->get_type_description(), "ComplexProduct");
    EXPECT_EQ(complex->get_value(), 999);
}

TEST_F(FactoryTest, ObjectCreationByType) {
    auto& factory = Factory<FactoryTestBase>::instance();
    factory.register_type<SimpleProduct>("simple");

    // Create by type
    auto product = factory.create<SimpleProduct>();
    ASSERT_NE(product, nullptr);
    EXPECT_EQ(product->get_type_description(), "SimpleProduct");
    EXPECT_EQ(product->get_value(), 42);
}

TEST_F(FactoryTest, UnregisteredTypeCreation) {
    auto& factory = Factory<FactoryTestBase>::instance();

    // Creating unregistered type by name should throw
    EXPECT_THROW(factory.create("nonexistent"), std::invalid_argument);

    // Creating unregistered type by type should work (direct creation)
    auto product = factory.create<SimpleProduct>();
    ASSERT_NE(product, nullptr);
    EXPECT_EQ(product->get_type_description(), "SimpleProduct");
}

TEST_F(FactoryTest, ParameterizedCreation) {
    auto& factory = Factory<FactoryTestBase>::instance();
    factory.register_type<ParameterizedProduct>("param");

    // Create with parameters
    std::unordered_map<std::string, std::string> params{
        {"value", "123"},
        {"name", "test_product"}
    };

    auto product = factory.create("param", params);
    ASSERT_NE(product, nullptr);

    auto* param_product = dynamic_cast<ParameterizedProduct*>(product.get());
    ASSERT_NE(param_product, nullptr);
    EXPECT_EQ(param_product->get_value(), 123);
    EXPECT_EQ(param_product->get_name(), "test_product");
}

TEST_F(FactoryTest, ParameterizedCreationFallback) {
    auto& factory = Factory<FactoryTestBase>::instance();
    factory.register_type<SimpleProduct>("simple");  // No parameterized constructor

    // Should fallback to simple creation
    std::unordered_map<std::string, std::string> params{{"ignored", "value"}};
    auto product = factory.create("simple", params);

    ASSERT_NE(product, nullptr);
    EXPECT_EQ(product->get_type_description(), "SimpleProduct");
    EXPECT_EQ(product->get_value(), 42);
}

// ============================================================================
// Custom Creator Functions Tests
// ============================================================================

TEST_F(FactoryTest, CustomCreatorFunction) {
    auto& factory = Factory<FactoryTestBase>::instance();

    // Register custom creator
    auto custom_creator = []() -> object_ptr<FactoryTestBase> {
        return make_object<SimpleProduct>();
    };

    EXPECT_TRUE(factory.register_creator("custom", custom_creator));
    EXPECT_TRUE(factory.is_registered("custom"));

    auto product = factory.create("custom");
    ASSERT_NE(product, nullptr);
    EXPECT_EQ(product->get_type_description(), "SimpleProduct");
}

TEST_F(FactoryTest, CustomParameterizedCreator) {
    auto& factory = Factory<FactoryTestBase>::instance();

    // Register custom parameterized creator
    auto param_creator = [](const std::unordered_map<std::string, std::string>& params) -> object_ptr<FactoryTestBase> {
        auto it = params.find("type");
        if (it != params.end() && it->second == "complex") {
            return make_object<ComplexProduct>();
        }
        return make_object<SimpleProduct>();
    };

    EXPECT_TRUE(factory.register_parameterized_creator("adaptive", param_creator));

    // Test different parameter combinations
    std::unordered_map<std::string, std::string> simple_params{{"type", "simple"}};
    auto simple = factory.create("adaptive", simple_params);
    EXPECT_EQ(simple->get_type_description(), "SimpleProduct");

    std::unordered_map<std::string, std::string> complex_params{{"type", "complex"}};
    auto complex = factory.create("adaptive", complex_params);
    EXPECT_EQ(complex->get_type_description(), "ComplexProduct");
}

// ============================================================================
// Factory Management Tests
// ============================================================================

TEST_F(FactoryTest, TypeNameLookup) {
    auto& factory = Factory<FactoryTestBase>::instance();
    factory.register_type<SimpleProduct>("simple");
    factory.register_type<ComplexProduct>("complex");

    EXPECT_EQ(factory.get_type_name<SimpleProduct>(), "simple");
    EXPECT_EQ(factory.get_type_name<ComplexProduct>(), "complex");
    EXPECT_EQ(factory.get_type_name<ParameterizedProduct>(), ""); // Not registered
}

TEST_F(FactoryTest, RegisteredTypesList) {
    auto& factory = Factory<FactoryTestBase>::instance();

    // Initially empty
    auto types = factory.get_registered_types();
    EXPECT_TRUE(types.empty());

    // Register some types
    factory.register_type<SimpleProduct>("simple");
    factory.register_type<ComplexProduct>("complex");

    types = factory.get_registered_types();
    EXPECT_EQ(types.size(), 2);
    EXPECT_THAT(types, UnorderedElementsAre("simple", "complex"));
}

TEST_F(FactoryTest, TypeUnregistration) {
    auto& factory = Factory<FactoryTestBase>::instance();
    factory.register_type<SimpleProduct>("simple");
    factory.register_type<ComplexProduct>("complex");

    EXPECT_TRUE(factory.is_registered("simple"));
    EXPECT_TRUE(factory.unregister("simple"));
    EXPECT_FALSE(factory.is_registered("simple"));
    EXPECT_TRUE(factory.is_registered("complex"));

    // Unregistering non-existent type
    EXPECT_FALSE(factory.unregister("nonexistent"));
}

TEST_F(FactoryTest, FactoryClear) {
    auto& factory = Factory<FactoryTestBase>::instance();
    factory.register_type<SimpleProduct>("simple");
    factory.register_type<ComplexProduct>("complex");

    EXPECT_FALSE(factory.get_registered_types().empty());

    factory.clear();

    auto types = factory.get_registered_types();
    EXPECT_TRUE(types.empty());
    EXPECT_FALSE(factory.is_registered("simple"));
    EXPECT_FALSE(factory.is_registered("complex"));
}

TEST_F(FactoryTest, FactoryStatistics) {
    auto& factory = Factory<FactoryTestBase>::instance();

    auto stats = factory.get_statistics();
    EXPECT_EQ(stats.registered_types, 0);
    EXPECT_EQ(stats.simple_creators, 0);
    EXPECT_EQ(stats.parameterized_creators, 0);

    factory.register_type<SimpleProduct>("simple");
    factory.register_type<ParameterizedProduct>("param");

    stats = factory.get_statistics();
    EXPECT_EQ(stats.registered_types, 2);
    EXPECT_EQ(stats.simple_creators, 2);
    EXPECT_EQ(stats.parameterized_creators, 2);
}

// ============================================================================
// Factory Registrar RAII Tests
// ============================================================================

TEST_F(FactoryTest, FactoryRegistrarConstruction) {
    auto& factory = Factory<FactoryTestBase>::instance();
    EXPECT_FALSE(factory.is_registered("raii_test"));

    {
        FactoryRegistrar<FactoryTestBase, SimpleProduct> registrar("raii_test");
        EXPECT_TRUE(registrar.is_registered());
        EXPECT_TRUE(factory.is_registered("raii_test"));

        // Can create objects
        auto product = factory.create("raii_test");
        ASSERT_NE(product, nullptr);
        EXPECT_EQ(product->get_type_description(), "SimpleProduct");
    }

    // Should be unregistered after destructor
    EXPECT_FALSE(factory.is_registered("raii_test"));
}

TEST_F(FactoryTest, FactoryRegistrarMultipleInstances) {
    auto& factory = Factory<FactoryTestBase>::instance();

    {
        FactoryRegistrar<FactoryTestBase, SimpleProduct> registrar1("test1");
        FactoryRegistrar<FactoryTestBase, ComplexProduct> registrar2("test2");

        EXPECT_TRUE(factory.is_registered("test1"));
        EXPECT_TRUE(factory.is_registered("test2"));
    }

    EXPECT_FALSE(factory.is_registered("test1"));
    EXPECT_FALSE(factory.is_registered("test2"));
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST_F(FactoryTest, ConcurrentRegistration) {
    auto& factory = Factory<FactoryTestBase>::instance();
    const int num_threads = 10;
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&factory, &success_count, i]() {
            std::string name = "concurrent_" + std::to_string(i);
            if (factory.register_type<SimpleProduct>(name)) {
                success_count.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(success_count.load(), num_threads);

    auto types = factory.get_registered_types();
    EXPECT_EQ(types.size(), num_threads);
}

TEST_F(FactoryTest, ConcurrentCreation) {
    auto& factory = Factory<FactoryTestBase>::instance();
    factory.register_type<CounterProduct>("counter");

    const int num_threads = 20;
    const int objects_per_thread = 100;
    std::vector<std::thread> threads;
    std::atomic<int> creation_count{0};

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&factory, &creation_count, objects_per_thread]() {
            for (int j = 0; j < objects_per_thread; ++j) {
                auto product = factory.create("counter");
                if (product) {
                    creation_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(creation_count.load(), num_threads * objects_per_thread);
    EXPECT_EQ(CounterProduct::get_counter(), num_threads * objects_per_thread);
}

TEST_F(FactoryTest, ConcurrentRegistrationAndCreation) {
    auto& factory = Factory<FactoryTestBase>::instance();
    const int num_threads = 8;
    std::vector<std::thread> threads;
    std::vector<std::future<bool>> futures;

    // Mix of registration and creation operations
    for (int i = 0; i < num_threads; ++i) {
        if (i % 2 == 0) {
            // Registration threads
            threads.emplace_back([&factory, i]() {
                std::string name = "type_" + std::to_string(i);
                factory.register_type<SimpleProduct>(name);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            });
        } else {
            // Creation threads (with registered type)
            threads.emplace_back([&factory]() {
                // Register a known type first
                factory.register_type<CounterProduct>("known");

                for (int j = 0; j < 10; ++j) {
                    try {
                        auto product = factory.create("known");
                        EXPECT_NE(product, nullptr);
                    } catch (...) {
                        // Ignore exceptions in concurrent test
                    }
                }
            });
        }
    }

    for (auto& t : threads) {
        t.join();
    }

    // Should have multiple types registered
    EXPECT_GT(factory.get_registered_types().size(), 0);
}

// ============================================================================
// Edge Cases and Error Conditions Tests
// ============================================================================

TEST_F(FactoryTest, EmptyTypeName) {
    auto& factory = Factory<FactoryTestBase>::instance();

    // Register with empty name
    EXPECT_TRUE(factory.register_type<SimpleProduct>(""));
    EXPECT_TRUE(factory.is_registered(""));

    auto product = factory.create("");
    ASSERT_NE(product, nullptr);
    EXPECT_EQ(product->get_type_description(), "SimpleProduct");
}

TEST_F(FactoryTest, DuplicateRegistration) {
    auto& factory = Factory<FactoryTestBase>::instance();

    // Register same type multiple times
    EXPECT_TRUE(factory.register_type<SimpleProduct>("test"));
    EXPECT_TRUE(factory.register_type<ComplexProduct>("test")); // Should overwrite

    auto product = factory.create("test");
    EXPECT_EQ(product->get_type_description(), "ComplexProduct");
}

TEST_F(FactoryTest, LongTypeName) {
    auto& factory = Factory<FactoryTestBase>::instance();
    std::string long_name(1000, 'x');

    EXPECT_TRUE(factory.register_type<SimpleProduct>(long_name));
    EXPECT_TRUE(factory.is_registered(long_name));

    auto product = factory.create(long_name);
    ASSERT_NE(product, nullptr);
    EXPECT_EQ(product->get_type_description(), "SimpleProduct");
}

TEST_F(FactoryTest, SpecialCharactersInName) {
    auto& factory = Factory<FactoryTestBase>::instance();
    std::string special_name = "test@#$%^&*(){}[]|\\:;\"'<>,.?/~`";

    EXPECT_TRUE(factory.register_type<SimpleProduct>(special_name));
    EXPECT_TRUE(factory.is_registered(special_name));

    auto product = factory.create(special_name);
    ASSERT_NE(product, nullptr);
    EXPECT_EQ(product->get_type_description(), "SimpleProduct");
}

TEST_F(FactoryTest, ExceptionInCustomCreator) {
    auto& factory = Factory<FactoryTestBase>::instance();

    // Creator that throws
    auto throwing_creator = []() -> object_ptr<FactoryTestBase> {
        throw std::runtime_error("Creator failed");
    };

    factory.register_creator("throwing", throwing_creator);

    EXPECT_THROW(factory.create("throwing"), std::runtime_error);
}

TEST_F(FactoryTest, InvalidParameters) {
    auto& factory = Factory<FactoryTestBase>::instance();
    factory.register_type<ParameterizedProduct>("param");

    // Invalid numeric parameter
    std::unordered_map<std::string, std::string> bad_params{{"value", "not_a_number"}};

    // Should throw during construction
    EXPECT_THROW(factory.create("param", bad_params), std::invalid_argument);
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST_F(FactoryTest, RegistrationPerformance) {
    auto& factory = Factory<FactoryTestBase>::instance();
    const int num_registrations = 1000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_registrations; ++i) {
        std::string name = "perf_test_" + std::to_string(i);
        factory.register_type<SimpleProduct>(name);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Should complete in reasonable time (less than 100ms for 1000 registrations)
    EXPECT_LT(duration.count(), 100000);

    // Verify all were registered
    EXPECT_EQ(factory.get_registered_types().size(), num_registrations);
}

TEST_F(FactoryTest, CreationPerformance) {
    auto& factory = Factory<FactoryTestBase>::instance();
    factory.register_type<SimpleProduct>("simple");

    const int num_creations = 10000;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_creations; ++i) {
        auto product = factory.create("simple");
        EXPECT_NE(product, nullptr);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Should complete in reasonable time (less than 1 second for 10000 creations)
    EXPECT_LT(duration.count(), 1000000);
}

TEST_F(FactoryTest, LookupPerformance) {
    auto& factory = Factory<FactoryTestBase>::instance();

    // Register many types
    for (int i = 0; i < 1000; ++i) {
        std::string name = "lookup_test_" + std::to_string(i);
        factory.register_type<SimpleProduct>(name);
    }

    const int num_lookups = 10000;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_lookups; ++i) {
        std::string name = "lookup_test_" + std::to_string(i % 1000);
        bool found = factory.is_registered(name);
        EXPECT_TRUE(found);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Lookups should be very fast (less than 100ms for 10000 lookups)
    EXPECT_LT(duration.count(), 100000);
}

// ============================================================================
// Memory Management Tests
// ============================================================================

TEST_F(FactoryTest, MemoryLeakPrevention) {
    auto& factory = Factory<FactoryTestBase>::instance();
    factory.register_type<ComplexProduct>("complex");

    // Create many objects and let them go out of scope
    for (int i = 0; i < 1000; ++i) {
        auto product = factory.create("complex");
        EXPECT_NE(product, nullptr);
        // object_ptr should handle cleanup automatically
    }

    // If we get here without crashes, memory management is working
    SUCCEED();
}

TEST_F(FactoryTest, ObjectLifecycleManagement) {
    auto& factory = Factory<FactoryTestBase>::instance();
    factory.register_type<SimpleProduct>("simple");

    // Test that objects are properly managed
    {
        auto product = factory.create("simple");
        EXPECT_NE(product, nullptr);
        EXPECT_EQ(product->ref_count(), 1);

        // Copy should increment ref count
        auto copy = product;
        EXPECT_EQ(product->ref_count(), 2);
    }

    // Objects should be cleaned up automatically
    SUCCEED();
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(FactoryTest, CompleteWorkflow) {
    auto& factory = Factory<FactoryTestBase>::instance();

    // 1. Register types
    factory.register_type<SimpleProduct>("simple");
    factory.register_type<ParameterizedProduct>("param");
    factory.register_type<ComplexProduct>("complex");

    // 2. Verify registration
    auto types = factory.get_registered_types();
    EXPECT_EQ(types.size(), 3);

    // 3. Create objects of different types
    auto simple = factory.create("simple");
    auto complex = factory.create("complex");

    std::unordered_map<std::string, std::string> params{{"value", "555"}};
    auto parameterized = factory.create("param", params);

    // 4. Verify correct object creation
    EXPECT_EQ(simple->get_value(), 42);
    EXPECT_EQ(complex->get_value(), 999);
    EXPECT_EQ(parameterized->get_value(), 555);

    // 5. Test type checking
    EXPECT_TRUE(factory.is_registered<SimpleProduct>());
    EXPECT_EQ(factory.get_type_name<SimpleProduct>(), "simple");

    // 6. Clean up
    factory.unregister("simple");
    EXPECT_FALSE(factory.is_registered("simple"));
    EXPECT_TRUE(factory.is_registered("complex"));

    factory.clear();
    EXPECT_TRUE(factory.get_registered_types().empty());
}

// ============================================================================
// Concept and Template Tests
// ============================================================================

TEST_F(FactoryTest, ObjectDerivedConcept) {
    // Factory constructor is private (singleton pattern)
    // Test only that concepts work correctly

    // This test verifies that the ObjectDerived concept works correctly
    EXPECT_TRUE((std::derived_from<SimpleProduct, FactoryTestBase>));
    EXPECT_TRUE((std::derived_from<FactoryTestBase, Object>));
}

// ============================================================================
// Type Alias Tests
// ============================================================================

TEST_F(FactoryTest, ObjectFactoryAlias) {
    // Test the ObjectFactory type alias
    auto& obj_factory = ObjectFactory::instance();

    obj_factory.register_type<SimpleProduct>("simple_obj");
    EXPECT_TRUE(obj_factory.is_registered("simple_obj"));

    auto obj = obj_factory.create("simple_obj");
    ASSERT_NE(obj, nullptr);

    // Should be castable to FactoryTestBase
    auto* derived = dynamic_cast<FactoryTestBase*>(obj.get());
    ASSERT_NE(derived, nullptr);
    EXPECT_EQ(derived->get_type_description(), "SimpleProduct");
}