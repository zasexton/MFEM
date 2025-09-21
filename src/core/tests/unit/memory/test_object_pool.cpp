#include <gtest/gtest.h>
#include <core/memory/object_pool.h>
#include <core/memory/memory_resource.h>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <algorithm>
#include <random>
#include <atomic>
#include <cstring>

namespace fcm = fem::core::memory;

// Test object with instrumentation - must be defined before ObjectPoolTest
struct TestObject {
    int id;
    double value;
    std::string name;
    bool* destroyed_flag;

    static std::atomic<int> construct_count;
    static std::atomic<int> destruct_count;
    static std::atomic<int> alive_count;

    TestObject(int i = 0, double v = 0.0, const std::string& n = "")
        : id(i), value(v), name(n), destroyed_flag(nullptr) {
        ++construct_count;
        ++alive_count;
    }

    TestObject(bool* flag)
        : id(0), value(0.0), name(""), destroyed_flag(flag) {
        ++construct_count;
        ++alive_count;
        *destroyed_flag = false;
    }

    ~TestObject() {
        ++destruct_count;
        --alive_count;
        if (destroyed_flag) {
            *destroyed_flag = true;
        }
    }

    static void reset_counts() {
        construct_count = 0;
        destruct_count = 0;
        alive_count = 0;
    }
};

std::atomic<int> TestObject::construct_count{0};
std::atomic<int> TestObject::destruct_count{0};
std::atomic<int> TestObject::alive_count{0};

// Test fixture class
class ObjectPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        resource_ = fcm::new_delete_resource();
        TestObject::reset_counts();
    }

    void TearDown() override {
        TestObject::reset_counts();
    }

    fcm::memory_resource* resource_ = nullptr;
};

// Simple POD type
struct PodType {
    int x;
    double y;
    char data[64];
};

// Basic functionality tests
TEST_F(ObjectPoolTest, BasicConstruction) {
    fcm::ObjectPool<TestObject> pool(resource_);

    EXPECT_EQ(pool.outstanding(), 0);
}

TEST_F(ObjectPoolTest, AcquireRelease) {
    fcm::ObjectPool<TestObject> pool(resource_);

    auto obj = pool.acquire(42, 3.14, "test");
    ASSERT_NE(obj, nullptr);
    EXPECT_EQ(obj->id, 42);
    EXPECT_DOUBLE_EQ(obj->value, 3.14);
    EXPECT_EQ(obj->name, "test");
    EXPECT_EQ(pool.outstanding(), 1);

    obj.reset();  // Release back to pool
    EXPECT_EQ(pool.outstanding(), 0);
}

TEST_F(ObjectPoolTest, MultipleAcquires) {
    fcm::ObjectPool<TestObject> pool(resource_);

    std::vector<fcm::ObjectPool<TestObject>::handle> objects;

    for (int i = 0; i < 10; ++i) {
        auto obj = pool.acquire(i, static_cast<double>(i) * 1.5, "obj_" + std::to_string(i));
        ASSERT_NE(obj, nullptr);
        objects.push_back(std::move(obj));
    }

    EXPECT_EQ(pool.outstanding(), 10);

    // Verify objects
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(objects[i]->id, i);
        EXPECT_DOUBLE_EQ(objects[i]->value, static_cast<double>(i) * 1.5);
        EXPECT_EQ(objects[i]->name, "obj_" + std::to_string(i));
    }

    // Release all
    objects.clear();
    EXPECT_EQ(pool.outstanding(), 0);
}

TEST_F(ObjectPoolTest, ReuseAfterRelease) {
    fcm::ObjectPool<TestObject> pool(resource_);

    TestObject::reset_counts();

    auto obj1 = pool.acquire(1, 1.0, "first");
    void* addr1 = obj1.get();
    EXPECT_EQ(TestObject::construct_count, 1);
    EXPECT_EQ(TestObject::destruct_count, 0);

    obj1.reset();  // Release
    EXPECT_EQ(TestObject::destruct_count, 1);

    auto obj2 = pool.acquire(2, 2.0, "second");
    void* addr2 = obj2.get();
    EXPECT_EQ(TestObject::construct_count, 2);

    // Should reuse the same memory
    EXPECT_EQ(addr1, addr2);
}

TEST_F(ObjectPoolTest, DestructionVerification) {
    fcm::ObjectPool<TestObject> pool(resource_);

    bool destroyed = false;
    {
        auto obj = pool.acquire(&destroyed);
        EXPECT_FALSE(destroyed);
    }
    // Object should be destroyed when handle goes out of scope
    EXPECT_TRUE(destroyed);
    EXPECT_EQ(pool.outstanding(), 0);
}

// Reserve tests
TEST_F(ObjectPoolTest, ReserveNodes) {
    fcm::ObjectPool<TestObject> pool(resource_);

    TestObject::reset_counts();

    // Pre-allocate nodes
    pool.reserve_nodes(10);

    // No objects should be constructed
    EXPECT_EQ(TestObject::construct_count, 0);
    EXPECT_EQ(pool.outstanding(), 0);

    // Acquire should be fast (no allocation needed)
    auto obj = pool.acquire(1, 1.0, "test");
    ASSERT_NE(obj, nullptr);
    EXPECT_EQ(pool.outstanding(), 1);
}

TEST_F(ObjectPoolTest, ReserveMultipleTimes) {
    fcm::ObjectPool<TestObject> pool(resource_);

    pool.reserve_nodes(5);
    pool.reserve_nodes(5);
    pool.reserve_nodes(5);

    // Should have at least 15 nodes reserved
    std::vector<fcm::ObjectPool<TestObject>::handle> objects;
    for (int i = 0; i < 15; ++i) {
        objects.push_back(pool.acquire(i, 0.0, ""));
    }

    EXPECT_EQ(pool.outstanding(), 15);
}

// Manual release tests
TEST_F(ObjectPoolTest, ManualRelease) {
    fcm::ObjectPool<TestObject> pool(resource_);

    TestObject* raw = pool.acquire(42, 3.14, "test").release();
    ASSERT_NE(raw, nullptr);
    EXPECT_EQ(pool.outstanding(), 1);

    // Manual release
    pool.release(raw);
    EXPECT_EQ(pool.outstanding(), 0);
}

TEST_F(ObjectPoolTest, ManualReleaseNull) {
    fcm::ObjectPool<TestObject> pool(resource_);

    // Should handle null gracefully
    pool.release(nullptr);
    EXPECT_EQ(pool.outstanding(), 0);
}

// Handle behavior tests
TEST_F(ObjectPoolTest, HandleMoveSemantics) {
    fcm::ObjectPool<TestObject> pool(resource_);

    auto obj1 = pool.acquire(1, 1.0, "one");
    EXPECT_EQ(pool.outstanding(), 1);

    auto obj2 = std::move(obj1);
    EXPECT_EQ(obj1.get(), nullptr);
    EXPECT_NE(obj2.get(), nullptr);
    EXPECT_EQ(pool.outstanding(), 1);

    obj2.reset();
    EXPECT_EQ(pool.outstanding(), 0);
}

TEST_F(ObjectPoolTest, HandleSwap) {
    fcm::ObjectPool<TestObject> pool(resource_);

    auto obj1 = pool.acquire(1, 1.0, "one");
    auto obj2 = pool.acquire(2, 2.0, "two");

    TestObject* ptr1 = obj1.get();
    TestObject* ptr2 = obj2.get();

    obj1.swap(obj2);

    EXPECT_EQ(obj1.get(), ptr2);
    EXPECT_EQ(obj2.get(), ptr1);
    EXPECT_EQ(pool.outstanding(), 2);
}

TEST_F(ObjectPoolTest, HandleReset) {
    fcm::ObjectPool<TestObject> pool(resource_);

    auto obj = pool.acquire(1, 1.0, "one");
    EXPECT_EQ(pool.outstanding(), 1);

    obj.reset();
    EXPECT_EQ(obj.get(), nullptr);
    EXPECT_EQ(pool.outstanding(), 0);

    // Can acquire again
    obj = pool.acquire(2, 2.0, "two");
    EXPECT_EQ(pool.outstanding(), 1);
}

// POD type tests
TEST_F(ObjectPoolTest, PodTypePool) {
    fcm::ObjectPool<PodType> pool(resource_);

    auto obj = pool.acquire();
    ASSERT_NE(obj, nullptr);

    obj->x = 42;
    obj->y = 3.14;
    std::memset(obj->data, 'A', sizeof(obj->data));

    EXPECT_EQ(obj->x, 42);
    EXPECT_DOUBLE_EQ(obj->y, 3.14);
    EXPECT_EQ(obj->data[0], 'A');

    obj.reset();
    EXPECT_EQ(pool.outstanding(), 0);
}

// Different block sizes
TEST_F(ObjectPoolTest, SmallBlockSize) {
    fcm::ObjectPool<TestObject, 256> pool(resource_);

    std::vector<fcm::ObjectPool<TestObject, 256>::handle> objects;
    for (int i = 0; i < 20; ++i) {
        objects.push_back(pool.acquire(i, static_cast<double>(i), ""));
    }

    EXPECT_EQ(pool.outstanding(), 20);

    objects.clear();
    EXPECT_EQ(pool.outstanding(), 0);
}

TEST_F(ObjectPoolTest, LargeBlockSize) {
    fcm::ObjectPool<TestObject, 8192> pool(resource_);

    std::vector<fcm::ObjectPool<TestObject, 8192>::handle> objects;
    for (int i = 0; i < 100; ++i) {
        objects.push_back(pool.acquire(i, static_cast<double>(i), ""));
    }

    EXPECT_EQ(pool.outstanding(), 100);

    objects.clear();
    EXPECT_EQ(pool.outstanding(), 0);
}

// Complex object tests
TEST_F(ObjectPoolTest, ComplexObjectConstruction) {
    struct ComplexObject {
        std::vector<int> data;
        std::unique_ptr<std::string> name;

        ComplexObject(std::initializer_list<int> init, const std::string& n)
            : data(init), name(std::make_unique<std::string>(n)) {}
    };

    fcm::ObjectPool<ComplexObject> pool(resource_);

    auto obj = pool.acquire(std::initializer_list<int>{1, 2, 3, 4, 5}, "complex");
    ASSERT_NE(obj, nullptr);

    EXPECT_EQ(obj->data.size(), 5);
    EXPECT_EQ(obj->data[2], 3);
    EXPECT_EQ(*obj->name, "complex");

    obj.reset();
    EXPECT_EQ(pool.outstanding(), 0);
}

// Stress tests
TEST_F(ObjectPoolTest, StressAcquireRelease) {
    fcm::ObjectPool<TestObject> pool(resource_);

    std::vector<fcm::ObjectPool<TestObject>::handle> objects;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> action(0, 1);

    for (int i = 0; i < 1000; ++i) {
        if (action(gen) == 0 || objects.empty()) {
            // Acquire
            objects.push_back(pool.acquire(i, static_cast<double>(i), "obj"));
        } else {
            // Release random
            std::uniform_int_distribution<> idx(0, static_cast<int>(objects.size()) - 1);
            int index = idx(gen);
            objects.erase(objects.begin() + index);
        }
    }

    std::size_t remaining = objects.size();
    objects.clear();

    EXPECT_EQ(pool.outstanding(), 0);
    EXPECT_GE(TestObject::destruct_count, remaining);
}

TEST_F(ObjectPoolTest, RapidReusePattern) {
    fcm::ObjectPool<TestObject> pool(resource_);

    // Rapid acquire/release cycles
    for (int cycle = 0; cycle < 100; ++cycle) {
        std::vector<fcm::ObjectPool<TestObject>::handle> batch;

        // Acquire batch
        for (int i = 0; i < 10; ++i) {
            batch.push_back(pool.acquire(cycle * 10 + i, 0.0, ""));
        }

        EXPECT_EQ(pool.outstanding(), 10);

        // Release batch
        batch.clear();

        EXPECT_EQ(pool.outstanding(), 0);
    }
}

// Memory pattern tests
TEST_F(ObjectPoolTest, MemoryStability) {
    fcm::ObjectPool<TestObject> pool(resource_);

    // Get addresses of allocated objects
    std::vector<void*> addresses;
    std::vector<fcm::ObjectPool<TestObject>::handle> objects;

    for (int i = 0; i < 5; ++i) {
        auto obj = pool.acquire(i, 0.0, "");
        addresses.push_back(obj.get());
        objects.push_back(std::move(obj));
    }

    // Release all
    objects.clear();

    // Reacquire - should get same addresses (in LIFO order)
    for (int i = 0; i < 5; ++i) {
        auto obj = pool.acquire(i + 10, 0.0, "");
        void* addr = obj.get();

        // Should find this address in our list
        auto it = std::find(addresses.begin(), addresses.end(), addr);
        EXPECT_NE(it, addresses.end()) << "Address reuse expected";

        objects.push_back(std::move(obj));
    }
}

// Large object pool
TEST_F(ObjectPoolTest, LargeObjectPool) {
    struct LargeObject {
        char data[4096];
        int id;

        LargeObject(int i) : id(i) {
            std::memset(data, i % 256, sizeof(data));
        }
    };

    fcm::ObjectPool<LargeObject, 16384> pool(resource_);

    std::vector<fcm::ObjectPool<LargeObject, 16384>::handle> objects;

    for (int i = 0; i < 10; ++i) {
        objects.push_back(pool.acquire(i));
    }

    // Verify data integrity
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(objects[i]->id, i);
        for (int j = 0; j < 4096; ++j) {
            EXPECT_EQ(objects[i]->data[j], static_cast<char>(i % 256));
        }
    }

    EXPECT_EQ(pool.outstanding(), 10);
}

// Edge cases
TEST_F(ObjectPoolTest, DestructorExceptionSafety) {
    struct ThrowingObject {
        bool* flag;
        ThrowingObject(bool* f) : flag(f) { *flag = false; }
        ~ThrowingObject() noexcept(false) {
            *flag = true;
            // Note: throwing from destructor is generally bad practice
            // This is just for testing
        }
    };

    fcm::ObjectPool<ThrowingObject> pool(resource_);

    bool destroyed = false;
    {
        auto obj = pool.acquire(&destroyed);
        EXPECT_FALSE(destroyed);
    }
    EXPECT_TRUE(destroyed);
    EXPECT_EQ(pool.outstanding(), 0);
}

TEST_F(ObjectPoolTest, ZeroSizeObject) {
    struct Empty {};

    fcm::ObjectPool<Empty> pool(resource_);

    auto obj1 = pool.acquire();
    auto obj2 = pool.acquire();

    ASSERT_NE(obj1, nullptr);
    ASSERT_NE(obj2, nullptr);
    EXPECT_NE(obj1.get(), obj2.get());  // Different objects

    EXPECT_EQ(pool.outstanding(), 2);
}

#if CORE_MEMORY_ENABLE_TELEMETRY
TEST_F(ObjectPoolTest, TelemetryTracking) {
    fcm::ObjectPool<TestObject> pool(resource_);

    std::vector<std::string> events;
    pool.set_telemetry_callback(
        [&events](const char* event, [[maybe_unused]] const fcm::ObjectPool<TestObject>::telemetry_t& t) {
            events.push_back(event);
        });

    auto obj1 = pool.acquire(1, 1.0, "one");
    auto obj2 = pool.acquire(2, 2.0, "two");

    auto& telemetry = pool.telemetry();
    EXPECT_EQ(telemetry.acquired, 2);
    EXPECT_EQ(telemetry.outstanding, 2);

    obj1.reset();
    EXPECT_EQ(telemetry.released, 1);
    EXPECT_EQ(telemetry.outstanding, 1);

    pool.reserve_nodes(5);
    EXPECT_EQ(telemetry.prewarmed, 5);

    EXPECT_GE(events.size(), 3);
}

TEST_F(ObjectPoolTest, TelemetryWithHighThroughput) {
    fcm::ObjectPool<TestObject> pool(resource_);

    std::atomic<int> event_count{0};
    pool.set_telemetry_callback(
        [&event_count]([[maybe_unused]] const char* event, [[maybe_unused]] const fcm::ObjectPool<TestObject>::telemetry_t& t) {
            ++event_count;
        });

    // High throughput operations
    for (int i = 0; i < 100; ++i) {
        auto obj = pool.acquire(i, static_cast<double>(i), "");
        // Immediate release
    }

    auto& telemetry = pool.telemetry();
    EXPECT_EQ(telemetry.acquired, 100);
    EXPECT_EQ(telemetry.released, 100);
    EXPECT_EQ(telemetry.outstanding, 0);

    EXPECT_GT(event_count.load(), 0);
}
#endif