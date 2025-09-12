#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "core/base/object.h"

#include <thread>
#include <vector>
#include <set>
#include <unordered_set>

using namespace fem::core::base;
using namespace testing;

// Forward declaration for test fixture
class TestObject;

// Test fixture for Object tests
class ObjectTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset any static state if needed
        // TestObject::reset_flags(); // This will be called where needed
    }

    void TearDown() override {
        // Cleanup if needed
    }
};

// Test derived class
class TestObject : public Object {
public:
    TestObject() : Object("TestObject") {
        constructor_called = true;
        initialize();
    }

    explicit TestObject(int value)
        : Object("TestObject"), value_(value) {
        constructor_called = true;
        initialize();
    }

    ~TestObject() override {
        destructor_called = true;
        on_destroy();
    }

    int get_value() const { return value_; }
    void set_value(int v) { value_ = v; }

    // For testing lifecycle callbacks
    static bool constructor_called;
    static bool destructor_called;
    static int on_create_call_count;
    static int on_destroy_call_count;
    static void reset_flags() {
        constructor_called = false;
        destructor_called = false;
        on_create_call_count = 0;
        on_destroy_call_count = 0;
    }

protected:
    void on_create() override {
        on_create_called_ = true;
        ++on_create_call_count;
    }

    void on_destroy() override {
        on_destroy_called_ = true;
        ++on_destroy_call_count;
    }

public:
    bool on_create_called_{false};
    bool on_destroy_called_{false};

private:
    int value_{42};
};

bool TestObject::constructor_called = false;
bool TestObject::destructor_called = false;
int TestObject::on_create_call_count = 0;
int TestObject::on_destroy_call_count = 0;

// Another test class for type checking
class AnotherTestObject : public Object {
public:
    AnotherTestObject() : Object("AnotherTestObject") {}
};

// Derived from TestObject
class DerivedTestObject : public TestObject {
public:
    DerivedTestObject() : TestObject(100) {}
};

// ============================================================================
// Construction and Identity Tests
// ============================================================================

TEST(ObjectConstructionTest, DefaultConstruction) {
    Object obj;

    EXPECT_GE(obj.id(), 0);
    EXPECT_EQ(obj.class_name(), "");
    EXPECT_TRUE(obj.is_valid());
    EXPECT_EQ(obj.ref_count(), 1u);
}

TEST(ObjectConstructionTest, ConstructionWithClassName) {
    Object obj("MyObject");

    EXPECT_EQ(obj.class_name(), "MyObject");
    EXPECT_TRUE(obj.is_valid());
}

TEST(ObjectConstructionTest, UniqueIDGeneration) {
    Object obj1;
    Object obj2;
    Object obj3;

    EXPECT_NE(obj1.id(), obj2.id());
    EXPECT_NE(obj2.id(), obj3.id());
    EXPECT_NE(obj1.id(), obj3.id());

    // IDs should be sequential (assuming no other objects created)
    EXPECT_GT(obj2.id(), obj1.id());
    EXPECT_GT(obj3.id(), obj2.id());
}

TEST(ObjectConstructionTest, CopyConstructionCreatesNewID) {
    Object obj1("Original");
    Object obj2(obj1);

    EXPECT_NE(obj2.id(), obj1.id());
    EXPECT_EQ(obj2.class_name(), obj1.class_name());
    EXPECT_EQ(obj2.ref_count(), 1u);  // New object starts with ref count 1
}

TEST(ObjectConstructionTest, MoveConstruction) {
    Object obj1("Movable");
    auto original_id = obj1.id();

    Object obj2(std::move(obj1));

    EXPECT_EQ(obj2.id(), original_id);
    EXPECT_EQ(obj2.class_name(), "Movable");
}

// ============================================================================
// Assignment Operator Tests
// ============================================================================

TEST(ObjectAssignmentTest, CopyAssignment) {
    Object obj1("First");
    Object obj2("Second");

    auto obj1_id = obj1.id();
    auto obj2_id = obj2.id();

    obj2 = obj1;

    // ID should NOT change on assignment
    EXPECT_EQ(obj2.id(), obj2_id);
    EXPECT_NE(obj2.id(), obj1_id);
}

TEST(ObjectAssignmentTest, MoveAssignment) {
    Object obj1("First");
    Object obj2("Second");

    auto obj1_id = obj1.id();
    (void)obj1_id; // Suppress unused variable warning

    obj2 = std::move(obj1);

    // Move assignment implementation dependent
    // But obj2 should be valid after move
    EXPECT_TRUE(obj2.is_valid());
}

TEST(ObjectAssignmentTest, SelfAssignment) {
    Object obj("Self");
    auto original_id = obj.id();

    // Self copy-assignment test
    // Note: GCC doesn't warn about self-assignment for this case
    obj = obj;  // Self copy-assignment

    EXPECT_EQ(obj.id(), original_id);
    EXPECT_TRUE(obj.is_valid());
}

// ============================================================================
// Type Checking Tests
// ============================================================================

TEST(ObjectTypeTest, TypeInformation) {
    TestObject test_obj;
    AnotherTestObject another_obj;

    EXPECT_NE(test_obj.type_info(), another_obj.type_info());
}

TEST(ObjectTypeTest, IsTypeChecks) {
    TestObject test_obj;

    EXPECT_TRUE(test_obj.is_type<TestObject>());
    EXPECT_TRUE(test_obj.is_type<Object>());
    EXPECT_FALSE(test_obj.is_type<AnotherTestObject>());
}

TEST(ObjectTypeTest, SafeCastingWithAs) {
    Object* base_ptr = new TestObject(123);

    auto* test_ptr = base_ptr->as<TestObject>();
    ASSERT_NE(test_ptr, nullptr);
    EXPECT_EQ(test_ptr->get_value(), 123);

    auto* another_ptr = base_ptr->as<AnotherTestObject>();
    EXPECT_EQ(another_ptr, nullptr);

    delete base_ptr;
}

TEST(ObjectTypeTest, SafeCastingWithAsRef) {
    TestObject test_obj(456);
    Object& base_ref = test_obj;

    auto& test_ref = base_ref.as_ref<TestObject>();
    EXPECT_EQ(test_ref.get_value(), 456);
    // as_ref() behaves differently depending on whether assertions are enabled.
    // In release builds it throws an exception, while in debug builds it aborts.
#if !CORE_ENABLE_ASSERTS
    // When assertions are disabled, as_ref throws std::bad_cast on failure.
    EXPECT_THROW({
        [[maybe_unused]] auto& unused_ref = base_ref.as_ref<AnotherTestObject>();
    }, std::bad_cast);
#else  // CORE_ENABLE_ASSERTS
    // With assertions enabled, as_ref aborts instead of throwing, so use EXPECT_DEATH.
    EXPECT_DEATH({ (void)base_ref.as_ref<AnotherTestObject>(); }, ".*");
#endif  // CORE_ENABLE_ASSERTS
}

TEST(ObjectTypeTest, ConstCasting) {
    const TestObject test_obj(789);
    const Object& base_ref = test_obj;

    auto* test_ptr = base_ref.as<TestObject>();
    ASSERT_NE(test_ptr, nullptr);
    EXPECT_EQ(test_ptr->get_value(), 789);

    const auto& test_ref = base_ref.as_ref<TestObject>();
    EXPECT_EQ(test_ref.get_value(), 789);
}

TEST(ObjectTypeTest, ConstAsRefFailure) {
    const TestObject test_obj(101);
    const Object& base_ref = test_obj;
#if !CORE_ENABLE_ASSERTS
    EXPECT_THROW({
        [[maybe_unused]] const auto& unused_ref = base_ref.as_ref<AnotherTestObject>();
    }, std::bad_cast);
#else  // CORE_ENABLE_ASSERTS
    EXPECT_DEATH({ (void)base_ref.as_ref<AnotherTestObject>(); }, "Invalid object cast");
#endif  // CORE_ENABLE_ASSERTS
}

// ============================================================================
// Reference Counting Tests
// ============================================================================

TEST(ObjectRefCountTest, BasicReferenceCounting) {
    Object* obj = new Object("RefCounted");

    EXPECT_EQ(obj->ref_count(), 1u);

    obj->add_ref();
    EXPECT_EQ(obj->ref_count(), 2u);

    obj->add_ref();
    EXPECT_EQ(obj->ref_count(), 3u);

    obj->release();  // count = 2
    EXPECT_EQ(obj->ref_count(), 2u);

    obj->release();  // count = 1
    EXPECT_EQ(obj->ref_count(), 1u);

    obj->release();  // count = 0, object deleted
    // obj is now deleted, cannot access
}

TEST(ObjectRefCountTest, ThreadSafeReferenceCounting) {
    Object* obj = new Object("ThreadSafe");
    const int num_threads = 10;
    const int refs_per_thread = 1000;

    // Add references from multiple threads
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([obj, refs_per_thread]() {
            for (int j = 0; j < refs_per_thread; ++j) {
                obj->add_ref();
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Should have 1 (initial) + num_threads * refs_per_thread
    EXPECT_EQ(obj->ref_count(), 1u + num_threads * refs_per_thread);

    // Clean up
    for (int i = 0; i < num_threads * refs_per_thread; ++i) {
        obj->release();
    }
    obj->release();  // Final release
}

// ============================================================================
// Lifecycle Tests
// ============================================================================

TEST(ObjectLifecycleTest, ValidState) {
    Object obj;
    EXPECT_TRUE(obj.is_valid());

    obj.destroy();
    EXPECT_FALSE(obj.is_valid());
}

TEST_F(ObjectTest, LifecycleCallbacks) {
    {
        TestObject obj;
        EXPECT_TRUE(TestObject::constructor_called);
        EXPECT_TRUE(obj.on_create_called_);
    }

    EXPECT_TRUE(TestObject::destructor_called);
}

TEST_F(ObjectTest, OnCreateAndOnDestroyCalledOnce) {
    TestObject::reset_flags();

    {
        auto obj = make_object<TestObject>();
        EXPECT_EQ(TestObject::on_create_call_count, 1);
        EXPECT_EQ(TestObject::on_destroy_call_count, 0);
    }

    EXPECT_EQ(TestObject::on_create_call_count, 1);
    EXPECT_EQ(TestObject::on_destroy_call_count, 1);
}

// ============================================================================
// Comparison and Hashing Tests
// ============================================================================

TEST(ObjectComparisonTest, EqualityComparison) {
    Object obj1;
    Object obj2;
    Object obj1_copy(obj1);

    EXPECT_EQ(obj1, obj1);  // Self equality
    EXPECT_NE(obj1, obj2);  // Different objects
    EXPECT_NE(obj1, obj1_copy);  // Copy has different ID
}

TEST(ObjectComparisonTest, OrderingComparison) {
    Object obj1;
    Object obj2;

    // Since IDs are sequential, obj2 should have higher ID
    EXPECT_LT(obj1, obj2);
    EXPECT_GT(obj2, obj1);
    EXPECT_LE(obj1, obj1);
    EXPECT_GE(obj1, obj1);
}

TEST(ObjectComparisonTest, HashSupport) {
    Object obj1;
    Object obj2;

    std::hash<Object> hasher;
    auto hash1 = hasher(obj1);
    auto hash2 = hasher(obj2);

    EXPECT_NE(hash1, hash2);  // Different objects should have different hashes

    // Can use in unordered containers
    std::unordered_set<Object::id_type> id_set;
    id_set.insert(obj1.id());
    id_set.insert(obj2.id());
    EXPECT_EQ(id_set.size(), 2u);
}

// ============================================================================
// Debug Utilities Tests
// ============================================================================

TEST(ObjectDebugTest, ToString) {
    Object obj("DebugObject");
    auto str = obj.to_string();

    EXPECT_THAT(str, HasSubstr("DebugObject"));
    EXPECT_THAT(str, HasSubstr(std::to_string(obj.id())));
    EXPECT_THAT(str, HasSubstr("refs=1"));
}

TEST(ObjectDebugTest, DebugInfo) {
    Object obj("DetailedDebug");
    auto info = obj.debug_info();

    EXPECT_THAT(info, HasSubstr("DetailedDebug"));
    EXPECT_THAT(info, HasSubstr(std::to_string(obj.id())));
    EXPECT_THAT(info, HasSubstr("Valid: true"));  // is_valid() == true
    EXPECT_THAT(info, HasSubstr("Refs: 1"));
}

// ============================================================================
// object_ptr Smart Pointer Tests
// ============================================================================

TEST_F(ObjectTest, ObjectPtrBasicConstruction) {
    {
        object_ptr<TestObject> ptr(new TestObject());
        EXPECT_TRUE(ptr);
        EXPECT_EQ(ptr->ref_count(), 1u);
    }

    EXPECT_TRUE(TestObject::destructor_called);
}

TEST(ObjectPtrTest, NullPointerHandling) {
    object_ptr<TestObject> null_ptr;
    EXPECT_FALSE(null_ptr);
    EXPECT_EQ(null_ptr.get(), nullptr);

    object_ptr<TestObject> null_ptr2(nullptr);
    EXPECT_FALSE(null_ptr2);
}

TEST(ObjectPtrTest, CopySemantics) {
    TestObject* raw = new TestObject(100);
    object_ptr<TestObject> ptr1(raw);
    EXPECT_EQ(raw->ref_count(), 1u);

    object_ptr<TestObject> ptr2(ptr1);
    EXPECT_EQ(raw->ref_count(), 2u);
    EXPECT_EQ(ptr2->get_value(), 100);

    {
        object_ptr<TestObject> ptr3 = ptr1;
        EXPECT_EQ(raw->ref_count(), 3u);
    }

    EXPECT_EQ(raw->ref_count(), 2u);
}

TEST(ObjectPtrTest, MoveSemantics) {
    object_ptr<TestObject> ptr1(new TestObject(200));
    auto* raw = ptr1.get();
    EXPECT_EQ(raw->ref_count(), 1u);

    object_ptr<TestObject> ptr2(std::move(ptr1));
    EXPECT_FALSE(ptr1);
    EXPECT_EQ(ptr2.get(), raw);
    EXPECT_EQ(raw->ref_count(), 1u);
}

TEST(ObjectPtrTest, AssignmentOperators) {
    object_ptr<TestObject> ptr1(new TestObject(1));
    object_ptr<TestObject> ptr2(new TestObject(2));

    auto* raw1 = ptr1.get();

    ptr2 = ptr1;  // Copy assignment
    EXPECT_EQ(ptr2.get(), raw1);
    EXPECT_EQ(raw1->ref_count(), 2u);

    object_ptr<TestObject> ptr3(new TestObject(3));
    ptr2 = std::move(ptr3);  // Move assignment
    EXPECT_FALSE(ptr3);
    EXPECT_EQ(ptr2->get_value(), 3);
    EXPECT_EQ(raw1->ref_count(), 1u);
}

TEST(ObjectPtrTest, ResetAndRelease) {
    object_ptr<TestObject> ptr(new TestObject(42));

    ptr.reset(new TestObject(84));
    EXPECT_EQ(ptr->get_value(), 84);

    auto* raw = ptr.release();
    EXPECT_FALSE(ptr);
    EXPECT_EQ(raw->get_value(), 84);
    EXPECT_EQ(raw->ref_count(), 1u);

    // Manual cleanup since we released
    raw->release();
}

TEST(ObjectPtrTest, PolymorphicUsage) {
    object_ptr<Object> base_ptr(new TestObject(999));
    EXPECT_TRUE(base_ptr);

    // Downcast
    auto* test_obj = base_ptr->as<TestObject>();
    ASSERT_NE(test_obj, nullptr);
    EXPECT_EQ(test_obj->get_value(), 999);
}

TEST(ObjectPtrTest, MakeObjectFactory) {
    auto ptr = make_object<TestObject>(555);
    EXPECT_TRUE(ptr);
    EXPECT_EQ(ptr->get_value(), 555);
    EXPECT_EQ(ptr->ref_count(), 1u);
}

TEST(ObjectPtrTest, HashSupport) {
    auto ptr1 = make_object<TestObject>(1);
    auto ptr2 = make_object<TestObject>(2);
    object_ptr<TestObject> null_ptr;

    std::hash<object_ptr<TestObject>> hasher;
    auto hash1 = hasher(ptr1);
    auto hash2 = hasher(ptr2);
    auto hash_null = hasher(null_ptr);

    EXPECT_NE(hash1, hash2);
    EXPECT_EQ(hash_null, 0u);

    // Can use in unordered containers
    std::unordered_set<object_ptr<TestObject>,
                      std::hash<object_ptr<TestObject>>> ptr_set;
    ptr_set.insert(ptr1);
    ptr_set.insert(ptr2);
    EXPECT_EQ(ptr_set.size(), 2u);
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST(ObjectThreadSafetyTest, ConcurrentIDGeneration) {
    const int num_threads = 20;
    const int objects_per_thread = 100;

    std::vector<std::thread> threads;
    std::vector<std::set<Object::id_type>> id_sets(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&id_sets, i, objects_per_thread]() {
            for (int j = 0; j < objects_per_thread; ++j) {
                Object obj;
                id_sets[i].insert(obj.id());
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Check that all IDs are unique
    std::set<Object::id_type> all_ids;
    for (const auto& id_set : id_sets) {
        for (auto id : id_set) {
            auto [it, inserted] = all_ids.insert(id);
            EXPECT_TRUE(inserted);  // Should always insert (no duplicates)
        }
    }

    EXPECT_EQ(all_ids.size(), static_cast<size_t>(num_threads * objects_per_thread));
}

TEST(ObjectThreadSafetyTest, ConcurrentRefCountingWithObjectPtr) {
    auto shared_obj = make_object<TestObject>(42);
    const int num_threads = 10;
    const int iterations = 1000;

    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([shared_obj, iterations]() {
            for (int j = 0; j < iterations; ++j) {
                object_ptr<TestObject> local_copy = shared_obj;
                EXPECT_EQ(local_copy->get_value(), 42);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Object should still be valid with ref count 1
    EXPECT_EQ(shared_obj->ref_count(), 1u);
    EXPECT_TRUE(shared_obj->is_valid());
}

// ============================================================================
// Death Tests (for debug assertions)
// ============================================================================

#ifdef CORE_DEBUG
TEST(ObjectDeathTest, DereferenceNullObjectPtr) {
    object_ptr<TestObject> null_ptr;
    EXPECT_DEATH(null_ptr->get_value(), "Dereferencing null object_ptr");
}

TEST(ObjectDeathTest, InvalidCastAsRef) {
    TestObject test_obj;
    Object& base_ref = test_obj;
    EXPECT_DEATH({ (void)base_ref.as_ref<AnotherTestObject>(); }, "Invalid object cast");
}
#endif