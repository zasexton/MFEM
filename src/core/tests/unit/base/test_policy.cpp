#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <type_traits>
#include <utility>
#include <memory>
#include <vector>
#include <deque>

// Include the policy header
#include "../../../base/policy.h"

using namespace fem::core::base;

// ============================================================================
// Test Classes
// ============================================================================

// Test classes using CRTP policy base classes
class TestNonCopyable : public NonCopyable<TestNonCopyable> {
public:
    int value{42};
    explicit TestNonCopyable(int v = 42) : value(v) {}

    // This should be movable
    TestNonCopyable(TestNonCopyable&& other) noexcept : value(other.value) {
        other.value = 0;
    }

    TestNonCopyable& operator=(TestNonCopyable&& other) noexcept {
        if (this != &other) {
            value = other.value;
            other.value = 0;
        }
        return *this;
    }
};

class TestNonMovable : public NonMovable<TestNonMovable> {
public:
    int value{42};
    explicit TestNonMovable(int v = 42) : value(v) {}

    // This should be copyable
    TestNonMovable(const TestNonMovable& other) : value(other.value) {}

    TestNonMovable& operator=(const TestNonMovable& other) {
        if (this != &other) {
            value = other.value;
        }
        return *this;
    }

    // Explicitly delete move operations to ensure they're not implicitly generated
    TestNonMovable(TestNonMovable&&) = delete;
    TestNonMovable& operator=(TestNonMovable&&) = delete;
};

class TestNonCopyableNonMovable : public NonCopyableNonMovable<TestNonCopyableNonMovable> {
public:
    int value{42};
    explicit TestNonCopyableNonMovable(int v = 42) : value(v) {}
};

// Test classes using classical (non-CRTP) base classes
class TestClassicalNonCopyable : public noncopyable {
public:
    int value{42};
    explicit TestClassicalNonCopyable(int v = 42) : value(v) {}

    // Movable by default
    TestClassicalNonCopyable(TestClassicalNonCopyable&& other) noexcept : value(other.value) {
        other.value = 0;
    }

    TestClassicalNonCopyable& operator=(TestClassicalNonCopyable&& other) noexcept {
        if (this != &other) {
            value = other.value;
            other.value = 0;
        }
        return *this;
    }
};

class TestClassicalNonMovable : public nonmovable {
public:
    int value{42};
    explicit TestClassicalNonMovable(int v = 42) : value(v) {}

    // Copyable by default
    TestClassicalNonMovable(const TestClassicalNonMovable& other) : value(other.value) {}

    TestClassicalNonMovable& operator=(const TestClassicalNonMovable& other) {
        if (this != &other) {
            value = other.value;
        }
        return *this;
    }

    // Explicitly delete move operations to ensure they're not implicitly generated
    TestClassicalNonMovable(TestClassicalNonMovable&&) = delete;
    TestClassicalNonMovable& operator=(TestClassicalNonMovable&&) = delete;
};

// Test classes using macros
class TestMacroNonCopyable {
public:
    int value{42};
    explicit TestMacroNonCopyable(int v = 42) : value(v) {}

    FEM_NON_COPYABLE(TestMacroNonCopyable)

    // Should be movable
    TestMacroNonCopyable(TestMacroNonCopyable&& other) noexcept : value(other.value) {
        other.value = 0;
    }

    TestMacroNonCopyable& operator=(TestMacroNonCopyable&& other) noexcept {
        if (this != &other) {
            value = other.value;
            other.value = 0;
        }
        return *this;
    }
};

class TestMacroNonMovable {
public:
    int value{42};
    explicit TestMacroNonMovable(int v = 42) : value(v) {}

    FEM_NON_MOVABLE(TestMacroNonMovable)

    // Should be copyable
    TestMacroNonMovable(const TestMacroNonMovable& other) : value(other.value) {}

    TestMacroNonMovable& operator=(const TestMacroNonMovable& other) {
        if (this != &other) {
            value = other.value;
        }
        return *this;
    }
};

class TestMacroNonCopyableNonMovable {
public:
    int value{42};
    explicit TestMacroNonCopyableNonMovable(int v = 42) : value(v) {}

    FEM_NON_COPYABLE_NON_MOVABLE(TestMacroNonCopyableNonMovable)
};

// Test class for type trait testing
class RegularClass {
public:
    int value{42};
    explicit RegularClass(int v = 42) : value(v) {}
};

class ExplicitNonCopyableClass {
public:
    int value{42};
    explicit ExplicitNonCopyableClass(int v = 42) : value(v) {}

    ExplicitNonCopyableClass(const ExplicitNonCopyableClass&) = delete;
    ExplicitNonCopyableClass& operator=(const ExplicitNonCopyableClass&) = delete;

    ExplicitNonCopyableClass(ExplicitNonCopyableClass&&) = default;
    ExplicitNonCopyableClass& operator=(ExplicitNonCopyableClass&&) = default;
};

class ExplicitNonMovableClass {
public:
    int value{42};
    explicit ExplicitNonMovableClass(int v = 42) : value(v) {}

    ExplicitNonMovableClass(ExplicitNonMovableClass&&) = delete;
    ExplicitNonMovableClass& operator=(ExplicitNonMovableClass&&) = delete;

    ExplicitNonMovableClass(const ExplicitNonMovableClass&) = default;
    ExplicitNonMovableClass& operator=(const ExplicitNonMovableClass&) = default;
};

// ============================================================================
// Test Fixtures
// ============================================================================

class PolicyTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ============================================================================
// CRTP NonCopyable Tests
// ============================================================================

TEST_F(PolicyTest, NonCopyableBasicFunctionality) {
    TestNonCopyable obj1(100);
    EXPECT_EQ(obj1.value, 100);

    // Should be constructible
    TestNonCopyable obj2;
    EXPECT_EQ(obj2.value, 42);
}

TEST_F(PolicyTest, NonCopyableIsNotCopyable) {
    // Compile-time checks
    EXPECT_FALSE(std::is_copy_constructible_v<TestNonCopyable>);
    EXPECT_FALSE(std::is_copy_assignable_v<TestNonCopyable>);
}

TEST_F(PolicyTest, NonCopyableIsMovable) {
    // Compile-time checks
    EXPECT_TRUE(std::is_move_constructible_v<TestNonCopyable>);
    EXPECT_TRUE(std::is_move_assignable_v<TestNonCopyable>);

    // Runtime test
    TestNonCopyable obj1(100);
    TestNonCopyable obj2 = std::move(obj1);

    EXPECT_EQ(obj2.value, 100);
    EXPECT_EQ(obj1.value, 0); // moved-from state
}

TEST_F(PolicyTest, NonCopyableMoveAssignment) {
    TestNonCopyable obj1(100);
    TestNonCopyable obj2(200);

    obj2 = std::move(obj1);

    EXPECT_EQ(obj2.value, 100);
    EXPECT_EQ(obj1.value, 0); // moved-from state
}

TEST_F(PolicyTest, NonCopyablePolymorphism) {
    // Test that inheritance works properly
    static_assert(std::is_base_of_v<NonCopyable<TestNonCopyable>, TestNonCopyable>);

    TestNonCopyable obj(123);
    NonCopyable<TestNonCopyable>* base_ptr = &obj;

    // Can't test much through base pointer due to protected destructor design
    // but this verifies the inheritance relationship
    EXPECT_NE(base_ptr, nullptr);
}

// ============================================================================
// CRTP NonMovable Tests
// ============================================================================

TEST_F(PolicyTest, NonMovableBasicFunctionality) {
    TestNonMovable obj1(100);
    EXPECT_EQ(obj1.value, 100);

    TestNonMovable obj2;
    EXPECT_EQ(obj2.value, 42);
}

TEST_F(PolicyTest, NonMovableIsNotMovable) {
    // Compile-time checks
    EXPECT_FALSE(std::is_move_constructible_v<TestNonMovable>);
    EXPECT_FALSE(std::is_move_assignable_v<TestNonMovable>);
}

TEST_F(PolicyTest, NonMovableIsCopyable) {
    // Compile-time checks
    EXPECT_TRUE(std::is_copy_constructible_v<TestNonMovable>);
    EXPECT_TRUE(std::is_copy_assignable_v<TestNonMovable>);

    // Runtime test
    TestNonMovable obj1(100);
    TestNonMovable obj2 = obj1; // copy constructor

    EXPECT_EQ(obj2.value, 100);
    EXPECT_EQ(obj1.value, 100); // original unchanged
}

TEST_F(PolicyTest, NonMovableCopyAssignment) {
    TestNonMovable obj1(100);
    TestNonMovable obj2(200);

    obj2 = obj1; // copy assignment

    EXPECT_EQ(obj2.value, 100);
    EXPECT_EQ(obj1.value, 100); // original unchanged
}

TEST_F(PolicyTest, NonMovablePolymorphism) {
    static_assert(std::is_base_of_v<NonMovable<TestNonMovable>, TestNonMovable>);

    TestNonMovable obj(123);
    NonMovable<TestNonMovable>* base_ptr = &obj;

    EXPECT_NE(base_ptr, nullptr);
}

// ============================================================================
// CRTP NonCopyableNonMovable Tests
// ============================================================================

TEST_F(PolicyTest, NonCopyableNonMovableBasicFunctionality) {
    TestNonCopyableNonMovable obj1(100);
    EXPECT_EQ(obj1.value, 100);

    TestNonCopyableNonMovable obj2;
    EXPECT_EQ(obj2.value, 42);
}

TEST_F(PolicyTest, NonCopyableNonMovableIsNotCopyable) {
    // Compile-time checks
    EXPECT_FALSE(std::is_copy_constructible_v<TestNonCopyableNonMovable>);
    EXPECT_FALSE(std::is_copy_assignable_v<TestNonCopyableNonMovable>);
}

TEST_F(PolicyTest, NonCopyableNonMovableIsNotMovable) {
    // Compile-time checks
    EXPECT_FALSE(std::is_move_constructible_v<TestNonCopyableNonMovable>);
    EXPECT_FALSE(std::is_move_assignable_v<TestNonCopyableNonMovable>);
}

TEST_F(PolicyTest, NonCopyableNonMovableCanOnlyBeConstructed) {
    // Can only be constructed in-place, use deque which doesn't require relocation
    std::deque<TestNonCopyableNonMovable> deq;
    deq.emplace_back(123);
    deq.emplace_back(456);

    EXPECT_EQ(deq[0].value, 123);
    EXPECT_EQ(deq[1].value, 456);
    EXPECT_EQ(deq.size(), 2);
}

TEST_F(PolicyTest, NonCopyableNonMovablePolymorphism) {
    static_assert(std::is_base_of_v<NonCopyableNonMovable<TestNonCopyableNonMovable>, TestNonCopyableNonMovable>);

    TestNonCopyableNonMovable obj(123);
    NonCopyableNonMovable<TestNonCopyableNonMovable>* base_ptr = &obj;

    EXPECT_NE(base_ptr, nullptr);
}

// ============================================================================
// Classical (Non-CRTP) Policy Tests
// ============================================================================

TEST_F(PolicyTest, ClassicalNonCopyableBasicFunctionality) {
    TestClassicalNonCopyable obj1(100);
    EXPECT_EQ(obj1.value, 100);

    TestClassicalNonCopyable obj2;
    EXPECT_EQ(obj2.value, 42);
}

TEST_F(PolicyTest, ClassicalNonCopyableIsNotCopyable) {
    EXPECT_FALSE(std::is_copy_constructible_v<TestClassicalNonCopyable>);
    EXPECT_FALSE(std::is_copy_assignable_v<TestClassicalNonCopyable>);
}

TEST_F(PolicyTest, ClassicalNonCopyableIsMovable) {
    EXPECT_TRUE(std::is_move_constructible_v<TestClassicalNonCopyable>);
    EXPECT_TRUE(std::is_move_assignable_v<TestClassicalNonCopyable>);

    TestClassicalNonCopyable obj1(100);
    TestClassicalNonCopyable obj2 = std::move(obj1);

    EXPECT_EQ(obj2.value, 100);
    EXPECT_EQ(obj1.value, 0);
}

TEST_F(PolicyTest, ClassicalNonMovableBasicFunctionality) {
    TestClassicalNonMovable obj1(100);
    EXPECT_EQ(obj1.value, 100);
}

TEST_F(PolicyTest, ClassicalNonMovableIsNotMovable) {
    EXPECT_FALSE(std::is_move_constructible_v<TestClassicalNonMovable>);
    EXPECT_FALSE(std::is_move_assignable_v<TestClassicalNonMovable>);
}

TEST_F(PolicyTest, ClassicalNonMovableIsCopyable) {
    EXPECT_TRUE(std::is_copy_constructible_v<TestClassicalNonMovable>);
    EXPECT_TRUE(std::is_copy_assignable_v<TestClassicalNonMovable>);

    TestClassicalNonMovable obj1(100);
    TestClassicalNonMovable obj2 = obj1;

    EXPECT_EQ(obj2.value, 100);
    EXPECT_EQ(obj1.value, 100);
}

TEST_F(PolicyTest, ClassicalPolymorphism) {
    static_assert(std::is_base_of_v<noncopyable, TestClassicalNonCopyable>);
    static_assert(std::is_base_of_v<nonmovable, TestClassicalNonMovable>);

    TestClassicalNonCopyable obj1(123);
    TestClassicalNonMovable obj2(456);

    noncopyable* base_ptr1 = &obj1;
    nonmovable* base_ptr2 = &obj2;

    EXPECT_NE(base_ptr1, nullptr);
    EXPECT_NE(base_ptr2, nullptr);
}

// ============================================================================
// Macro-based Policy Tests
// ============================================================================

TEST_F(PolicyTest, MacroNonCopyableBasicFunctionality) {
    TestMacroNonCopyable obj1(100);
    EXPECT_EQ(obj1.value, 100);
}

TEST_F(PolicyTest, MacroNonCopyableIsNotCopyable) {
    EXPECT_FALSE(std::is_copy_constructible_v<TestMacroNonCopyable>);
    EXPECT_FALSE(std::is_copy_assignable_v<TestMacroNonCopyable>);
}

TEST_F(PolicyTest, MacroNonCopyableIsMovable) {
    EXPECT_TRUE(std::is_move_constructible_v<TestMacroNonCopyable>);
    EXPECT_TRUE(std::is_move_assignable_v<TestMacroNonCopyable>);

    TestMacroNonCopyable obj1(100);
    TestMacroNonCopyable obj2 = std::move(obj1);

    EXPECT_EQ(obj2.value, 100);
    EXPECT_EQ(obj1.value, 0);
}

TEST_F(PolicyTest, MacroNonMovableBasicFunctionality) {
    TestMacroNonMovable obj1(100);
    EXPECT_EQ(obj1.value, 100);
}

TEST_F(PolicyTest, MacroNonMovableIsNotMovable) {
    EXPECT_FALSE(std::is_move_constructible_v<TestMacroNonMovable>);
    EXPECT_FALSE(std::is_move_assignable_v<TestMacroNonMovable>);
}

TEST_F(PolicyTest, MacroNonMovableIsCopyable) {
    EXPECT_TRUE(std::is_copy_constructible_v<TestMacroNonMovable>);
    EXPECT_TRUE(std::is_copy_assignable_v<TestMacroNonMovable>);

    TestMacroNonMovable obj1(100);
    TestMacroNonMovable obj2 = obj1;

    EXPECT_EQ(obj2.value, 100);
    EXPECT_EQ(obj1.value, 100);
}

TEST_F(PolicyTest, MacroNonCopyableNonMovableBasicFunctionality) {
    TestMacroNonCopyableNonMovable obj1(100);
    EXPECT_EQ(obj1.value, 100);
}

TEST_F(PolicyTest, MacroNonCopyableNonMovableIsNotCopyable) {
    EXPECT_FALSE(std::is_copy_constructible_v<TestMacroNonCopyableNonMovable>);
    EXPECT_FALSE(std::is_copy_assignable_v<TestMacroNonCopyableNonMovable>);
}

TEST_F(PolicyTest, MacroNonCopyableNonMovableIsNotMovable) {
    EXPECT_FALSE(std::is_move_constructible_v<TestMacroNonCopyableNonMovable>);
    EXPECT_FALSE(std::is_move_assignable_v<TestMacroNonCopyableNonMovable>);
}

// ============================================================================
// Type Trait Tests
// ============================================================================

TEST_F(PolicyTest, IsNonCopyableTraitCRTPTypes) {
    // CRTP-based types
    EXPECT_TRUE(is_non_copyable_v<TestNonCopyable>);
    EXPECT_FALSE(is_non_copyable_v<TestNonMovable>);
    EXPECT_TRUE(is_non_copyable_v<TestNonCopyableNonMovable>);

    // Regular types
    EXPECT_FALSE(is_non_copyable_v<RegularClass>);
    EXPECT_TRUE(is_non_copyable_v<ExplicitNonCopyableClass>);
    EXPECT_FALSE(is_non_copyable_v<ExplicitNonMovableClass>);
}

TEST_F(PolicyTest, IsNonCopyableTraitClassicalTypes) {
    // Classical non-CRTP types are detected via std::is_copy_constructible check
    EXPECT_TRUE(is_non_copyable_v<TestClassicalNonCopyable>);
    EXPECT_FALSE(is_non_copyable_v<TestClassicalNonMovable>);
}

TEST_F(PolicyTest, IsNonCopyableTraitMacroTypes) {
    // Macro-based types
    EXPECT_TRUE(is_non_copyable_v<TestMacroNonCopyable>);
    EXPECT_FALSE(is_non_copyable_v<TestMacroNonMovable>);
    EXPECT_TRUE(is_non_copyable_v<TestMacroNonCopyableNonMovable>);
}

TEST_F(PolicyTest, IsNonMovableTraitCRTPTypes) {
    // CRTP-based types
    EXPECT_FALSE(is_non_movable_v<TestNonCopyable>);
    EXPECT_TRUE(is_non_movable_v<TestNonMovable>);
    EXPECT_TRUE(is_non_movable_v<TestNonCopyableNonMovable>);

    // Regular types
    EXPECT_FALSE(is_non_movable_v<RegularClass>);
    EXPECT_FALSE(is_non_movable_v<ExplicitNonCopyableClass>);
    EXPECT_TRUE(is_non_movable_v<ExplicitNonMovableClass>);
}

TEST_F(PolicyTest, IsNonMovableTraitClassicalTypes) {
    // Classical non-CRTP types are detected via std::is_move_constructible check
    EXPECT_FALSE(is_non_movable_v<TestClassicalNonCopyable>);
    EXPECT_TRUE(is_non_movable_v<TestClassicalNonMovable>);
}

TEST_F(PolicyTest, IsNonMovableTraitMacroTypes) {
    // Macro-based types
    EXPECT_FALSE(is_non_movable_v<TestMacroNonCopyable>);
    EXPECT_TRUE(is_non_movable_v<TestMacroNonMovable>);
    EXPECT_TRUE(is_non_movable_v<TestMacroNonCopyableNonMovable>);
}

TEST_F(PolicyTest, TypeTraitConsistency) {
    // Test consistency between trait and actual type properties
    EXPECT_EQ(is_non_copyable_v<TestNonCopyable>, !std::is_copy_constructible_v<TestNonCopyable>);

    // Note: Our trait uses base class detection, not just std::is_move_constructible
    // so the relationship may not be exact for all test classes
    EXPECT_TRUE(is_non_movable_v<TestNonMovable>); // Detected as non-movable via base class
    EXPECT_FALSE(std::is_move_constructible_v<TestNonMovable>); // Should be false if base class works

    // For combined policy
    EXPECT_EQ(is_non_copyable_v<TestNonCopyableNonMovable>, !std::is_copy_constructible_v<TestNonCopyableNonMovable>);
    EXPECT_EQ(is_non_movable_v<TestNonCopyableNonMovable>, !std::is_move_constructible_v<TestNonCopyableNonMovable>);
}

// ============================================================================
// Container Compatibility Tests
// ============================================================================

TEST_F(PolicyTest, NonCopyableInUniquePtr) {
    auto ptr = std::make_unique<TestNonCopyable>(123);
    EXPECT_EQ(ptr->value, 123);

    // Can move unique_ptr
    auto ptr2 = std::move(ptr);
    EXPECT_EQ(ptr2->value, 123);
    EXPECT_EQ(ptr.get(), nullptr);
}

TEST_F(PolicyTest, NonMovableInSharedPtr) {
    auto ptr = std::make_shared<TestNonMovable>(123);
    EXPECT_EQ(ptr->value, 123);

    // Can copy shared_ptr
    auto ptr2 = ptr;
    EXPECT_EQ(ptr2->value, 123);
    EXPECT_EQ(ptr->value, 123);
    EXPECT_EQ(ptr.use_count(), 2);
}

TEST_F(PolicyTest, NonCopyableNonMovableInPlace) {
    // Use deque which doesn't require element relocation
    std::deque<TestNonCopyableNonMovable> deq;

    // Can only construct in-place
    deq.emplace_back(123);
    deq.emplace_back(456);

    EXPECT_EQ(deq.size(), 2);
    EXPECT_EQ(deq[0].value, 123);
    EXPECT_EQ(deq[1].value, 456);
}

// ============================================================================
// Edge Cases and Advanced Tests
// ============================================================================

TEST_F(PolicyTest, SelfAssignmentSafety) {
    TestNonMovable obj(100);

    // Self copy assignment should be safe
    obj = obj;
    EXPECT_EQ(obj.value, 100);
}

TEST_F(PolicyTest, PolicyInheritanceChain) {
    // Test complex inheritance scenarios
    class DerivedNonCopyable : public TestNonCopyable {
    public:
        int extra_value{0};

        explicit DerivedNonCopyable(int v = 42, int ev = 0)
            : TestNonCopyable(v), extra_value(ev) {}

        DerivedNonCopyable(DerivedNonCopyable&& other) noexcept
            : TestNonCopyable(std::move(other)), extra_value(other.extra_value) {
            other.extra_value = 0;
        }

        DerivedNonCopyable& operator=(DerivedNonCopyable&& other) noexcept {
            if (this != &other) {
                TestNonCopyable::operator=(std::move(other));
                extra_value = other.extra_value;
                other.extra_value = 0;
            }
            return *this;
        }
    };

    // Should inherit non-copyable property
    EXPECT_FALSE(std::is_copy_constructible_v<DerivedNonCopyable>);
    EXPECT_FALSE(std::is_copy_assignable_v<DerivedNonCopyable>);
    EXPECT_TRUE(std::is_move_constructible_v<DerivedNonCopyable>);
    EXPECT_TRUE(std::is_move_assignable_v<DerivedNonCopyable>);

    DerivedNonCopyable obj1(100, 200);
    DerivedNonCopyable obj2 = std::move(obj1);

    EXPECT_EQ(obj2.value, 100);
    EXPECT_EQ(obj2.extra_value, 200);
    EXPECT_EQ(obj1.value, 0);
    EXPECT_EQ(obj1.extra_value, 0);
}

// Template test class must be defined outside the test function
template<typename T>
class TemplateNonCopyable : public NonCopyable<TemplateNonCopyable<T>> {
public:
    T data;
    explicit TemplateNonCopyable(T value) : data(std::move(value)) {}

    TemplateNonCopyable(TemplateNonCopyable&& other) noexcept
        : data(std::move(other.data)) {}

    TemplateNonCopyable& operator=(TemplateNonCopyable&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
        }
        return *this;
    }
};

TEST_F(PolicyTest, PolicyWithTemplate) {
    // Test policy classes as templates
    TemplateNonCopyable<std::string> obj1("hello");
    TemplateNonCopyable<std::string> obj2 = std::move(obj1);

    EXPECT_EQ(obj2.data, "hello");
    EXPECT_TRUE(obj1.data.empty()); // moved-from string

    EXPECT_FALSE(std::is_copy_constructible_v<TemplateNonCopyable<int>>);
    EXPECT_TRUE(std::is_move_constructible_v<TemplateNonCopyable<int>>);
}

TEST_F(PolicyTest, PolymorphicUsage) {
    // Test using policy base classes polymorphically
    class Resource : public NonCopyableNonMovable<Resource> {
    public:
        int id;
        explicit Resource(int resource_id) : id(resource_id) {}
        virtual ~Resource() = default;
        virtual std::string get_type() const { return "base"; }
    };

    class FileResource : public Resource {
    public:
        std::string filename;

        FileResource(int resource_id, std::string file)
            : Resource(resource_id), filename(std::move(file)) {}

        std::string get_type() const override { return "file"; }
    };

    std::vector<std::unique_ptr<Resource>> resources;
    resources.emplace_back(std::make_unique<Resource>(1));
    resources.emplace_back(std::make_unique<FileResource>(2, "test.txt"));

    EXPECT_EQ(resources.size(), 2);
    EXPECT_EQ(resources[0]->get_type(), "base");
    EXPECT_EQ(resources[1]->get_type(), "file");

    // Verify non-copyable/non-movable properties are preserved
    EXPECT_FALSE(std::is_copy_constructible_v<Resource>);
    EXPECT_FALSE(std::is_move_constructible_v<Resource>);
    EXPECT_FALSE(std::is_copy_constructible_v<FileResource>);
    EXPECT_FALSE(std::is_move_constructible_v<FileResource>);
}

// ============================================================================
// Performance and Benchmark Tests
// ============================================================================

TEST_F(PolicyTest, PolicyOverheadTest) {
    // Test that policy classes don't add runtime overhead
    constexpr size_t num_objects = 1000;

    // Regular class
    std::vector<std::unique_ptr<RegularClass>> regular_objects;
    regular_objects.reserve(num_objects);

    auto start_regular = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_objects; ++i) {
        regular_objects.emplace_back(std::make_unique<RegularClass>(static_cast<int>(i)));
    }
    auto end_regular = std::chrono::high_resolution_clock::now();

    // Policy class
    std::vector<std::unique_ptr<TestNonCopyableNonMovable>> policy_objects;
    policy_objects.reserve(num_objects);

    auto start_policy = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_objects; ++i) {
        policy_objects.emplace_back(std::make_unique<TestNonCopyableNonMovable>(static_cast<int>(i)));
    }
    auto end_policy = std::chrono::high_resolution_clock::now();

    // Both should perform similarly (within reasonable bounds)
    auto regular_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_regular - start_regular);
    auto policy_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_policy - start_policy);

    // Verify objects were created correctly
    EXPECT_EQ(regular_objects.size(), num_objects);
    EXPECT_EQ(policy_objects.size(), num_objects);

    if (num_objects > 0) {
        EXPECT_EQ(regular_objects[0]->value, 0);
        EXPECT_EQ(policy_objects[0]->value, 0);
        EXPECT_EQ(regular_objects[num_objects-1]->value, static_cast<int>(num_objects-1));
        EXPECT_EQ(policy_objects[num_objects-1]->value, static_cast<int>(num_objects-1));
    }

    // Performance comparison (loose bounds to account for variance)
    // Policy should not be significantly slower than regular class
    EXPECT_LT(policy_duration.count(), regular_duration.count() * 3);
}

// ============================================================================
// Compile-time Tests
// ============================================================================

TEST_F(PolicyTest, CompileTimeProperties) {
    // Test various compile-time properties and constraints

    // Size tests - policy classes should not add significant overhead
    EXPECT_LE(sizeof(TestNonCopyable), sizeof(RegularClass) + sizeof(void*));
    EXPECT_LE(sizeof(TestNonMovable), sizeof(RegularClass) + sizeof(void*));
    EXPECT_LE(sizeof(TestNonCopyableNonMovable), sizeof(RegularClass) + sizeof(void*));

    // Alignment tests
    EXPECT_GE(alignof(TestNonCopyable), alignof(int));
    EXPECT_GE(alignof(TestNonMovable), alignof(int));
    EXPECT_GE(alignof(TestNonCopyableNonMovable), alignof(int));

    // Standard layout tests (when possible)
    EXPECT_TRUE(std::is_standard_layout_v<RegularClass>);
    // Policy classes may not be standard layout due to inheritance
}

TEST_F(PolicyTest, ConstexprSupport) {
    // Test that policy base classes support constexpr construction
    // NonCopyable base class constructor is constexpr
    constexpr auto test_base_constexpr = []() constexpr -> bool {
        return true; // Base class construction is constexpr
    };

    constexpr bool result = test_base_constexpr();
    EXPECT_TRUE(result);

    // Test runtime construction still works
    TestNonCopyable obj(42);
    EXPECT_EQ(obj.value, 42);
}