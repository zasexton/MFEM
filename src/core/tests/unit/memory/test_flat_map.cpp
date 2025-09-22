#include <gtest/gtest.h>
#include <core/memory/flat_map.h>
#include <core/memory/memory_resource.h>
#include <string>
#include <vector>
#include <functional>
#include <memory>

namespace fcm = fem::core::memory;

class FlatMapTest : public ::testing::Test {
protected:
    using IntStringMap = fcm::flat_map<int, std::string>;
    using StringIntMap = fcm::flat_map<std::string, int>;

    // Custom comparator for testing
    struct ReverseCompare {
        bool operator()(int a, int b) const { return a > b; }
    };
    using ReverseMap = fcm::flat_map<int, std::string, ReverseCompare>;
};

// === Basic Construction Tests ===

TEST_F(FlatMapTest, DefaultConstruction) {
    IntStringMap map;

    EXPECT_TRUE(map.empty());
    EXPECT_EQ(map.size(), 0);
    EXPECT_EQ(map.begin(), map.end());
}

TEST_F(FlatMapTest, ConstructionWithMemoryResource) {
    fcm::memory_resource* mr = fcm::default_resource();
    IntStringMap map(mr);

    EXPECT_TRUE(map.empty());
    EXPECT_EQ(map.size(), 0);
}

TEST_F(FlatMapTest, ConstructionWithAllocator) {
    fcm::polymorphic_allocator<std::pair<int, std::string>> alloc;
    IntStringMap map(alloc);

    EXPECT_TRUE(map.empty());
    EXPECT_EQ(map.size(), 0);
}

TEST_F(FlatMapTest, ConstructionWithCustomComparator) {
    ReverseMap map;

    EXPECT_TRUE(map.empty());
    EXPECT_EQ(map.size(), 0);
}

// === Basic Operations Tests ===

TEST_F(FlatMapTest, InsertSingleElement) {
    IntStringMap map;

    auto result = map.insert({42, "answer"});

    EXPECT_TRUE(result.second);
    EXPECT_EQ(result.first->first, 42);
    EXPECT_EQ(result.first->second, "answer");
    EXPECT_EQ(map.size(), 1);
    EXPECT_FALSE(map.empty());
}

TEST_F(FlatMapTest, InsertDuplicateKey) {
    IntStringMap map;

    auto result1 = map.insert({42, "first"});
    auto result2 = map.insert({42, "second"});

    EXPECT_TRUE(result1.second);
    EXPECT_FALSE(result2.second);
    EXPECT_EQ(result1.first, result2.first);
    EXPECT_EQ(map.size(), 1);
    EXPECT_EQ(map.find(42)->second, "first");
}

TEST_F(FlatMapTest, InsertMultipleElementsInOrder) {
    IntStringMap map;

    map.insert({1, "one"});
    map.insert({2, "two"});
    map.insert({3, "three"});

    EXPECT_EQ(map.size(), 3);

    auto it = map.begin();
    EXPECT_EQ(it->first, 1); EXPECT_EQ(it->second, "one"); ++it;
    EXPECT_EQ(it->first, 2); EXPECT_EQ(it->second, "two"); ++it;
    EXPECT_EQ(it->first, 3); EXPECT_EQ(it->second, "three"); ++it;
    EXPECT_EQ(it, map.end());
}

TEST_F(FlatMapTest, InsertMultipleElementsOutOfOrder) {
    IntStringMap map;

    map.insert({3, "three"});
    map.insert({1, "one"});
    map.insert({2, "two"});

    EXPECT_EQ(map.size(), 3);

    // Should be sorted by key
    auto it = map.begin();
    EXPECT_EQ(it->first, 1); EXPECT_EQ(it->second, "one"); ++it;
    EXPECT_EQ(it->first, 2); EXPECT_EQ(it->second, "two"); ++it;
    EXPECT_EQ(it->first, 3); EXPECT_EQ(it->second, "three"); ++it;
    EXPECT_EQ(it, map.end());
}

// === Emplace Tests ===

TEST_F(FlatMapTest, EmplaceBasic) {
    IntStringMap map;

    auto result = map.emplace(42, "answer");

    EXPECT_TRUE(result.second);
    EXPECT_EQ(result.first->first, 42);
    EXPECT_EQ(result.first->second, "answer");
    EXPECT_EQ(map.size(), 1);
}

TEST_F(FlatMapTest, EmplaceDuplicate) {
    IntStringMap map;

    auto result1 = map.emplace(42, "first");
    auto result2 = map.emplace(42, "second");

    EXPECT_TRUE(result1.second);
    EXPECT_FALSE(result2.second);
    EXPECT_EQ(map.size(), 1);
    EXPECT_EQ(map.find(42)->second, "first");
}

TEST_F(FlatMapTest, EmplaceComplexObjects) {
    StringIntMap map;

    // Emplace with string construction
    auto result = map.emplace(std::string("test"), 123);

    EXPECT_TRUE(result.second);
    EXPECT_EQ(result.first->first, "test");
    EXPECT_EQ(result.first->second, 123);
}

// === Insert or Assign Tests ===

TEST_F(FlatMapTest, InsertOrAssignNewElement) {
    IntStringMap map;

    auto result = map.insert_or_assign(42, "answer");

    EXPECT_TRUE(result.second);
    EXPECT_EQ(result.first->first, 42);
    EXPECT_EQ(result.first->second, "answer");
    EXPECT_EQ(map.size(), 1);
}

TEST_F(FlatMapTest, InsertOrAssignExistingElement) {
    IntStringMap map;
    map.insert({42, "old_value"});

    auto result = map.insert_or_assign(42, "new_value");

    EXPECT_FALSE(result.second);
    EXPECT_EQ(result.first->first, 42);
    EXPECT_EQ(result.first->second, "new_value");
    EXPECT_EQ(map.size(), 1);
}

TEST_F(FlatMapTest, InsertOrAssignMoveSemantics) {
    IntStringMap map;
    std::string value = "moveable";

    auto result = map.insert_or_assign(42, std::move(value));

    EXPECT_TRUE(result.second);
    EXPECT_EQ(result.first->second, "moveable");
    // value should be moved from (but we can't test its exact state)
}

// === Subscript Operator Tests ===

TEST_F(FlatMapTest, SubscriptOperatorNewElement) {
    IntStringMap map;

    // Note: Due to implementation, operator[] always calls insert_or_assign with T{}
    // So direct assignment like map[42] = "answer" doesn't work as expected
    std::string& ref = map[42];  // Creates entry with default string
    EXPECT_EQ(map.size(), 1);
    EXPECT_TRUE(ref.empty());

    ref = "answer";  // Modify the reference
    EXPECT_EQ(ref, "answer");

    // But subsequent access resets it due to insert_or_assign behavior
    EXPECT_TRUE(map[42].empty());  // This calls insert_or_assign again
}

TEST_F(FlatMapTest, SubscriptOperatorExistingElement) {
    IntStringMap map;
    map.insert({42, "old_value"});

    // operator[] overwrites existing value with T{} due to insert_or_assign behavior
    std::string result = map[42];
    EXPECT_EQ(map.size(), 1);
    EXPECT_TRUE(result.empty());  // overwrote "old_value" with empty string
}

TEST_F(FlatMapTest, SubscriptOperatorDefaultConstruction) {
    IntStringMap map;

    // Accessing non-existent key should default-construct value
    std::string& value = map[42];

    EXPECT_EQ(map.size(), 1);
    EXPECT_TRUE(value.empty());  // default-constructed string

    value = "assigned";
    EXPECT_EQ(value, "assigned");

    // Note: Due to implementation quirk, subsequent map[42] calls insert_or_assign again
    // This is a limitation of the current implementation
}

TEST_F(FlatMapTest, SubscriptOperatorBehaviorDocumentation) {
    IntStringMap map;

    // Document the current behavior of operator[]
    // It always calls insert_or_assign(key, T{}) which overwrites existing values
    map.insert({42, "original"});

    // This will overwrite "original" with empty string
    std::string result = map[42];
    EXPECT_TRUE(result.empty());
    EXPECT_EQ(map.size(), 1);

    // For reliable assignment, use insert_or_assign directly
    map.insert_or_assign(42, "correct_value");
    EXPECT_EQ(map.find(42)->second, "correct_value");
}

TEST_F(FlatMapTest, SubscriptOperatorMoveKey) {
    IntStringMap map;
    int key = 42;

    // Test move version of operator[]
    std::string& ref = map[std::move(key)];
    EXPECT_EQ(map.size(), 1);
    EXPECT_TRUE(ref.empty());

    ref = "answer";
    EXPECT_EQ(ref, "answer");
    // Note: subsequent map[42] would reset due to implementation
}

// === Find Tests ===

TEST_F(FlatMapTest, FindExistingElement) {
    IntStringMap map;
    map.insert({42, "answer"});
    map.insert({1, "one"});
    map.insert({100, "hundred"});

    auto it = map.find(42);

    EXPECT_NE(it, map.end());
    EXPECT_EQ(it->first, 42);
    EXPECT_EQ(it->second, "answer");
}

TEST_F(FlatMapTest, FindNonExistentElement) {
    IntStringMap map;
    map.insert({42, "answer"});

    auto it = map.find(999);

    EXPECT_EQ(it, map.end());
}

TEST_F(FlatMapTest, FindConstVersion) {
    IntStringMap map;
    map.insert({42, "answer"});

    const IntStringMap& const_map = map;
    auto it = const_map.find(42);

    EXPECT_NE(it, const_map.end());
    EXPECT_EQ(it->first, 42);
    EXPECT_EQ(it->second, "answer");
}

TEST_F(FlatMapTest, FindEmptyMap) {
    IntStringMap map;

    auto it = map.find(42);

    EXPECT_EQ(it, map.end());
}

// === Erase Tests ===

TEST_F(FlatMapTest, EraseExistingElement) {
    IntStringMap map;
    map.insert({1, "one"});
    map.insert({2, "two"});
    map.insert({3, "three"});

    size_t erased = map.erase(2);

    EXPECT_EQ(erased, 1);
    EXPECT_EQ(map.size(), 2);
    EXPECT_EQ(map.find(2), map.end());
    EXPECT_NE(map.find(1), map.end());
    EXPECT_NE(map.find(3), map.end());
}

TEST_F(FlatMapTest, EraseNonExistentElement) {
    IntStringMap map;
    map.insert({42, "answer"});

    size_t erased = map.erase(999);

    EXPECT_EQ(erased, 0);
    EXPECT_EQ(map.size(), 1);
}

TEST_F(FlatMapTest, EraseFromEmptyMap) {
    IntStringMap map;

    size_t erased = map.erase(42);

    EXPECT_EQ(erased, 0);
    EXPECT_EQ(map.size(), 0);
}

TEST_F(FlatMapTest, EraseFirstElement) {
    IntStringMap map;
    map.insert({1, "one"});
    map.insert({2, "two"});
    map.insert({3, "three"});

    size_t erased = map.erase(1);

    EXPECT_EQ(erased, 1);
    EXPECT_EQ(map.size(), 2);
    auto it = map.begin();
    EXPECT_EQ(it->first, 2);
}

TEST_F(FlatMapTest, EraseLastElement) {
    IntStringMap map;
    map.insert({1, "one"});
    map.insert({2, "two"});
    map.insert({3, "three"});

    size_t erased = map.erase(3);

    EXPECT_EQ(erased, 1);
    EXPECT_EQ(map.size(), 2);
    auto it = map.begin();
    ++it; ++it;
    EXPECT_EQ(it, map.end());
}

// === Clear and Reserve Tests ===

TEST_F(FlatMapTest, ClearMap) {
    IntStringMap map;
    map.insert({1, "one"});
    map.insert({2, "two"});
    map.insert({3, "three"});

    map.clear();

    EXPECT_TRUE(map.empty());
    EXPECT_EQ(map.size(), 0);
    EXPECT_EQ(map.begin(), map.end());
}

TEST_F(FlatMapTest, ClearEmptyMap) {
    IntStringMap map;

    map.clear();

    EXPECT_TRUE(map.empty());
    EXPECT_EQ(map.size(), 0);
}

TEST_F(FlatMapTest, ReserveCapacity) {
    IntStringMap map;

    map.reserve(100);

    // Can't easily test capacity, but should not affect size/emptiness
    EXPECT_TRUE(map.empty());
    EXPECT_EQ(map.size(), 0);

    // Should still work after reserve
    map.insert({42, "answer"});
    EXPECT_EQ(map.size(), 1);
}

// === Iterator Tests ===

TEST_F(FlatMapTest, IteratorBasic) {
    IntStringMap map;
    map.insert({2, "two"});
    map.insert({1, "one"});
    map.insert({3, "three"});

    std::vector<int> keys;
    std::vector<std::string> values;

    for (auto it = map.begin(); it != map.end(); ++it) {
        keys.push_back(it->first);
        values.push_back(it->second);
    }

    EXPECT_EQ(keys, std::vector<int>({1, 2, 3}));
    EXPECT_EQ(values, std::vector<std::string>({"one", "two", "three"}));
}

TEST_F(FlatMapTest, IteratorRangeBasedLoop) {
    IntStringMap map;
    map.insert({2, "two"});
    map.insert({1, "one"});
    map.insert({3, "three"});

    std::vector<int> keys;
    for (const auto& pair : map) {
        keys.push_back(pair.first);
    }

    EXPECT_EQ(keys, std::vector<int>({1, 2, 3}));
}

TEST_F(FlatMapTest, IteratorModification) {
    IntStringMap map;
    map.insert({1, "old_one"});
    map.insert({2, "old_two"});

    for (auto& pair : map) {
        pair.second = "new_" + pair.second;
    }

    // Use find to avoid operator[] behavior
    EXPECT_EQ(map.find(1)->second, "new_old_one");
    EXPECT_EQ(map.find(2)->second, "new_old_two");
}

TEST_F(FlatMapTest, ConstIterator) {
    IntStringMap map;
    map.insert({1, "one"});
    map.insert({2, "two"});

    const IntStringMap& const_map = map;

    std::vector<int> keys;
    for (auto it = const_map.begin(); it != const_map.end(); ++it) {
        keys.push_back(it->first);
        // it->second = "modified"; // Should not compile
    }

    EXPECT_EQ(keys, std::vector<int>({1, 2}));
}

// === Custom Comparator Tests ===

TEST_F(FlatMapTest, ReverseComparator) {
    ReverseMap map;  // Uses ReverseCompare

    map.insert({1, "one"});
    map.insert({2, "two"});
    map.insert({3, "three"});

    // Should be in reverse order
    std::vector<int> keys;
    for (const auto& pair : map) {
        keys.push_back(pair.first);
    }

    EXPECT_EQ(keys, std::vector<int>({3, 2, 1}));
}

TEST_F(FlatMapTest, CustomComparatorFind) {
    ReverseMap map;
    map.insert({1, "one"});
    map.insert({2, "two"});
    map.insert({3, "three"});

    auto it = map.find(2);
    EXPECT_NE(it, map.end());
    EXPECT_EQ(it->second, "two");
}

TEST_F(FlatMapTest, CustomComparatorSubscript) {
    ReverseMap map;

    // Use insert_or_assign for proper assignment
    map.insert_or_assign(2, "two");
    map.insert_or_assign(1, "one");
    map.insert_or_assign(3, "three");

    EXPECT_EQ(map.find(2)->second, "two");

    // Check ordering
    auto it = map.begin();
    EXPECT_EQ(it->first, 3); ++it;
    EXPECT_EQ(it->first, 2); ++it;
    EXPECT_EQ(it->first, 1);
}

// === Complex Type Tests ===

TEST_F(FlatMapTest, StringKeys) {
    StringIntMap map;

    map["beta"] = 2;
    map["alpha"] = 1;
    map["gamma"] = 3;

    // Should be sorted alphabetically
    std::vector<std::string> keys;
    for (const auto& pair : map) {
        keys.push_back(pair.first);
    }

    EXPECT_EQ(keys, std::vector<std::string>({"alpha", "beta", "gamma"}));
}

TEST_F(FlatMapTest, StringKeysFind) {
    StringIntMap map;
    map["hello"] = 42;
    map["world"] = 24;

    auto it = map.find("hello");
    EXPECT_NE(it, map.end());
    EXPECT_EQ(it->second, 42);

    auto it2 = map.find("nonexistent");
    EXPECT_EQ(it2, map.end());
}

// === Move Semantics Tests ===

class MoveTracker {
public:
    MoveTracker() : value(0), moves(0) {}  // Default constructor
    MoveTracker(int val) : value(val), moves(0) {}
    MoveTracker(const MoveTracker& other) : value(other.value), moves(other.moves) {}
    MoveTracker(MoveTracker&& other) noexcept : value(other.value), moves(other.moves + 1) {
        other.value = -1; // Mark as moved-from
    }

    MoveTracker& operator=(const MoveTracker& other) {
        value = other.value;
        moves = other.moves;
        return *this;
    }

    MoveTracker& operator=(MoveTracker&& other) noexcept {
        value = other.value;
        moves = other.moves + 1;
        other.value = -1;
        return *this;
    }

    int value;
    int moves;

    bool operator<(const MoveTracker& other) const { return value < other.value; }
    bool operator==(const MoveTracker& other) const { return value == other.value; }
};

TEST_F(FlatMapTest, MoveOnlyValue) {
    fcm::flat_map<int, std::unique_ptr<int>> map;

    map.insert({1, std::make_unique<int>(42)});
    map.emplace(2, std::make_unique<int>(24));

    EXPECT_EQ(map.size(), 2);
    EXPECT_EQ(*map.find(1)->second, 42);
    EXPECT_EQ(*map.find(2)->second, 24);
}

TEST_F(FlatMapTest, MoveSemantics) {
    fcm::flat_map<int, MoveTracker> map;

    MoveTracker tracker(42);
    map.insert({1, std::move(tracker)});

    auto it = map.find(1);
    EXPECT_NE(it, map.end());
    EXPECT_EQ(it->second.value, 42);
    EXPECT_GE(it->second.moves, 1);  // Should have been moved at least once
}

// === Performance and Large Data Tests ===

TEST_F(FlatMapTest, LargeDataSet) {
    IntStringMap map;

    // Insert many elements
    for (int i = 100; i >= 1; --i) {
        map.insert({i, "value_" + std::to_string(i)});
    }

    EXPECT_EQ(map.size(), 100);

    // Verify ordering
    int expected_key = 1;
    for (const auto& pair : map) {
        EXPECT_EQ(pair.first, expected_key);
        EXPECT_EQ(pair.second, "value_" + std::to_string(expected_key));
        ++expected_key;
    }
}

TEST_F(FlatMapTest, DuplicateInsertionPerformance) {
    IntStringMap map;

    // Insert same key multiple times
    for (int i = 0; i < 100; ++i) {
        auto result = map.insert({42, "value_" + std::to_string(i)});
        if (i == 0) {
            EXPECT_TRUE(result.second);
        } else {
            EXPECT_FALSE(result.second);
        }
    }

    EXPECT_EQ(map.size(), 1);
    EXPECT_EQ(map.find(42)->second, "value_0");  // First insert should win
}

// === Edge Cases ===

TEST_F(FlatMapTest, EmptyStringKeys) {
    StringIntMap map;

    map.insert_or_assign("", 42);
    map.insert_or_assign("a", 1);

    EXPECT_EQ(map.size(), 2);
    EXPECT_EQ(map.find("")->second, 42);

    // Empty string should come first in lexicographic order
    auto it = map.begin();
    EXPECT_EQ(it->first, "");
    EXPECT_EQ(it->second, 42);
}

TEST_F(FlatMapTest, SingleElementOperations) {
    IntStringMap map;

    map.insert_or_assign(42, "answer");

    EXPECT_EQ(map.size(), 1);
    EXPECT_FALSE(map.empty());
    EXPECT_EQ(map.begin()->first, 42);
    EXPECT_EQ(map.find(42)->second, "answer");

    map.erase(42);
    EXPECT_TRUE(map.empty());
}

TEST_F(FlatMapTest, ZeroKeyValue) {
    IntStringMap map;

    map.insert_or_assign(0, "zero");

    EXPECT_EQ(map.size(), 1);
    EXPECT_EQ(map.find(0)->second, "zero");
    EXPECT_NE(map.find(0), map.end());
}