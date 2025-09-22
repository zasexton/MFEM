#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <utility>

#include <memory/flat_set.h>

namespace fcm = fem::core::memory;

class FlatSetTest : public ::testing::Test {
protected:
    using IntSet = fcm::flat_set<int>;
    using StringSet = fcm::flat_set<std::string>;

    struct ReverseCompare {
        bool operator()(int a, int b) const {
            return a > b;
        }
    };
    using ReverseSet = fcm::flat_set<int, ReverseCompare>;

    struct MoveTracker {
        int value;
        mutable int moves = 0;

        MoveTracker() : value(0), moves(0) {}
        explicit MoveTracker(int v) : value(v), moves(0) {}
        MoveTracker(const MoveTracker& other) : value(other.value), moves(other.moves) {}
        MoveTracker(MoveTracker&& other) noexcept : value(other.value), moves(other.moves + 1) {
            other.moves++;
        }
        MoveTracker& operator=(const MoveTracker& other) {
            value = other.value;
            moves = other.moves;
            return *this;
        }
        MoveTracker& operator=(MoveTracker&& other) noexcept {
            value = other.value;
            moves = other.moves + 1;
            other.moves++;
            return *this;
        }

        bool operator<(const MoveTracker& other) const {
            return value < other.value;
        }
        bool operator==(const MoveTracker& other) const {
            return value == other.value;
        }
    };
    using TrackerSet = fcm::flat_set<MoveTracker>;
};

// === Construction Tests ===

TEST_F(FlatSetTest, DefaultConstruction) {
    IntSet set;
    EXPECT_TRUE(set.empty());
    EXPECT_EQ(set.size(), 0);
    EXPECT_EQ(set.begin(), set.end());
}

TEST_F(FlatSetTest, MemoryResourceConstruction) {
    auto mr = fcm::default_resource();
    IntSet set(mr);
    EXPECT_TRUE(set.empty());
    EXPECT_EQ(set.size(), 0);
}

TEST_F(FlatSetTest, AllocatorConstruction) {
    fcm::polymorphic_allocator<int> alloc;
    IntSet set(alloc);
    EXPECT_TRUE(set.empty());
    EXPECT_EQ(set.size(), 0);
}

TEST_F(FlatSetTest, CustomComparatorConstruction) {
    ReverseSet set;
    EXPECT_TRUE(set.empty());
    EXPECT_EQ(set.size(), 0);
}

// === Basic Operations Tests ===

TEST_F(FlatSetTest, SingleInsert) {
    IntSet set;
    auto result = set.insert(42);

    EXPECT_TRUE(result.second);
    EXPECT_EQ(*result.first, 42);
    EXPECT_FALSE(set.empty());
    EXPECT_EQ(set.size(), 1);
}

TEST_F(FlatSetTest, DuplicateInsert) {
    IntSet set;
    auto result1 = set.insert(42);
    auto result2 = set.insert(42);

    EXPECT_TRUE(result1.second);
    EXPECT_FALSE(result2.second);
    EXPECT_EQ(result1.first, result2.first);
    EXPECT_EQ(set.size(), 1);
}

TEST_F(FlatSetTest, MultipleInserts) {
    IntSet set;
    std::vector<int> values = {5, 2, 8, 1, 9, 3};

    for (int val : values) {
        set.insert(val);
    }

    EXPECT_EQ(set.size(), 6);

    // Should be sorted
    std::vector<int> result(set.begin(), set.end());
    std::vector<int> expected = {1, 2, 3, 5, 8, 9};
    EXPECT_EQ(result, expected);
}

TEST_F(FlatSetTest, EmplaceConstruction) {
    StringSet set;
    auto result = set.emplace("hello");

    EXPECT_TRUE(result.second);
    EXPECT_EQ(*result.first, "hello");
    EXPECT_EQ(set.size(), 1);
}

TEST_F(FlatSetTest, EmplaceWithArgs) {
    StringSet set;
    auto result = set.emplace(5, 'x');  // string(5, 'x') -> "xxxxx"

    EXPECT_TRUE(result.second);
    EXPECT_EQ(*result.first, "xxxxx");
    EXPECT_EQ(set.size(), 1);
}

// === Find and Search Tests ===

TEST_F(FlatSetTest, FindExisting) {
    IntSet set;
    set.insert(1);
    set.insert(2);
    set.insert(3);

    auto it = set.find(2);
    EXPECT_NE(it, set.end());
    EXPECT_EQ(*it, 2);
}

TEST_F(FlatSetTest, FindNonexistent) {
    IntSet set;
    set.insert(1);
    set.insert(3);

    auto it = set.find(2);
    EXPECT_EQ(it, set.end());
}

TEST_F(FlatSetTest, FindConst) {
    IntSet set;
    set.insert(42);

    const IntSet& const_set = set;
    auto it = const_set.find(42);
    EXPECT_NE(it, const_set.end());
    EXPECT_EQ(*it, 42);
}

TEST_F(FlatSetTest, FindEmpty) {
    IntSet set;
    auto it = set.find(42);
    EXPECT_EQ(it, set.end());
}

// === Erase Tests ===

TEST_F(FlatSetTest, EraseExisting) {
    IntSet set;
    set.insert(1);
    set.insert(2);
    set.insert(3);

    std::size_t removed = set.erase(2);
    EXPECT_EQ(removed, 1);
    EXPECT_EQ(set.size(), 2);
    EXPECT_EQ(set.find(2), set.end());

    std::vector<int> result(set.begin(), set.end());
    std::vector<int> expected = {1, 3};
    EXPECT_EQ(result, expected);
}

TEST_F(FlatSetTest, EraseNonexistent) {
    IntSet set;
    set.insert(1);
    set.insert(3);

    std::size_t removed = set.erase(2);
    EXPECT_EQ(removed, 0);
    EXPECT_EQ(set.size(), 2);
}

TEST_F(FlatSetTest, EraseEmpty) {
    IntSet set;
    std::size_t removed = set.erase(42);
    EXPECT_EQ(removed, 0);
    EXPECT_TRUE(set.empty());
}

TEST_F(FlatSetTest, EraseAll) {
    IntSet set;
    set.insert(1);
    set.insert(2);
    set.insert(3);

    set.erase(1);
    set.erase(2);
    set.erase(3);

    EXPECT_TRUE(set.empty());
    EXPECT_EQ(set.size(), 0);
}

// === Iterator Tests ===

TEST_F(FlatSetTest, IteratorTraversal) {
    IntSet set;
    std::vector<int> values = {3, 1, 4, 1, 5, 9, 2, 6};

    for (int val : values) {
        set.insert(val);
    }

    std::vector<int> result;
    for (auto it = set.begin(); it != set.end(); ++it) {
        result.push_back(*it);
    }

    std::vector<int> expected = {1, 2, 3, 4, 5, 6, 9};
    EXPECT_EQ(result, expected);
}

TEST_F(FlatSetTest, RangeBasedFor) {
    IntSet set;
    std::vector<int> values = {5, 2, 8, 1, 9};

    for (int val : values) {
        set.insert(val);
    }

    std::vector<int> result;
    for (int val : set) {
        result.push_back(val);
    }

    std::vector<int> expected = {1, 2, 5, 8, 9};
    EXPECT_EQ(result, expected);
}

TEST_F(FlatSetTest, ConstIterator) {
    IntSet set;
    set.insert(1);
    set.insert(2);
    set.insert(3);

    const IntSet& const_set = set;
    std::vector<int> result;
    for (auto it = const_set.begin(); it != const_set.end(); ++it) {
        result.push_back(*it);
    }

    std::vector<int> expected = {1, 2, 3};
    EXPECT_EQ(result, expected);
}

TEST_F(FlatSetTest, EmptyIterators) {
    IntSet set;
    EXPECT_EQ(set.begin(), set.end());

    const IntSet& const_set = set;
    EXPECT_EQ(const_set.begin(), const_set.end());
}

// === Custom Comparator Tests ===

TEST_F(FlatSetTest, ReverseComparator) {
    ReverseSet set;
    std::vector<int> values = {3, 1, 4, 1, 5, 9, 2, 6};

    for (int val : values) {
        set.insert(val);
    }

    std::vector<int> result(set.begin(), set.end());
    std::vector<int> expected = {9, 6, 5, 4, 3, 2, 1};
    EXPECT_EQ(result, expected);
}

TEST_F(FlatSetTest, ReverseFind) {
    ReverseSet set;
    set.insert(1);
    set.insert(2);
    set.insert(3);

    auto it = set.find(2);
    EXPECT_NE(it, set.end());
    EXPECT_EQ(*it, 2);
}

TEST_F(FlatSetTest, ReverseErase) {
    ReverseSet set;
    set.insert(1);
    set.insert(2);
    set.insert(3);

    std::size_t removed = set.erase(2);
    EXPECT_EQ(removed, 1);
    EXPECT_EQ(set.size(), 2);

    std::vector<int> result(set.begin(), set.end());
    std::vector<int> expected = {3, 1};
    EXPECT_EQ(result, expected);
}

// === String Operations Tests ===

TEST_F(FlatSetTest, StringOperations) {
    StringSet set;
    set.insert("zebra");
    set.insert("apple");
    set.insert("banana");
    set.insert("apple");  // duplicate

    EXPECT_EQ(set.size(), 3);

    std::vector<std::string> result(set.begin(), set.end());
    std::vector<std::string> expected = {"apple", "banana", "zebra"};
    EXPECT_EQ(result, expected);
}

TEST_F(FlatSetTest, StringFind) {
    StringSet set;
    set.insert("hello");
    set.insert("world");

    auto it = set.find("hello");
    EXPECT_NE(it, set.end());
    EXPECT_EQ(*it, "hello");

    it = set.find("missing");
    EXPECT_EQ(it, set.end());
}

// === Move Semantics Tests ===

TEST_F(FlatSetTest, MoveInsert) {
    TrackerSet set;
    MoveTracker tracker(42);

    auto result = set.insert(std::move(tracker));
    EXPECT_TRUE(result.second);
    EXPECT_EQ(result.first->value, 42);

    // Should have moved
    EXPECT_GT(result.first->moves, 0);
}

TEST_F(FlatSetTest, MoveEmplace) {
    TrackerSet set;
    auto result = set.emplace(42);

    EXPECT_TRUE(result.second);
    EXPECT_EQ(result.first->value, 42);
    // emplace() constructs temporary Key then moves to insert, so expect moves
    EXPECT_GT(result.first->moves, 0);
}

// === Clear and Utility Tests ===

TEST_F(FlatSetTest, Clear) {
    IntSet set;
    set.insert(1);
    set.insert(2);
    set.insert(3);

    EXPECT_FALSE(set.empty());
    EXPECT_EQ(set.size(), 3);

    set.clear();

    EXPECT_TRUE(set.empty());
    EXPECT_EQ(set.size(), 0);
    EXPECT_EQ(set.begin(), set.end());
}

TEST_F(FlatSetTest, Reserve) {
    IntSet set;
    set.reserve(100);

    // Reserve doesn't change logical size
    EXPECT_TRUE(set.empty());
    EXPECT_EQ(set.size(), 0);

    // But we can insert without reallocation concerns
    for (int i = 0; i < 50; ++i) {
        set.insert(i);
    }
    EXPECT_EQ(set.size(), 50);
}

// === Edge Cases and Error Conditions ===

TEST_F(FlatSetTest, LargeDataset) {
    IntSet set;
    const int N = 1000;

    // Insert in reverse order to test sorting
    for (int i = N; i >= 1; --i) {
        set.insert(i);
    }

    EXPECT_EQ(set.size(), N);

    // Verify sorted order
    int prev = 0;
    for (int val : set) {
        EXPECT_GT(val, prev);
        prev = val;
    }
    EXPECT_EQ(prev, N);
}

TEST_F(FlatSetTest, FindAfterClear) {
    IntSet set;
    set.insert(42);

    EXPECT_NE(set.find(42), set.end());

    set.clear();

    EXPECT_EQ(set.find(42), set.end());
}

TEST_F(FlatSetTest, MultipleEraseFind) {
    IntSet set;
    for (int i = 1; i <= 10; ++i) {
        set.insert(i);
    }

    // Erase every other element
    for (int i = 2; i <= 10; i += 2) {
        std::size_t removed = set.erase(i);
        EXPECT_EQ(removed, 1);
    }

    EXPECT_EQ(set.size(), 5);

    // Check remaining elements
    for (int i = 1; i <= 10; i += 2) {
        EXPECT_NE(set.find(i), set.end());
    }
    for (int i = 2; i <= 10; i += 2) {
        EXPECT_EQ(set.find(i), set.end());
    }
}

TEST_F(FlatSetTest, DuplicateOperationsStress) {
    IntSet set;

    // Insert the same element multiple times
    for (int i = 0; i < 100; ++i) {
        auto result = set.insert(42);
        if (i == 0) {
            EXPECT_TRUE(result.second);
        } else {
            EXPECT_FALSE(result.second);
        }
        EXPECT_EQ(*result.first, 42);
    }

    EXPECT_EQ(set.size(), 1);
    EXPECT_EQ(*set.begin(), 42);
}

// === Comparison with std::set behavior ===

TEST_F(FlatSetTest, SetLikeBehavior) {
    IntSet flat_set;
    std::set<int> std_set;
    std::vector<int> values = {5, 2, 8, 2, 1, 9, 5, 3};

    for (int val : values) {
        auto flat_result = flat_set.insert(val);
        auto std_result = std_set.insert(val);

        EXPECT_EQ(flat_result.second, std_result.second);
        EXPECT_EQ(*flat_result.first, *std_result.first);
    }

    EXPECT_EQ(flat_set.size(), std_set.size());

    std::vector<int> flat_values(flat_set.begin(), flat_set.end());
    std::vector<int> std_values(std_set.begin(), std_set.end());
    EXPECT_EQ(flat_values, std_values);
}