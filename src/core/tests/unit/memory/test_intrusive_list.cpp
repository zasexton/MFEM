#include <gtest/gtest.h>
#include <core/memory/intrusive_list.h>
#include <vector>
#include <algorithm>

namespace fcm = fem::core::memory;

// Test object that can be used in intrusive lists
struct TestItem {
    int value;
    fcm::intrusive_list_node hook{};

    explicit TestItem(int v) : value(v) {}

    bool operator==(const TestItem& other) const {
        return value == other.value;
    }
};

using TestList = fcm::intrusive_list<TestItem, &TestItem::hook>;

class IntrusiveListTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Basic functionality tests
TEST_F(IntrusiveListTest, EmptyList) {
    TestList list;

    EXPECT_TRUE(list.empty());
    EXPECT_EQ(list.size(), 0u);
    EXPECT_EQ(list.begin(), list.end());
}

TEST_F(IntrusiveListTest, PushFront) {
    TestList list;
    TestItem item1(1);
    TestItem item2(2);
    TestItem item3(3);

    list.push_front(item1);
    EXPECT_FALSE(list.empty());
    EXPECT_EQ(list.begin()->value, 1);

    list.push_front(item2);
    EXPECT_EQ(list.begin()->value, 2);

    list.push_front(item3);
    EXPECT_EQ(list.begin()->value, 3);
}

TEST_F(IntrusiveListTest, PushBack) {
    TestList list;
    TestItem item1(1);
    TestItem item2(2);
    TestItem item3(3);

    list.push_back(item1);
    EXPECT_FALSE(list.empty());
    EXPECT_EQ(list.begin()->value, 1);

    list.push_back(item2);
    EXPECT_EQ(list.begin()->value, 1); // Still first element

    list.push_back(item3);
    EXPECT_EQ(list.begin()->value, 1); // Still first element
}

TEST_F(IntrusiveListTest, BasicIterator) {
    TestList list;
    TestItem item1(10);
    TestItem item2(20);

    list.push_back(item1);
    list.push_back(item2);

    auto it = list.begin();
    EXPECT_NE(it, list.end());
    EXPECT_EQ(it->value, 10);

    ++it;
    EXPECT_EQ(it->value, 20);

    ++it;
    EXPECT_EQ(it, list.end());
}

TEST_F(IntrusiveListTest, EraseElement) {
    TestList list;
    TestItem item1(1);
    TestItem item2(2);
    TestItem item3(3);

    list.push_back(item1);
    list.push_back(item2);
    list.push_back(item3);

    EXPECT_FALSE(list.empty());
    EXPECT_EQ(list.size(), 3u);

    // Erase middle element
    list.erase(item2);
    EXPECT_EQ(list.size(), 2u);
    EXPECT_EQ(list.begin()->value, 1);

    // Erase first element
    list.erase(item1);
    EXPECT_EQ(list.size(), 1u);
    EXPECT_EQ(list.begin()->value, 3);

    // Erase last element
    list.erase(item3);
    EXPECT_TRUE(list.empty());
    EXPECT_EQ(list.size(), 0u);
}

TEST_F(IntrusiveListTest, Clear) {
    TestList list;
    TestItem item1(1);
    TestItem item2(2);
    TestItem item3(3);

    list.push_back(item1);
    list.push_back(item2);
    list.push_back(item3);
    EXPECT_EQ(list.size(), 3u);

    list.clear();
    EXPECT_TRUE(list.empty());
    EXPECT_EQ(list.size(), 0u);
    EXPECT_EQ(list.begin(), list.end());
}

TEST_F(IntrusiveListTest, SingleElement) {
    TestList list;
    TestItem item(42);

    list.push_back(item);
    EXPECT_EQ(list.size(), 1u);
    EXPECT_FALSE(list.empty());

    auto it = list.begin();
    EXPECT_NE(it, list.end());
    EXPECT_EQ(it->value, 42);

    ++it;
    EXPECT_EQ(it, list.end());

    list.erase(item);
    EXPECT_TRUE(list.empty());
}

TEST_F(IntrusiveListTest, NodeLinkedState) {
    TestList list;
    TestItem item(1);

    // Initially not linked
    EXPECT_FALSE(item.hook.linked());

    list.push_back(item);
    EXPECT_TRUE(item.hook.linked());

    list.erase(item);
    EXPECT_FALSE(item.hook.linked());
}

TEST_F(IntrusiveListTest, DestructorClearslist) {
    TestItem item1(1);
    TestItem item2(2);
    TestItem item3(3);

    {
        TestList list;
        list.push_back(item1);
        list.push_back(item2);
        list.push_back(item3);
        // List destructor should clear all items
    }

    // All items should be unlinked after list destruction
    EXPECT_FALSE(item1.hook.linked());
    EXPECT_FALSE(item2.hook.linked());
    EXPECT_FALSE(item3.hook.linked());
}