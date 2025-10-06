#include <gtest/gtest.h>
#include <core/concurrency/work_stealing_pool.h>
#include <thread>
#include <vector>
#include <atomic>
#include <algorithm>
#include <set>

namespace fcc = fem::core::concurrency;

// Test fixture
class ChaseLevDequeTest : public ::testing::Test {
protected:
    using Deque = fcc::ChaseLevDeque<int>;

    void SetUp() override {}
    void TearDown() override {}
};

// ====================  Basic Functionality Tests ====================

TEST_F(ChaseLevDequeTest, DefaultConstruction) {
    Deque deque;
    EXPECT_TRUE(deque.empty());
    EXPECT_EQ(deque.size(), 0);
}

TEST_F(ChaseLevDequeTest, SinglePushPop) {
    Deque deque;
    int value = 42;

    deque.push(&value);
    EXPECT_FALSE(deque.empty());
    EXPECT_EQ(deque.size(), 1);

    int* result = deque.pop();
    EXPECT_EQ(result, &value);
    EXPECT_EQ(*result, 42);
    EXPECT_TRUE(deque.empty());
}

TEST_F(ChaseLevDequeTest, PopFromEmpty) {
    Deque deque;
    EXPECT_EQ(deque.pop(), nullptr);
}

TEST_F(ChaseLevDequeTest, StealFromEmpty) {
    Deque deque;
    EXPECT_EQ(deque.steal(), nullptr);
}

TEST_F(ChaseLevDequeTest, MultiplePushPop) {
    Deque deque;
    std::vector<int> values = {1, 2, 3, 4, 5};

    for (auto& v : values) {
        deque.push(&v);
    }

    EXPECT_EQ(deque.size(), 5);

    // Pop should return in LIFO order
    for (int i = 4; i >= 0; --i) {
        int* result = deque.pop();
        EXPECT_EQ(*result, values[i]);
    }

    EXPECT_TRUE(deque.empty());
}

TEST_F(ChaseLevDequeTest, SinglePushSteal) {
    Deque deque;
    int value = 100;

    deque.push(&value);

    int* result = deque.steal();
    EXPECT_EQ(result, &value);
    EXPECT_EQ(*result, 100);
    EXPECT_TRUE(deque.empty());
}

TEST_F(ChaseLevDequeTest, MultiplePushSteal) {
    Deque deque;
    std::vector<int> values = {10, 20, 30, 40, 50};

    for (auto& v : values) {
        deque.push(&v);
    }

    // Steal should return in FIFO order
    for (size_t i = 0; i < values.size(); ++i) {
        int* result = deque.steal();
        EXPECT_EQ(*result, values[i]);
    }

    EXPECT_TRUE(deque.empty());
}

TEST_F(ChaseLevDequeTest, MixedPushPopSteal) {
    Deque deque;
    std::vector<int> values = {1, 2, 3, 4, 5, 6, 7, 8};

    for (auto& v : values) {
        deque.push(&v);
    }

    // Pop a few (LIFO from bottom)
    EXPECT_EQ(*deque.pop(), 8);
    EXPECT_EQ(*deque.pop(), 7);

    // Steal a few (FIFO from top)
    EXPECT_EQ(*deque.steal(), 1);
    EXPECT_EQ(*deque.steal(), 2);

    // Pop again
    EXPECT_EQ(*deque.pop(), 6);

    // Steal again
    EXPECT_EQ(*deque.steal(), 3);

    EXPECT_EQ(deque.size(), 2);
}

// ====================  Concurrent Tests ====================

TEST_F(ChaseLevDequeTest, ConcurrentPushPop) {
    Deque deque;
    const size_t num_items = 10000;
    std::vector<int> values(num_items);

    for (size_t i = 0; i < num_items; ++i) {
        values[i] = static_cast<int>(i);
    }

    // Owner thread: push and pop
    std::thread owner([&]() {
        for (auto& v : values) {
            deque.push(&v);
        }

        for (size_t i = 0; i < num_items; ++i) {
            int* result = deque.pop();
            EXPECT_NE(result, nullptr);
        }
    });

    owner.join();
    EXPECT_TRUE(deque.empty());
}

TEST_F(ChaseLevDequeTest, SingleThiefSteal) {
    Deque deque;
    const size_t num_items = 1000;
    std::vector<int> values(num_items);
    std::atomic<size_t> stolen_count{0};

    for (size_t i = 0; i < num_items; ++i) {
        values[i] = static_cast<int>(i);
        deque.push(&values[i]);
    }

    std::thread thief([&]() {
        while (true) {
            int* result = deque.steal();
            if (result == nullptr) {
                break;
            }
            stolen_count.fetch_add(1);
        }
    });

    thief.join();
    EXPECT_GT(stolen_count.load(), 0);
}

TEST_F(ChaseLevDequeTest, OwnerPopThiefSteal) {
    Deque deque;
    const size_t num_items = 10000;
    std::vector<int> values(num_items);
    std::atomic<size_t> owner_popped{0};
    std::atomic<size_t> thief_stolen{0};
    std::atomic<bool> done{false};

    for (size_t i = 0; i < num_items; ++i) {
        values[i] = static_cast<int>(i);
    }

    // Owner thread: push then pop
    std::thread owner([&]() {
        for (auto& v : values) {
            deque.push(&v);
        }

        while (owner_popped.load() + thief_stolen.load() < num_items) {
            int* result = deque.pop();
            if (result != nullptr) {
                owner_popped.fetch_add(1);
            }
        }
        done.store(true);
    });

    // Thief thread: steal
    std::thread thief([&]() {
        while (!done.load() || !deque.empty()) {
            int* result = deque.steal();
            if (result != nullptr) {
                thief_stolen.fetch_add(1);
            }
        }
    });

    owner.join();
    thief.join();

    EXPECT_EQ(owner_popped.load() + thief_stolen.load(), num_items);
    EXPECT_TRUE(deque.empty());
}

TEST_F(ChaseLevDequeTest, MultipleThieves) {
    Deque deque;
    const size_t num_items = 10000;
    const size_t num_thieves = 4;
    std::vector<int> values(num_items);
    std::atomic<size_t> total_stolen{0};
    std::atomic<bool> pushing_done{false};

    for (size_t i = 0; i < num_items; ++i) {
        values[i] = static_cast<int>(i);
    }

    // Owner thread: push items
    std::thread owner([&]() {
        for (auto& v : values) {
            deque.push(&v);
        }
        pushing_done.store(true);
    });

    // Thief threads: steal items
    std::vector<std::thread> thieves;
    for (size_t t = 0; t < num_thieves; ++t) {
        thieves.emplace_back([&]() {
            while (!pushing_done.load() || !deque.empty()) {
                int* result = deque.steal();
                if (result != nullptr) {
                    total_stolen.fetch_add(1);
                }
            }
        });
    }

    owner.join();
    for (auto& thief : thieves) {
        thief.join();
    }

    EXPECT_EQ(total_stolen.load(), num_items);
    EXPECT_TRUE(deque.empty());
}

TEST_F(ChaseLevDequeTest, NoDoubleExecution) {
    Deque deque;
    const size_t num_items = 10000;
    std::vector<int> values(num_items);
    std::vector<std::atomic<int>> execution_count(num_items);
    std::atomic<bool> done{false};

    for (size_t i = 0; i < num_items; ++i) {
        values[i] = static_cast<int>(i);
        execution_count[i].store(0);
        deque.push(&values[i]);
    }

    // Owner thread
    std::thread owner([&]() {
        while (!deque.empty()) {
            int* result = deque.pop();
            if (result != nullptr) {
                size_t idx = result - &values[0];
                execution_count[idx].fetch_add(1);
            }
        }
        done.store(true);
    });

    // Thief threads
    std::vector<std::thread> thieves;
    for (int t = 0; t < 4; ++t) {
        thieves.emplace_back([&]() {
            while (!done.load()) {
                int* result = deque.steal();
                if (result != nullptr) {
                    size_t idx = result - &values[0];
                    execution_count[idx].fetch_add(1);
                }
            }
        });
    }

    owner.join();
    for (auto& thief : thieves) {
        thief.join();
    }

    // Verify each item was executed exactly once
    for (size_t i = 0; i < num_items; ++i) {
        EXPECT_EQ(execution_count[i].load(), 1)
            << "Item " << i << " executed " << execution_count[i].load() << " times";
    }
}

// ====================  Stress Tests ====================

TEST_F(ChaseLevDequeTest, StressTestLargeWorkload) {
    Deque deque;
    const size_t num_items = 100000;
    std::vector<int> values(num_items);
    std::atomic<size_t> owner_ops{0};
    std::atomic<size_t> thief_ops{0};
    std::atomic<bool> pushing_done{false};

    for (size_t i = 0; i < num_items; ++i) {
        values[i] = static_cast<int>(i);
    }

    std::thread owner([&]() {
        // Push all items
        for (auto& v : values) {
            deque.push(&v);
        }
        pushing_done.store(true);

        // Owner pops until empty
        while (!deque.empty() || !pushing_done.load()) {
            int* result = deque.pop();
            if (result) owner_ops.fetch_add(1);
        }
    });

    std::vector<std::thread> thieves;
    for (int t = 0; t < 8; ++t) {
        thieves.emplace_back([&]() {
            // Thieves steal until done
            while (!pushing_done.load() || !deque.empty()) {
                int* result = deque.steal();
                if (result) {
                    thief_ops.fetch_add(1);
                }
            }
        });
    }

    owner.join();
    for (auto& thief : thieves) {
        thief.join();
    }

    EXPECT_EQ(owner_ops.load() + thief_ops.load(), num_items);
}

TEST_F(ChaseLevDequeTest, StressTestResizing) {
    Deque deque;
    const size_t num_items = 10000;  // More than INITIAL_SIZE to force resize
    std::vector<int> values(num_items);

    for (size_t i = 0; i < num_items; ++i) {
        values[i] = static_cast<int>(i);
        deque.push(&values[i]);
    }

    EXPECT_GE(deque.size(), num_items);

    // Now steal while owner is also popping
    std::atomic<size_t> total{0};

    std::thread owner([&]() {
        while (true) {
            int* result = deque.pop();
            if (!result) break;
            total.fetch_add(1);
        }
    });

    std::thread thief([&]() {
        while (true) {
            int* result = deque.steal();
            if (!result) {
                std::this_thread::yield();
                if (deque.empty()) break;
            } else {
                total.fetch_add(1);
            }
        }
    });

    owner.join();
    thief.join();

    EXPECT_EQ(total.load(), num_items);
}
