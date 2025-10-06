#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <random>
#include <thread>
#include <vector>

#include "core/concurrency/task_graph.h"
#include "core/concurrency/thread_pool.h"

namespace fcc = fem::core::concurrency;

class TaskGraphTest : public ::testing::Test {
protected:
    void SetUp() override {
        pool_ = std::make_unique<fcc::ThreadPool>(4);
        counter_ = 0;
        execution_order_.clear();
    }

    void TearDown() override {
        pool_.reset();
    }

    std::unique_ptr<fcc::ThreadPool> pool_;
    std::atomic<int> counter_;
    std::vector<std::string> execution_order_;
    std::mutex order_mutex_;

    void record_execution(const std::string& task_id) {
        std::lock_guard<std::mutex> lock(order_mutex_);
        execution_order_.push_back(task_id);
    }
};

// ==================== Basic Construction Tests ====================

TEST_F(TaskGraphTest, DefaultConstruction) {
    fcc::TaskGraph<int> graph;
    EXPECT_EQ(graph.size(), 0);
    EXPECT_TRUE(graph.empty());
    EXPECT_TRUE(graph.validate());
}

TEST_F(TaskGraphTest, ConstructionWithThreadPool) {
    fcc::TaskGraph<int> graph(pool_.get());
    EXPECT_EQ(graph.size(), 0);
    EXPECT_TRUE(graph.empty());
}

// ==================== Task Addition Tests ====================

TEST_F(TaskGraphTest, AddSingleTask) {
    fcc::TaskGraph<int> graph;

    auto node = graph.add_task("task1", []() { return 42; });

    EXPECT_EQ(graph.size(), 1);
    EXPECT_FALSE(graph.empty());
    EXPECT_EQ(node->id(), "task1");
    EXPECT_EQ(node->priority(), 0);
}

TEST_F(TaskGraphTest, AddMultipleTasks) {
    fcc::TaskGraph<int> graph;

    graph.add_task("task1", []() { return 1; });
    graph.add_task("task2", []() { return 2; });
    graph.add_task("task3", []() { return 3; });

    EXPECT_EQ(graph.size(), 3);
}

TEST_F(TaskGraphTest, AddTaskWithPriority) {
    fcc::TaskGraph<int> graph;

    auto node = graph.add_task("high_priority", []() { return 99; }, 10);

    EXPECT_EQ(node->priority(), 10);
}

TEST_F(TaskGraphTest, AddDuplicateTaskThrows) {
    fcc::TaskGraph<int> graph;

    graph.add_task("task1", []() { return 42; });

    EXPECT_THROW(
        graph.add_task("task1", []() { return 99; }),
        std::invalid_argument
    );
}

// ==================== Dependency Tests ====================

TEST_F(TaskGraphTest, AddSimpleDependency) {
    fcc::TaskGraph<int> graph;

    graph.add_task("task1", []() { return 1; });
    graph.add_task("task2", []() { return 2; });

    EXPECT_NO_THROW(graph.add_dependency("task2", "task1"));
    EXPECT_TRUE(graph.validate());
}

TEST_F(TaskGraphTest, AddMultipleDependencies) {
    fcc::TaskGraph<int> graph;

    graph.add_task("task1", []() { return 1; });
    graph.add_task("task2", []() { return 2; });
    graph.add_task("task3", []() { return 3; });
    graph.add_task("task4", []() { return 4; });

    graph.add_dependencies("task4", {"task1", "task2", "task3"});

    EXPECT_TRUE(graph.validate());
}

TEST_F(TaskGraphTest, AddDependencyNonExistentTaskThrows) {
    fcc::TaskGraph<int> graph;

    graph.add_task("task1", []() { return 1; });

    EXPECT_THROW(
        graph.add_dependency("task1", "nonexistent"),
        fcc::TaskNotFoundException
    );

    EXPECT_THROW(
        graph.add_dependency("nonexistent", "task1"),
        fcc::TaskNotFoundException
    );
}

// ==================== Cycle Detection Tests ====================

TEST_F(TaskGraphTest, DetectSimpleCycle) {
    fcc::TaskGraph<int> graph;

    graph.add_task("A", []() { return 1; });
    graph.add_task("B", []() { return 2; });

    graph.add_dependency("A", "B");
    graph.add_dependency("B", "A");

    EXPECT_TRUE(graph.has_cycle());
    EXPECT_FALSE(graph.validate());
}

TEST_F(TaskGraphTest, DetectComplexCycle) {
    fcc::TaskGraph<int> graph;

    graph.add_task("A", []() { return 1; });
    graph.add_task("B", []() { return 2; });
    graph.add_task("C", []() { return 3; });
    graph.add_task("D", []() { return 4; });

    graph.add_dependency("B", "A");
    graph.add_dependency("C", "B");
    graph.add_dependency("D", "C");
    graph.add_dependency("B", "D"); // Creates cycle B->C->D->B

    EXPECT_TRUE(graph.has_cycle());
    EXPECT_FALSE(graph.validate());
}

TEST_F(TaskGraphTest, NoCycleInDAG) {
    fcc::TaskGraph<int> graph;

    graph.add_task("A", []() { return 1; });
    graph.add_task("B", []() { return 2; });
    graph.add_task("C", []() { return 3; });
    graph.add_task("D", []() { return 4; });

    graph.add_dependency("B", "A");
    graph.add_dependency("C", "A");
    graph.add_dependency("D", "B");
    graph.add_dependency("D", "C");

    EXPECT_FALSE(graph.has_cycle());
    EXPECT_TRUE(graph.validate());
}

// ==================== Topological Sort Tests ====================

TEST_F(TaskGraphTest, TopologicalSortLinearChain) {
    fcc::TaskGraph<int> graph;

    graph.add_task("A", []() { return 1; });
    graph.add_task("B", []() { return 2; });
    graph.add_task("C", []() { return 3; });

    graph.add_dependency("B", "A");
    graph.add_dependency("C", "B");

    auto order = graph.topological_sort();

    EXPECT_EQ(order.size(), 3);

    // A must come before B, B must come before C
    auto a_pos = std::find(order.begin(), order.end(), "A");
    auto b_pos = std::find(order.begin(), order.end(), "B");
    auto c_pos = std::find(order.begin(), order.end(), "C");

    EXPECT_LT(a_pos, b_pos);
    EXPECT_LT(b_pos, c_pos);
}

TEST_F(TaskGraphTest, TopologicalSortDiamond) {
    fcc::TaskGraph<int> graph;

    graph.add_task("A", []() { return 1; });
    graph.add_task("B", []() { return 2; });
    graph.add_task("C", []() { return 3; });
    graph.add_task("D", []() { return 4; });

    graph.add_dependency("B", "A");
    graph.add_dependency("C", "A");
    graph.add_dependency("D", "B");
    graph.add_dependency("D", "C");

    auto order = graph.topological_sort();

    EXPECT_EQ(order.size(), 4);

    // A must come first, D must come last
    EXPECT_EQ(order.front(), "A");
    EXPECT_EQ(order.back(), "D");
}

TEST_F(TaskGraphTest, TopologicalSortWithCycleThrows) {
    fcc::TaskGraph<int> graph;

    graph.add_task("A", []() { return 1; });
    graph.add_task("B", []() { return 2; });

    graph.add_dependency("A", "B");
    graph.add_dependency("B", "A");

    EXPECT_THROW(
        graph.topological_sort(),
        fcc::CyclicDependencyException
    );
}

// ==================== Execution Tests ====================

TEST_F(TaskGraphTest, ExecuteSingleTask) {
    fcc::TaskGraph<int> graph;

    bool executed = false;
    graph.add_task("task1", [&executed]() {
        executed = true;
        return 42;
    });

    graph.execute(false); // Sequential execution

    EXPECT_TRUE(executed);

    auto node = graph.get_task("task1");
    EXPECT_EQ(node->get(), 42);
}

TEST_F(TaskGraphTest, ExecuteLinearChain) {
    fcc::TaskGraph<void> graph;

    graph.add_task("A", [this]() {
        record_execution("A");
        counter_.fetch_add(1);
    });
    graph.add_task("B", [this]() {
        record_execution("B");
        counter_.fetch_add(10);
    });
    graph.add_task("C", [this]() {
        record_execution("C");
        counter_.fetch_add(100);
    });

    graph.add_dependency("B", "A");
    graph.add_dependency("C", "B");

    graph.execute(false);

    EXPECT_EQ(counter_.load(), 111);
    EXPECT_EQ(execution_order_, std::vector<std::string>({"A", "B", "C"}));
}

TEST_F(TaskGraphTest, ExecuteParallelTasks) {
    fcc::TaskGraph<int> graph(pool_.get());

    std::atomic<int> parallel_count{0};

    graph.add_task("A", []() { return 1; });

    graph.add_task("B", [&parallel_count]() {
        parallel_count.fetch_add(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return 2;
    });

    graph.add_task("C", [&parallel_count]() {
        parallel_count.fetch_add(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return 3;
    });

    graph.add_task("D", []() { return 4; });

    graph.add_dependency("B", "A");
    graph.add_dependency("C", "A");
    graph.add_dependency("D", "B");
    graph.add_dependency("D", "C");

    graph.execute(true); // Parallel execution

    // B and C should have executed in parallel
    EXPECT_EQ(parallel_count.load(), 2);
}

TEST_F(TaskGraphTest, ExecuteDiamondPattern) {
    fcc::TaskGraph<int> graph;

    int result = 0;

    graph.add_task("start", []() { return 10; });
    graph.add_task("left", []() { return 20; });
    graph.add_task("right", []() { return 30; });
    graph.add_task("end", [&result]() {
        result = 100;
        return result;
    });

    graph.add_dependency("left", "start");
    graph.add_dependency("right", "start");
    graph.add_dependency("end", "left");
    graph.add_dependency("end", "right");

    graph.execute(true);

    EXPECT_EQ(result, 100);

    auto end_node = graph.get_task("end");
    EXPECT_EQ(end_node->get(), 100);
}

TEST_F(TaskGraphTest, ExecuteWithPriority) {
    fcc::TaskGraph<void> graph(pool_.get());

    // Add tasks with different priorities
    graph.add_task("low", [this]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        record_execution("low");
    }, 1);

    graph.add_task("high", [this]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        record_execution("high");
    }, 10);

    graph.add_task("medium", [this]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        record_execution("medium");
    }, 5);

    // All tasks are independent
    graph.execute(true);

    // With priorities, high should generally execute before low
    // (though not guaranteed due to thread scheduling)
    EXPECT_EQ(execution_order_.size(), 3);
}

// ==================== Exception Handling Tests ====================

TEST_F(TaskGraphTest, TaskThrowsException) {
    fcc::TaskGraph<int> graph;

    graph.add_task("failing_task", []() -> int {
        throw std::runtime_error("Task failed");
    });

    graph.execute(false);

    auto node = graph.get_task("failing_task");
    EXPECT_EQ(node->status(), fcc::TaskNode<int>::Status::Failed);

    EXPECT_THROW(node->get(), std::runtime_error);
}

TEST_F(TaskGraphTest, ExceptionInMiddleOfChain) {
    fcc::TaskGraph<void> graph;

    graph.add_task("A", [this]() { record_execution("A"); });
    graph.add_task("B", []() {
        throw std::runtime_error("B failed");
    });
    graph.add_task("C", [this]() { record_execution("C"); });

    graph.add_dependency("B", "A");
    graph.add_dependency("C", "B");

    graph.execute(false);

    // A should have executed, B failed, C should not execute
    EXPECT_EQ(execution_order_.size(), 1);
    EXPECT_EQ(execution_order_[0], "A");

    auto b_node = graph.get_task("B");
    EXPECT_EQ(b_node->status(), fcc::TaskNode<void>::Status::Failed);
}

// ==================== Cancellation Tests ====================

TEST_F(TaskGraphTest, CancelBeforeExecution) {
    fcc::TaskGraph<int> graph;

    bool executed = false;
    graph.add_task("task1", [&executed]() {
        executed = true;
        return 42;
    });

    graph.cancel();

    EXPECT_THROW(graph.execute(false), std::exception);
    EXPECT_FALSE(executed);
}

TEST_F(TaskGraphTest, CancelDuringExecution) {
    fcc::TaskGraph<void> graph(pool_.get());

    std::atomic<bool> started{false};
    std::atomic<bool> should_cancel{false};

    graph.add_task("slow_task", [&started, &should_cancel]() {
        started = true;
        while (!should_cancel) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    auto future = graph.execute_async(true);

    // Wait for task to start
    while (!started) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Cancel the graph
    graph.cancel();
    should_cancel = true;

    // Execution should complete (with cancellation)
    future.wait();

    auto node = graph.get_task("slow_task");
    auto status = node->status();
    EXPECT_TRUE(status == fcc::TaskNode<void>::Status::Cancelled ||
                status == fcc::TaskNode<void>::Status::Completed);
}

// ==================== Async Execution Tests ====================

TEST_F(TaskGraphTest, ExecuteAsync) {
    fcc::TaskGraph<int> graph(pool_.get());

    graph.add_task("task1", []() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return 42;
    });

    auto future = graph.execute_async(true);

    // Should return immediately
    EXPECT_EQ(future.wait_for(std::chrono::seconds(1)),
              std::future_status::ready);

    auto node = graph.get_task("task1");
    EXPECT_EQ(node->get(), 42);
}

// ==================== Wait Tests ====================

TEST_F(TaskGraphTest, WaitForCompletion) {
    fcc::TaskGraph<void> graph(pool_.get());

    graph.add_task("task1", []() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    });

    auto future = graph.execute_async(true);
    graph.wait();

    auto node = graph.get_task("task1");
    EXPECT_EQ(node->status(), fcc::TaskNode<void>::Status::Completed);
}

TEST_F(TaskGraphTest, WaitWithTimeout) {
    fcc::TaskGraph<void> graph(pool_.get());

    graph.add_task("slow_task", []() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    });

    auto future = graph.execute_async(true);

    // Short timeout should fail
    bool completed = graph.wait_for(std::chrono::milliseconds(10));
    EXPECT_FALSE(completed);

    // Long timeout should succeed
    completed = graph.wait_for(std::chrono::milliseconds(200));
    EXPECT_TRUE(completed);
}

// ==================== Root and Leaf Tests ====================

TEST_F(TaskGraphTest, IdentifyRootsAndLeaves) {
    fcc::TaskGraph<int> graph;

    graph.add_task("root1", []() { return 1; });
    graph.add_task("root2", []() { return 2; });
    graph.add_task("middle", []() { return 3; });
    graph.add_task("leaf1", []() { return 4; });
    graph.add_task("leaf2", []() { return 5; });

    graph.add_dependency("middle", "root1");
    graph.add_dependency("middle", "root2");
    graph.add_dependency("leaf1", "middle");
    graph.add_dependency("leaf2", "middle");

    graph.execute(false);

    auto roots = graph.get_roots();
    auto leaves = graph.get_leaves();

    EXPECT_EQ(roots.size(), 2);
    EXPECT_EQ(leaves.size(), 2);

    // Check root IDs
    std::vector<std::string> root_ids;
    for (const auto& root : roots) {
        root_ids.push_back(root->id());
    }
    std::sort(root_ids.begin(), root_ids.end());
    EXPECT_EQ(root_ids, std::vector<std::string>({"root1", "root2"}));

    // Check leaf IDs
    std::vector<std::string> leaf_ids;
    for (const auto& leaf : leaves) {
        leaf_ids.push_back(leaf->id());
    }
    std::sort(leaf_ids.begin(), leaf_ids.end());
    EXPECT_EQ(leaf_ids, std::vector<std::string>({"leaf1", "leaf2"}));
}

// ==================== Statistics Tests ====================

TEST_F(TaskGraphTest, GetStatistics) {
    fcc::TaskGraph<void> graph;

    graph.add_task("task1", []() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    });

    graph.add_task("task2", []() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    });

    graph.add_task("failing_task", []() {
        throw std::runtime_error("Failed");
    });

    graph.add_dependency("task2", "task1");
    graph.add_dependency("failing_task", "task1");

    graph.execute(false);

    auto stats = graph.get_stats();

    EXPECT_EQ(stats.total_tasks, 3);
    EXPECT_EQ(stats.completed_tasks, 2);
    EXPECT_EQ(stats.failed_tasks, 1);
    EXPECT_EQ(stats.cancelled_tasks, 0);
    EXPECT_GT(stats.total_duration.count(), 0);
    EXPECT_GT(stats.average_task_duration.count(), 0);
}

// ==================== Helper Function Tests ====================

TEST_F(TaskGraphTest, MakePipeline) {
    std::vector<std::pair<std::string, std::function<int()>>> tasks = {
        {"stage1", []() { return 1; }},
        {"stage2", []() { return 2; }},
        {"stage3", []() { return 3; }}
    };

    auto graph = fcc::make_pipeline<int>(tasks);

    EXPECT_EQ(graph->size(), 3);

    graph->execute(false);

    auto stage3 = graph->get_task("stage3");
    EXPECT_EQ(stage3->get(), 3);
}

TEST_F(TaskGraphTest, MakeForkJoin) {
    auto graph = fcc::make_fork_join<int>(
        "fork",
        {
            {"parallel1", []() { return 10; }},
            {"parallel2", []() { return 20; }},
            {"parallel3", []() { return 30; }}
        },
        "join",
        []() { return 1; },  // Fork function
        []() { return 100; } // Join function
    );

    EXPECT_EQ(graph->size(), 5);

    graph->set_thread_pool(pool_.get());
    graph->execute(true);

    auto join = graph->get_task("join");
    EXPECT_EQ(join->get(), 100);
}

// ==================== Complex Graph Tests ====================

TEST_F(TaskGraphTest, ComplexDAG) {
    fcc::TaskGraph<int> graph(pool_.get());

    // Create a complex DAG:
    //     A
    //    / \         (Forward slash backslash)
    //   B   C
    //  / \ / \       (Multiple paths)
    // D   E   F
    //  \ / \ /       (Convergence)
    //   G   H
    //    \ /         (Final join)
    //     I

    graph.add_task("A", []() { return 1; });
    graph.add_task("B", []() { return 2; });
    graph.add_task("C", []() { return 3; });
    graph.add_task("D", []() { return 4; });
    graph.add_task("E", []() { return 5; });
    graph.add_task("F", []() { return 6; });
    graph.add_task("G", []() { return 7; });
    graph.add_task("H", []() { return 8; });
    graph.add_task("I", []() { return 9; });

    graph.add_dependency("B", "A");
    graph.add_dependency("C", "A");
    graph.add_dependency("D", "B");
    graph.add_dependency("E", "B");
    graph.add_dependency("E", "C");
    graph.add_dependency("F", "C");
    graph.add_dependency("G", "D");
    graph.add_dependency("G", "E");
    graph.add_dependency("H", "E");
    graph.add_dependency("H", "F");
    graph.add_dependency("I", "G");
    graph.add_dependency("I", "H");

    EXPECT_TRUE(graph.validate());

    graph.execute(true);

    // All tasks should complete
    for (const auto& id : {"A", "B", "C", "D", "E", "F", "G", "H", "I"}) {
        auto node = graph.get_task(id);
        EXPECT_EQ(node->status(), fcc::TaskNode<int>::Status::Completed);
    }
}

TEST_F(TaskGraphTest, StressTestManyIndependentTasks) {
    fcc::TaskGraph<int> graph(pool_.get());

    const int num_tasks = 100;
    std::atomic<int> completed_count{0};

    for (int i = 0; i < num_tasks; ++i) {
        std::string id = "task_" + std::to_string(i);
        graph.add_task(id, [i, &completed_count]() {
            completed_count.fetch_add(1);
            return i * i;
        });
    }

    graph.execute(true);

    EXPECT_EQ(completed_count.load(), num_tasks);

    // Verify all results
    for (int i = 0; i < num_tasks; ++i) {
        std::string id = "task_" + std::to_string(i);
        auto node = graph.get_task(id);
        EXPECT_EQ(node->get(), i * i);
    }
}

TEST_F(TaskGraphTest, StressTestLongChain) {
    fcc::TaskGraph<int> graph;

    const int chain_length = 50;

    for (int i = 0; i < chain_length; ++i) {
        std::string id = "task_" + std::to_string(i);
        graph.add_task(id, [i]() { return i; });

        if (i > 0) {
            std::string prev_id = "task_" + std::to_string(i - 1);
            graph.add_dependency(id, prev_id);
        }
    }

    graph.execute(false);

    // Last task should have value chain_length - 1
    std::string last_id = "task_" + std::to_string(chain_length - 1);
    auto last_node = graph.get_task(last_id);
    EXPECT_EQ(last_node->get(), chain_length - 1);
}

// ==================== Edge Case Tests ====================

TEST_F(TaskGraphTest, EmptyGraphExecution) {
    fcc::TaskGraph<void> graph;

    EXPECT_NO_THROW(graph.execute(false));
    EXPECT_NO_THROW(graph.execute(true));
}

TEST_F(TaskGraphTest, SingleTaskGraph) {
    fcc::TaskGraph<int> graph;

    graph.add_task("only", []() { return 42; });

    graph.execute(false);

    auto node = graph.get_task("only");
    EXPECT_EQ(node->get(), 42);
}

TEST_F(TaskGraphTest, RemoveTask) {
    fcc::TaskGraph<int> graph;

    graph.add_task("task1", []() { return 1; });
    graph.add_task("task2", []() { return 2; });

    EXPECT_EQ(graph.size(), 2);

    graph.remove_task("task1");

    EXPECT_EQ(graph.size(), 1);
    EXPECT_EQ(graph.get_task("task1"), nullptr);
    EXPECT_NE(graph.get_task("task2"), nullptr);
}

TEST_F(TaskGraphTest, ClearGraph) {
    fcc::TaskGraph<int> graph;

    graph.add_task("task1", []() { return 1; });
    graph.add_task("task2", []() { return 2; });
    graph.add_dependency("task2", "task1");

    EXPECT_EQ(graph.size(), 2);

    graph.clear();

    EXPECT_EQ(graph.size(), 0);
    EXPECT_TRUE(graph.empty());
}

TEST_F(TaskGraphTest, VoidTaskGraph) {
    fcc::TaskGraph<void> graph;

    int value = 0;

    graph.add_task("set_value", [&value]() {
        value = 42;
    });

    graph.execute(false);

    EXPECT_EQ(value, 42);

    auto node = graph.get_task("set_value");
    EXPECT_NO_THROW(node->get()); // void get() should not throw
}

TEST_F(TaskGraphTest, ExecuteAlreadyExecutedGraph) {
    fcc::TaskGraph<int> graph;

    int counter = 0;
    graph.add_task("task", [&counter]() {
        return ++counter;
    });

    graph.execute(false);
    EXPECT_EQ(counter, 1);

    // Second execution should not re-run completed tasks
    graph.execute(false);
    EXPECT_EQ(counter, 1);
}

TEST_F(TaskGraphTest, ConcurrentGraphExecution) {
    // Test that we can't execute the same graph concurrently
    fcc::TaskGraph<void> graph(pool_.get());

    graph.add_task("slow_task", []() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    });

    auto future1 = std::async(std::launch::async, [&graph]() {
        graph.execute(true);
    });

    // Give first execution time to start
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Second execution should throw
    EXPECT_THROW(graph.execute(true), std::runtime_error);

    future1.wait();
}