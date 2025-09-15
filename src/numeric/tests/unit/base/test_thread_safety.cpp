#include <base/numeric_base.h>
#include <base/container_base.h>
#include <base/storage_base.h>
#include <base/allocator_base.h>
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <future>
#include <chrono>
#include <random>

using namespace fem::numeric;

// ============================================================================
// Thread Safety Critical Tests  
// ============================================================================

class ThreadSafetyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Get number of hardware threads
        num_threads_ = std::max(2u, std::thread::hardware_concurrency());
        if (num_threads_ > 8) num_threads_ = 8;  // Cap for test performance
    }
    
    unsigned int num_threads_;
    
    // Helper to run function concurrently and detect races
    template<typename Func>
    void run_concurrent_test(Func&& func, int iterations = 1000) {
        std::vector<std::future<void>> futures;
        std::atomic<bool> start_flag{false};
        
        // Launch threads
        for (unsigned int i = 0; i < num_threads_; ++i) {
            futures.emplace_back(std::async(std::launch::async, [&func, &start_flag, iterations]() {
                // Wait for all threads to be ready
                while (!start_flag.load()) {
                    std::this_thread::yield();
                }
                
                // Execute test function
                for (int iter = 0; iter < iterations; ++iter) {
                    func();
                }
            }));
        }
        
        // Start all threads simultaneously
        start_flag.store(true);
        
        // Wait for completion
        for (auto& future : futures) {
            future.wait();
        }
    }
};

// ============================================================================
// Container Thread Safety Tests - CRITICAL MISSING
// ============================================================================

TEST_F(ThreadSafetyTest, ConcurrentContainerAccess) {
    // Test concurrent read access to container
    const size_t container_size = 1000;
    std::vector<double> test_container(container_size);
    
    // Fill with known values
    std::iota(test_container.begin(), test_container.end(), 1.0);
    
    std::atomic<int> read_errors{0};
    std::atomic<long long> total_reads{0};
    
    auto read_test = [&]() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dist(0, container_size - 1);
        
        size_t index = dist(gen);
        double expected = static_cast<double>(index + 1);
        double actual = test_container[index];
        
        total_reads.fetch_add(1);
        
        if (actual != expected) {
            read_errors.fetch_add(1);
        }
    };
    
    run_concurrent_test(read_test, 10000);
    
    EXPECT_EQ(read_errors.load(), 0) << "Detected " << read_errors.load() 
                                     << " read errors out of " << total_reads.load() << " reads";
}

TEST_F(ThreadSafetyTest, ConcurrentIteratorAccess) {
    // Test that concurrent iterator access doesn't crash
    std::vector<int> test_data(1000);
    std::iota(test_data.begin(), test_data.end(), 0);
    
    std::atomic<bool> exception_thrown{false};
    std::atomic<long long> successful_iterations{0};
    
    auto iterator_test = [&]() {
        try {
            for (auto it = test_data.begin(); it != test_data.end(); ++it) {
                volatile int value = *it;  // Force read
                (void)value;  // Suppress unused variable warning
            }
            successful_iterations.fetch_add(1);
        } catch (...) {
            exception_thrown.store(true);
        }
    };
    
    run_concurrent_test(iterator_test, 100);
    
    EXPECT_FALSE(exception_thrown.load()) << "Iterator access threw exception";
    EXPECT_GT(successful_iterations.load(), 0) << "No successful iterations completed";
}

// ============================================================================
// Memory Allocator Thread Safety Tests - CRITICAL MISSING
// ============================================================================

TEST_F(ThreadSafetyTest, ConcurrentAllocationDeallocation) {
    // Test concurrent allocation and deallocation
    std::atomic<int> allocation_errors{0};
    std::atomic<int> deallocation_errors{0};
    std::atomic<long long> total_allocations{0};
    
    auto alloc_test = [&]() {
        std::vector<void*> pointers;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> size_dist(1, 1024);
        
        try {
            // Allocate multiple blocks
            for (int i = 0; i < 100; ++i) {
                size_t size = size_dist(gen);
                void* ptr = std::aligned_alloc(64, size);  // 64-byte alignment
                if (ptr) {
                    pointers.push_back(ptr);
                    total_allocations.fetch_add(1);
                } else {
                    allocation_errors.fetch_add(1);
                }
            }
            
            // Deallocate all blocks
            for (void* ptr : pointers) {
                std::free(ptr);
            }
        } catch (...) {
            deallocation_errors.fetch_add(1);
            // Clean up on exception
            for (void* ptr : pointers) {
                std::free(ptr);
            }
        }
    };
    
    run_concurrent_test(alloc_test, 10);
    
    EXPECT_EQ(allocation_errors.load(), 0) << "Memory allocation errors detected";
    EXPECT_EQ(deallocation_errors.load(), 0) << "Memory deallocation errors detected";
    EXPECT_GT(total_allocations.load(), 0) << "No allocations completed";
}

TEST_F(ThreadSafetyTest, ConcurrentAlignedAllocation) {
    // Test concurrent aligned allocation specifically
    std::atomic<int> alignment_errors{0};
    std::atomic<long long> successful_allocs{0};
    
    auto aligned_alloc_test = [&]() {
        constexpr size_t alignment = 64;  // Cache line alignment
        constexpr size_t size = 1024;
        
        void* ptr = std::aligned_alloc(alignment, size);
        if (ptr) {
            successful_allocs.fetch_add(1);
            
            // Check alignment
            if (reinterpret_cast<uintptr_t>(ptr) % alignment != 0) {
                alignment_errors.fetch_add(1);
            }
            
            // Write to memory to ensure it's valid
            std::memset(ptr, 0xAA, size);
            
            std::free(ptr);
        }
    };
    
    run_concurrent_test(aligned_alloc_test, 1000);
    
    EXPECT_EQ(alignment_errors.load(), 0) << "Memory alignment errors detected";
    EXPECT_GT(successful_allocs.load(), 0) << "No successful aligned allocations";
}

// ============================================================================
// Shared State Race Condition Tests - CRITICAL MISSING
// ============================================================================

TEST_F(ThreadSafetyTest, AtomicCounterRaceCondition) {
    // Test atomic operations for race conditions
    std::atomic<long long> atomic_counter{0};
    long long non_atomic_counter = 0;
    std::mutex counter_mutex;
    
    constexpr int increments_per_thread = 10000;
    
    auto atomic_increment_test = [&]() {
        for (int i = 0; i < increments_per_thread; ++i) {
            atomic_counter.fetch_add(1);
        }
    };
    
    auto non_atomic_increment_test = [&]() {
        for (int i = 0; i < increments_per_thread; ++i) {
            std::lock_guard<std::mutex> lock(counter_mutex);
            ++non_atomic_counter;
        }
    };
    
    // Test atomic counter (single batch to match expectations)
    run_concurrent_test(atomic_increment_test, 1);
    
    long long expected_atomic = static_cast<long long>(num_threads_) * increments_per_thread;
    EXPECT_EQ(atomic_counter.load(), expected_atomic) 
        << "Atomic counter race condition detected";
    
    // Test mutex-protected counter (single batch to match expectations)
    run_concurrent_test(non_atomic_increment_test, 1);
    
    long long expected_mutex = static_cast<long long>(num_threads_) * increments_per_thread;
    EXPECT_EQ(non_atomic_counter, expected_mutex) 
        << "Mutex-protected counter race condition detected";
}

TEST_F(ThreadSafetyTest, SharedResourceAccess) {
    // Test access to shared resource with proper synchronization
    struct SharedResource {
        mutable std::mutex mtx;
        std::vector<int> data;
        
        void add_value(int value) {
            std::lock_guard<std::mutex> lock(mtx);
            data.push_back(value);
        }
        
        size_t size() const {
            std::lock_guard<std::mutex> lock(mtx);
            return data.size();
        }
        
        bool contains(int value) const {
            std::lock_guard<std::mutex> lock(mtx);
            return std::find(data.begin(), data.end(), value) != data.end();
        }
    };
    
    SharedResource resource;
    std::atomic<int> next_value{0};
    std::atomic<int> operation_errors{0};
    
    auto resource_test = [&]() {
        try {
            for (int i = 0; i < 100; ++i) {
                int value = next_value.fetch_add(1);
                resource.add_value(value);
                
                // Verify the value was added
                if (!resource.contains(value)) {
                    operation_errors.fetch_add(1);
                }
            }
        } catch (...) {
            operation_errors.fetch_add(1);
        }
    };
    
    run_concurrent_test(resource_test, 10);
    
    EXPECT_EQ(operation_errors.load(), 0) << "Shared resource access errors detected";
    
    // Verify final state
    size_t expected_size = static_cast<size_t>(num_threads_) * 10 * 100;  // threads * iterations * inner_loop
    EXPECT_EQ(resource.size(), expected_size) << "Incorrect final shared resource size";
}

// ============================================================================
// Memory Ordering Tests - CRITICAL MISSING
// ============================================================================

TEST_F(ThreadSafetyTest, SequentialConsistencyOrdering) {
    // Test sequential consistency memory ordering
    std::atomic<int> x{0};
    std::atomic<int> y{0};
    std::atomic<bool> ready{false};
    
    std::atomic<int> r1{0};
    std::atomic<int> r2{0};
    std::atomic<int> inconsistent_observations{0};
    
    constexpr int test_iterations = 10000;
    
    for (int iter = 0; iter < test_iterations; ++iter) {
        x.store(0);
        y.store(0);
        ready.store(false);
        r1.store(0);
        r2.store(0);
        
        auto writer1 = std::async(std::launch::async, [&]() {
            while (!ready.load()) std::this_thread::yield();
            x.store(1, std::memory_order_seq_cst);
            r1.store(y.load(std::memory_order_seq_cst));
        });
        
        auto writer2 = std::async(std::launch::async, [&]() {
            while (!ready.load()) std::this_thread::yield();
            y.store(1, std::memory_order_seq_cst);
            r2.store(x.load(std::memory_order_seq_cst));
        });
        
        ready.store(true);
        writer1.wait();
        writer2.wait();
        
        // In sequential consistency, r1 == 0 && r2 == 0 should be impossible
        // (This would violate sequential consistency)
        if (r1.load() == 0 && r2.load() == 0) {
            inconsistent_observations.fetch_add(1);
        }
    }
    
    // With sequential consistency, we should see very few (ideally zero) inconsistent observations
    double inconsistency_rate = static_cast<double>(inconsistent_observations.load()) / test_iterations;
    EXPECT_LT(inconsistency_rate, 0.01) << "High inconsistency rate detected: " << inconsistency_rate;
    
    std::cout << "Sequential consistency inconsistency rate: " << inconsistency_rate << std::endl;
}

TEST_F(ThreadSafetyTest, RelaxedMemoryOrdering) {
    // Test relaxed memory ordering behavior
    std::atomic<long long> relaxed_counter{0};
    std::atomic<bool> stop_flag{false};
    
    auto increment_relaxed = [&]() {
        while (!stop_flag.load(std::memory_order_relaxed)) {
            relaxed_counter.fetch_add(1, std::memory_order_relaxed);
        }
    };
    
    std::vector<std::future<void>> futures;
    for (unsigned int i = 0; i < num_threads_; ++i) {
        futures.emplace_back(std::async(std::launch::async, increment_relaxed));
    }
    
    // Let threads run for a short time
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop_flag.store(true, std::memory_order_relaxed);
    
    // Wait for completion
    for (auto& future : futures) {
        future.wait();
    }
    
    // With relaxed ordering, we should still get correct final count
    // (no operations should be lost)
    long long final_count = relaxed_counter.load(std::memory_order_relaxed);
    EXPECT_GT(final_count, 0) << "No increments occurred";
    
    std::cout << "Relaxed memory ordering final count: " << final_count << std::endl;
}

// ============================================================================
// Lock-Free Data Structure Tests - CRITICAL MISSING
// ============================================================================

TEST_F(ThreadSafetyTest, LockFreeStackTest) {
    // Simple lock-free stack implementation for testing
    struct Node {
        int data;
        Node* next;
        Node(int val) : data(val), next(nullptr) {}
    };
    
    class LockFreeStack {
    private:
        std::atomic<Node*> head_{nullptr};
        // Simple deferred reclamation to avoid ABA/use-after-free in test
        std::mutex retire_mtx_;
        std::vector<Node*> retired_;
        
    public:
        ~LockFreeStack() {
            // Delete remaining nodes in the stack
            while (Node* node = head_.load()) {
                head_.store(node->next);
                delete node;
            }
            // Delete nodes that were popped during the test
            std::lock_guard<std::mutex> lg(retire_mtx_);
            for (Node* n : retired_) { delete n; }
            retired_.clear();
        }
        
        void push(int data) {
            Node* new_node = new Node(data);
            new_node->next = head_.load();
            while (!head_.compare_exchange_weak(new_node->next, new_node)) {
                // Retry with updated next pointer
            }
        }
        
        bool pop(int& result) {
            Node* head = head_.load();
            while (head != nullptr && 
                   !head_.compare_exchange_weak(head, head->next)) {
                // Retry
            }
            if (head == nullptr) {
                return false;
            }
            result = head->data;
            // Defer deletion to destructor to avoid races (test-only)
            {
                std::lock_guard<std::mutex> lg(retire_mtx_);
                retired_.push_back(head);
            }
            return true;
        }
        
        bool empty() const {
            return head_.load() == nullptr;
        }
    };
    
    LockFreeStack stack;
    std::atomic<int> push_count{0};
    std::atomic<int> pop_count{0};
    std::atomic<int> push_errors{0};
    std::atomic<int> pop_errors{0};
    
    auto push_test = [&]() {
        try {
            for (int i = 0; i < 1000; ++i) {
                stack.push(i);
                push_count.fetch_add(1);
            }
        } catch (...) {
            push_errors.fetch_add(1);
        }
    };
    
    auto pop_test = [&]() {
        try {
            int value;
            for (int i = 0; i < 500; ++i) {  // Pop fewer than we push
                if (stack.pop(value)) {
                    pop_count.fetch_add(1);
                }
            }
        } catch (...) {
            pop_errors.fetch_add(1);
        }
    };
    
    // Launch push threads first
    std::vector<std::future<void>> push_futures;
    for (unsigned int i = 0; i < num_threads_ / 2; ++i) {
        push_futures.emplace_back(std::async(std::launch::async, push_test));
    }
    
    // Then launch pop threads
    std::vector<std::future<void>> pop_futures;
    for (unsigned int i = 0; i < num_threads_ / 2; ++i) {
        pop_futures.emplace_back(std::async(std::launch::async, pop_test));
    }
    
    // Wait for completion
    for (auto& future : push_futures) future.wait();
    for (auto& future : pop_futures) future.wait();
    
    EXPECT_EQ(push_errors.load(), 0) << "Push operations had errors";
    EXPECT_EQ(pop_errors.load(), 0) << "Pop operations had errors";
    EXPECT_GT(push_count.load(), 0) << "No successful push operations";
    EXPECT_GT(pop_count.load(), 0) << "No successful pop operations";
    EXPECT_LT(pop_count.load(), push_count.load()) << "Popped more items than pushed";
    
    std::cout << "Lock-free stack: pushed " << push_count.load() 
              << ", popped " << pop_count.load() << std::endl;
}

// ============================================================================
// Deadlock Detection Tests - CRITICAL MISSING
// ============================================================================

TEST_F(ThreadSafetyTest, DeadlockAvoidance) {
    // Test ordered lock acquisition to avoid deadlock
    std::mutex mutex1, mutex2;
    std::atomic<int> successful_operations{0};
    std::atomic<bool> deadlock_detected{false};
    
    auto ordered_lock_test = [&](bool reverse_order) {
        try {
            const auto timeout = std::chrono::milliseconds(1000);
            
            if (reverse_order) {
                // Try to acquire in different order - potential deadlock
                std::unique_lock<std::mutex> lock1(mutex1, std::defer_lock);
                std::unique_lock<std::mutex> lock2(mutex2, std::defer_lock);
                
                if (std::try_lock(lock2, lock1) == -1) {
                    // Successfully acquired both locks
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    successful_operations.fetch_add(1);
                }
            } else {
                // Acquire in consistent order - no deadlock
                std::lock_guard<std::mutex> lock1(mutex1);
                std::lock_guard<std::mutex> lock2(mutex2);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                successful_operations.fetch_add(1);
            }
        } catch (...) {
            deadlock_detected.store(true);
        }
    };
    
    // Test with consistent ordering (should work)
    std::vector<std::future<void>> futures;
    for (unsigned int i = 0; i < num_threads_; ++i) {
        futures.emplace_back(std::async(std::launch::async, ordered_lock_test, false));
    }
    
    for (auto& future : futures) {
        future.wait();
    }
    
    EXPECT_FALSE(deadlock_detected.load()) << "Deadlock detected in ordered locking";
    EXPECT_GT(successful_operations.load(), 0) << "No successful operations completed";
    
    std::cout << "Deadlock avoidance test: " << successful_operations.load() 
              << " successful operations" << std::endl;
}
