# Core Memory Management - AGENT.md

## Purpose
The `memory/` layer provides advanced memory management facilities including custom allocators, memory pools, specialized containers, and memory tracking. It enables efficient memory usage patterns, reduces fragmentation, and provides tools for memory debugging and optimization.

## Architecture Philosophy
- **Allocation strategies**: Different allocators for different use cases
- **Cache efficiency**: Data structure layout for cache optimization
- **Zero overhead**: Optional features don't cost when not used
- **NUMA awareness**: Support for non-uniform memory architectures
- **Debug support**: Memory tracking and leak detection in debug builds

## Files Overview

### Allocators
```cpp
allocator_base.hpp      // Base allocator interface
malloc_allocator.hpp    // Standard malloc wrapper
aligned_allocator.hpp   // SIMD/cache-aligned allocation
pool_allocator.hpp      // Fixed-size object pools
arena_allocator.hpp     // Linear arena allocation
stack_allocator.hpp     // Stack-based allocation
freelist_allocator.hpp  // Free list management
buddy_allocator.hpp     // Buddy system allocation
slab_allocator.hpp      // Slab allocation for kernel-style
```

### Memory Pools
```cpp
memory_pool.hpp         // Generic memory pool
object_pool.hpp         // Type-safe object pool
thread_pool_allocator.hpp // Thread-local pools
concurrent_pool.hpp     // Lock-free memory pool
growing_pool.hpp        // Dynamically growing pool
```

### Specialized Containers
```cpp
small_vector.hpp        // Small buffer optimization vector
stable_vector.hpp       // Stable pointer vector
circular_buffer.hpp     // Fixed-size circular buffer
flat_map.hpp           // Cache-friendly map
flat_set.hpp           // Cache-friendly set
intrusive_list.hpp     // Zero-allocation list
bump_vector.hpp        // Append-only vector
```

### Memory Management
```cpp
memory_resource.hpp     // Polymorphic allocator interface
memory_tracker.hpp      // Memory usage tracking
memory_mapped.hpp       // Memory-mapped file support
shared_memory.hpp       // Inter-process shared memory
memory_barrier.hpp      // Memory ordering primitives
```

### Utilities
```cpp
aligned_storage.hpp     // Aligned storage utilities
memory_utils.hpp        // Memory manipulation functions
cache_line.hpp         // Cache line utilities
prefetch.hpp           // Prefetch hints
memory_stats.hpp       // Memory statistics collection
```

## Detailed Component Specifications

### `allocator_base.hpp`
```cpp
template<typename T>
class AllocatorBase {
public:
    using value_type = T;
    using pointer = T*;
    using size_type = std::size_t;
    
    // Required allocator interface
    [[nodiscard]] pointer allocate(size_type n);
    void deallocate(pointer p, size_type n);
    
    // Optional optimizations
    [[nodiscard]] pointer allocate_aligned(size_type n, size_type alignment);
    void deallocate_aligned(pointer p, size_type n, size_type alignment);
    
    // Bulk operations
    [[nodiscard]] pointer allocate_many(size_type n, size_type count);
    void deallocate_many(pointer p, size_type n, size_type count);
    
    // Memory info
    size_type max_size() const noexcept;
    bool can_allocate(size_type n) const noexcept;
    
    // Statistics
    size_type bytes_allocated() const noexcept;
    size_type bytes_available() const noexcept;
};

// Allocator traits
template<typename Alloc>
struct allocator_traits {
    static constexpr bool is_stateless = /* detect */;
    static constexpr bool is_thread_safe = /* detect */;
    static constexpr size_t alignment = /* detect */;
};
```
**Why necessary**: Unified interface for all allocators, enables allocator-aware containers.
**Usage**: Custom memory strategies, performance optimization.

### `pool_allocator.hpp`
```cpp
template<typename T, size_t ChunkSize = 1024>
class PoolAllocator : public AllocatorBase<T> {
    struct Chunk {
        alignas(T) std::byte storage[sizeof(T) * ChunkSize];
        std::bitset<ChunkSize> used;
        Chunk* next;
    };
    
    Chunk* head_;
    std::vector<std::unique_ptr<Chunk>> chunks_;
    
public:
    PoolAllocator();
    
    [[nodiscard]] T* allocate(size_t n = 1);
    void deallocate(T* p, size_t n = 1);
    
    // Pool management
    void reserve(size_t count);
    void shrink_to_fit();
    void clear();
    
    // Statistics
    size_t chunks_allocated() const;
    size_t objects_allocated() const;
    double fragmentation() const;
};

// Thread-safe version
template<typename T, size_t ChunkSize = 1024>
class ConcurrentPoolAllocator : public PoolAllocator<T, ChunkSize> {
    mutable std::mutex mutex_;
    // Lock-free free list implementation
};
```
**Why necessary**: Eliminates allocation overhead for fixed-size objects, reduces fragmentation.
**Usage**: Particle systems, node-based data structures, frequent allocations.

### `arena_allocator.hpp`
```cpp
class Arena {
    std::byte* begin_;
    std::byte* current_;
    std::byte* end_;
    std::vector<std::unique_ptr<std::byte[]>> blocks_;
    
public:
    explicit Arena(size_t initial_size = 4096);
    
    [[nodiscard]] void* allocate(size_t size, size_t alignment = alignof(std::max_align_t));
    
    // Cannot deallocate individual allocations
    void reset();  // Reset to beginning
    void clear();  // Free all memory
    
    // Scoped arena
    class Scope {
        Arena* arena_;
        std::byte* mark_;
    public:
        explicit Scope(Arena& arena);
        ~Scope();  // Resets arena to mark
    };
    
    Scope make_scope() { return Scope(*this); }
    
    // Statistics
    size_t bytes_used() const;
    size_t bytes_allocated() const;
};

// Type-safe arena allocator
template<typename T>
class ArenaAllocator : public AllocatorBase<T> {
    Arena* arena_;
public:
    explicit ArenaAllocator(Arena& arena) : arena_(&arena) {}
    
    [[nodiscard]] T* allocate(size_t n) {
        return static_cast<T*>(arena_->allocate(n * sizeof(T), alignof(T)));
    }
    
    void deallocate(T*, size_t) { /* no-op */ }
};
```
**Why necessary**: Fast allocation for temporary data, perfect for frame allocations.
**Usage**: Temporary computations, per-frame allocations, expression evaluation.

### `object_pool.hpp`
```cpp
template<typename T>
class ObjectPool {
    struct Node {
        union {
            alignas(T) std::byte storage[sizeof(T)];
            Node* next;
        };
    };
    
    std::vector<std::unique_ptr<Node[]>> blocks_;
    Node* free_list_;
    size_t block_size_;
    
public:
    explicit ObjectPool(size_t block_size = 32);
    
    template<typename... Args>
    T* acquire(Args&&... args) {
        Node* node = pop_free_list();
        if (!node) {
            node = allocate_new_block();
        }
        return new (node->storage) T(std::forward<Args>(args)...);
    }
    
    void release(T* obj) {
        obj->~T();
        push_free_list(reinterpret_cast<Node*>(obj));
    }
    
    // RAII wrapper
    class UniquePtr {
        T* ptr_;
        ObjectPool* pool_;
    public:
        UniquePtr(T* ptr, ObjectPool* pool);
        ~UniquePtr() { if (ptr_) pool_->release(ptr_); }
        // Smart pointer interface...
    };
    
    template<typename... Args>
    UniquePtr make_unique(Args&&... args) {
        return UniquePtr(acquire(std::forward<Args>(args)...), this);
    }
};
```
**Why necessary**: Reuse objects without heap allocation, reduce construction overhead.
**Usage**: Temporary objects, message passing, game entities.

### `small_vector.hpp`
```cpp
template<typename T, size_t N = 16>
class SmallVector {
    union Storage {
        alignas(T) std::byte stack[N * sizeof(T)];
        T* heap;
    };
    
    Storage storage_;
    size_t size_;
    size_t capacity_;
    bool on_heap_;
    
public:
    SmallVector();
    ~SmallVector();
    
    // Vector interface
    void push_back(const T& value);
    void push_back(T&& value);
    
    template<typename... Args>
    T& emplace_back(Args&&... args);
    
    void pop_back();
    void clear();
    void reserve(size_t new_capacity);
    void resize(size_t new_size);
    
    // Accessors
    T& operator[](size_t index);
    T& at(size_t index);
    T* data();
    
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    bool empty() const { return size_ == 0; }
    bool on_stack() const { return !on_heap_; }
    
    // Iterators
    using iterator = T*;
    iterator begin() { return data(); }
    iterator end() { return data() + size_; }
    
private:
    void move_to_heap();
};
```
**Why necessary**: Avoid heap allocation for small arrays, cache efficiency.
**Usage**: Temporary collections, small fixed-size arrays, return values.

### `stable_vector.hpp`
```cpp
template<typename T, size_t BlockSize = 16>
class StableVector {
    struct Block {
        alignas(T) std::byte storage[BlockSize * sizeof(T)];
        std::bitset<BlockSize> occupied;
    };
    
    std::vector<std::unique_ptr<Block>> blocks_;
    size_t size_;
    
public:
    class reference {
        Block* block_;
        size_t index_;
    public:
        operator T&() const;
        T& get() const;
    };
    
    class iterator {
        // Iterator that skips unoccupied slots
    };
    
    // Vector-like interface
    reference push_back(const T& value);
    reference emplace_back(auto&&... args);
    void erase(reference ref);
    
    // Stable references
    reference operator[](size_t index);
    bool is_valid(reference ref) const;
    
    // Iteration
    iterator begin();
    iterator end();
    
    // Never invalidates references/pointers
    void reserve(size_t n);
    void shrink_to_fit();
};
```
**Why necessary**: Pointers remain valid through insertions/deletions.
**Usage**: Graph nodes, observer lists, cached references.

### `memory_tracker.hpp`
```cpp
class MemoryTracker {
    struct AllocationInfo {
        size_t size;
        size_t alignment;
        std::string type_name;
        SourceLocation location;
        std::chrono::time_point<std::chrono::steady_clock> timestamp;
        std::optional<StackTrace> stack_trace;
    };
    
    std::unordered_map<void*, AllocationInfo> allocations_;
    std::atomic<size_t> total_allocated_;
    std::atomic<size_t> peak_allocated_;
    std::atomic<size_t> allocation_count_;
    
public:
    static MemoryTracker& instance();
    
    void record_allocation(void* ptr, size_t size, 
                          const char* type = nullptr,
                          SourceLocation loc = SourceLocation::current());
    
    void record_deallocation(void* ptr);
    
    // Leak detection
    std::vector<AllocationInfo> find_leaks() const;
    void check_leaks_and_report();
    
    // Statistics
    size_t current_usage() const;
    size_t peak_usage() const;
    size_t allocation_count() const;
    
    // Reports
    void dump_allocations(std::ostream& os) const;
    void generate_report() const;
};

// RAII tracker
class ScopedMemoryTracker {
    size_t start_usage_;
public:
    ScopedMemoryTracker();
    ~ScopedMemoryTracker();
    size_t memory_used() const;
};

// Macros for automatic tracking
#ifdef DEBUG_MEMORY
    #define TRACK_ALLOCATION(ptr, size) \
        MemoryTracker::instance().record_allocation(ptr, size, #ptr)
#else
    #define TRACK_ALLOCATION(ptr, size) ((void)0)
#endif
```
**Why necessary**: Memory leak detection, profiling, debugging.
**Usage**: Debug builds, memory profiling, resource tracking.

### `memory_mapped.hpp`
```cpp
class MemoryMappedFile {
    void* base_;
    size_t size_;
    int fd_;
    bool writable_;
    
public:
    enum class Mode {
        ReadOnly,
        ReadWrite,
        CopyOnWrite
    };
    
    MemoryMappedFile(const std::filesystem::path& path, Mode mode);
    ~MemoryMappedFile();
    
    // Access
    void* data() { return base_; }
    const void* data() const { return base_; }
    size_t size() const { return size_; }
    
    // Typed access
    template<typename T>
    T* as() { return static_cast<T*>(base_); }
    
    template<typename T>
    std::span<T> as_span() {
        return std::span<T>(as<T>(), size_ / sizeof(T));
    }
    
    // Memory management
    void flush();
    void advise_sequential();
    void advise_random();
    void lock_in_memory();
    
    // Create view
    MemoryMappedView create_view(size_t offset, size_t length);
};
```
**Why necessary**: Efficient large file access, zero-copy I/O, shared memory.
**Usage**: Large datasets, databases, inter-process communication.

## Memory Management Patterns

### Arena Allocation Pattern
```cpp
Arena frame_arena(1024 * 1024);  // 1MB

void process_frame() {
    Arena::Scope scope(frame_arena);  // Auto-reset
    
    // All allocations in this frame
    auto* particles = frame_arena.allocate<Particle>(1000);
    auto* temp_buffer = frame_arena.allocate<float>(10000);
    
    // Process...
    
    // Automatic cleanup when scope ends
}
```

### Object Pool Pattern
```cpp
ObjectPool<Message> message_pool(100);

void send_message(const Data& data) {
    auto msg = message_pool.make_unique(data);
    
    queue.push(std::move(msg));
    // Automatically returned to pool when consumed
}
```

### Small Vector Usage
```cpp
SmallVector<int, 8> get_neighbors(Node* node) {
    SmallVector<int, 8> result;  // Stack allocation for <= 8 items
    
    for (auto& edge : node->edges) {
        result.push_back(edge.target);
    }
    
    return result;  // Efficient move
}
```

## Performance Considerations

- **Pool allocation**: O(1) allocation/deallocation
- **Arena allocation**: O(1) allocation, batch deallocation
- **Small vector**: Zero heap allocation for small sizes
- **Memory mapping**: Zero-copy file access
- **Alignment**: Proper alignment for SIMD operations

## Testing Strategy

- **Allocation patterns**: Test various allocation sizes
- **Thread safety**: Concurrent allocation stress tests
- **Memory leaks**: Valgrind/AddressSanitizer integration
- **Performance**: Benchmark against standard allocators
- **Edge cases**: Out of memory, fragmentation

## Usage Guidelines

1. **Choose the right allocator**: Pool for fixed-size, arena for temporary
2. **Prefer stack allocation**: Use SmallVector for small collections
3. **Track in debug**: Enable memory tracking in debug builds
4. **Align for SIMD**: Use aligned allocators for vectorized code
5. **Batch deallocations**: Use arenas for frame-based allocation

## Anti-patterns to Avoid

- Using global allocator for everything
- Not considering alignment requirements
- Forgetting to return objects to pools
- Memory leaks from circular references
- Excessive small allocations

## Dependencies
- `base/` - For Object patterns
- `error/` - For error handling
- Standard library (C++20)

## Future Enhancements
- NUMA-aware allocators
- GPU memory management
- Persistent memory support
- Memory compression
- Automatic defragmentation