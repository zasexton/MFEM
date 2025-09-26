# Core Memory Management Library

## Overview

The Core Memory Management library provides high-performance, flexible memory allocation and management tools for C++ applications. It offers specialized allocators, memory pools, cache-friendly containers, and memory tracking utilities designed to optimize memory usage patterns and reduce fragmentation.

## Quick Start

```cpp
#include <core/memory/object_pool.h>
#include <core/memory/small_vector.h>
#include <core/memory/arena.h>

// Use an object pool for frequent allocations
fem::core::memory::ObjectPool<MyClass> pool(100);
auto obj = pool.acquire(constructor_args...);
// ... use obj
pool.release(obj);  // Return to pool for reuse

// Use small_vector to avoid heap allocations
fem::core::memory::SmallVector<int, 16> vec;  // Stack storage for up to 16 ints
vec.push_back(42);  // No heap allocation

// Use arena for temporary allocations
fem::core::memory::Arena arena(4096);
auto* temp_buffer = arena.allocate(1024);
// ... use buffer
arena.reset();  // Free all arena allocations at once
```

## Component Categories

### 1. Memory Allocators

Specialized allocators for different allocation patterns:

| File | Purpose | When to Use |
|------|---------|------------|
| `malloc_allocator.h` | Wrapper around standard malloc/free | Default fallback, C interoperability |
| `aligned_allocator.h` | Guarantees memory alignment | SIMD operations, cache-line alignment |
| `stack_allocator.h` | Stack-based linear allocation | Known lifetime, LIFO deallocation pattern |
| `freelist_allocator.h` | Free list for fixed-size blocks | Frequent same-size allocations |
| `buddy_allocator.h` | Buddy system for power-of-2 sizes | Variable sizes with low fragmentation |
| `slab_allocator.h` | Slab allocation (kernel-style) | Multiple fixed-size object caches |

**Example: Using Aligned Allocator for SIMD**
```cpp
#include <core/memory/aligned_allocator.h>

// Ensure 32-byte alignment for AVX operations
using AlignedAlloc = fem::core::memory::AlignedAllocator<float, 32>;
std::vector<float, AlignedAlloc> simd_data(1024);

// Now simd_data.data() is guaranteed to be 32-byte aligned
__m256 vec = _mm256_load_ps(&simd_data[0]);  // Aligned load
```

### 2. Memory Pools

Pre-allocated memory pools for efficient object management:

| File | Purpose | When to Use |
|------|---------|------------|
| `memory_pool.h` | Generic memory pool | Base implementation for custom pools |
| `object_pool.h` | Type-safe object pool | Reusable objects, game entities |
| `concurrent_pool.h` | Thread-safe pool (mutex) | Multi-threaded object allocation |
| `growing_pool.h` | Dynamically expanding pool | Unknown max capacity |
| `thread_pool_allocator.h` | Thread-local pools | Eliminate thread contention |

**Example: Object Pool for Game Entities**
```cpp
#include <core/memory/object_pool.h>

class Particle {
public:
    Particle(float x, float y, float vx, float vy);
    void update(float dt);
    bool is_dead() const;
};

fem::core::memory::ObjectPool<Particle> particle_pool(1000);

// Spawn particles without heap allocation
void spawn_explosion(float x, float y) {
    for (int i = 0; i < 100; ++i) {
        auto* particle = particle_pool.acquire(
            x + random(), y + random(),
            random_velocity(), random_velocity()
        );
        active_particles.push_back(particle);
    }
}

// Return dead particles to pool
void update_particles(float dt) {
    for (auto it = active_particles.begin(); it != active_particles.end();) {
        (*it)->update(dt);
        if ((*it)->is_dead()) {
            particle_pool.release(*it);
            it = active_particles.erase(it);
        } else {
            ++it;
        }
    }
}
```

### 3. Arena Allocation

Linear allocation with bulk deallocation:

| File | Purpose | When to Use |
|------|---------|------------|
| `arena.h` | Memory arena base | Temporary allocations, frame data |
| `arena_allocator.h` | STL-compatible arena allocator | Use arena with STL containers |

**Example: Frame-Based Arena Allocation**
```cpp
#include <core/memory/arena.h>
#include <core/memory/arena_allocator.h>

fem::core::memory::Arena frame_arena(1024 * 1024);  // 1MB arena

void render_frame() {
    // Create scoped arena - automatically resets on destruction
    auto scope = frame_arena.make_scope();

    // All allocations are from the arena
    fem::core::memory::ArenaAllocator<RenderCommand> alloc(frame_arena);
    std::vector<RenderCommand, decltype(alloc)> commands(alloc);

    // Build render commands
    for (const auto& object : scene_objects) {
        commands.emplace_back(object.get_render_command());
    }

    // Process commands
    gpu.execute(commands);

    // Arena automatically resets here, freeing all memory at once
}
```

### 4. Cache-Friendly Containers

Optimized containers for better cache utilization:

| File | Purpose | When to Use |
|------|---------|------------|
| `small_vector.h` | Stack-allocated small arrays | Avoid heap for small collections |
| `stable_vector.h` | Stable pointers on growth | Need persistent pointers/references |
| `ring_buffer.h` | Fixed-size circular buffer | Producer/consumer, streaming data |
| `circular_buffer.h` | Alias for ring_buffer | Alternative naming preference |
| `flat_map.h` | Contiguous key-value storage | Small-medium maps, cache efficiency |
| `flat_set.h` | Contiguous sorted set | Small-medium sets, cache efficiency |
| `intrusive_list.h` | Zero-overhead linked list | Embedded list nodes, no allocation |
| `bump_vector.h` | Append-only vector | Write-once data, minimal overhead |

**Example: Small Vector for Return Values**
```cpp
#include <core/memory/small_vector.h>

// Avoid heap allocation for typical cases
fem::core::memory::SmallVector<int, 8> get_neighbors(int node_id) {
    fem::core::memory::SmallVector<int, 8> neighbors;

    // Most nodes have < 8 neighbors, so no heap allocation
    for (const auto& edge : graph[node_id]) {
        neighbors.push_back(edge.target);
    }

    return neighbors;  // Efficient move, usually no heap involved
}
```

**Example: Ring Buffer for Audio Processing**
```cpp
#include <core/memory/ring_buffer.h>

fem::core::memory::ring_buffer<float> audio_buffer(1024);

// Producer thread
void audio_callback(const float* input, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (!audio_buffer.push(input[i])) {
            // Buffer full, handle overflow
            break;
        }
    }
}

// Consumer thread
void process_audio() {
    float sample;
    while (audio_buffer.pop(sample)) {
        apply_effects(sample);
        output_device.write(sample);
    }
}
```

### 5. Memory Management Utilities

Tools for memory tracking, mapping, and optimization:

| File | Purpose | When to Use |
|------|---------|------------|
| `memory_resource.h` | Polymorphic allocator interface | Runtime allocator selection |
| `memory_tracker.h` | Track allocations/leaks | Debug builds, profiling |
| `scoped_memory_tracker.h` | RAII memory tracking | Measure function memory usage |
| `memory_mapped.h` | Memory-mapped file I/O | Large files, zero-copy I/O |
| `shared_memory.h` | Inter-process shared memory | IPC, shared data structures |
| `memory_stats.h` | Memory statistics | Performance monitoring |
| `memory_barrier.h` | Memory ordering primitives | Lock-free programming |
| `cache_line.h` | Cache line utilities | Avoid false sharing |
| `prefetch.h` | Prefetch hints | Optimize memory access patterns |

**Example: Memory-Mapped File for Large Dataset**
```cpp
#include <core/memory/memory_mapped.h>

// Map large dataset without loading into RAM
fem::core::memory::MemoryMappedFile dataset(
    "huge_dataset.bin",
    fem::core::memory::MemoryMappedFile::Mode::ReadOnly
);

// Access data as if it were in memory
auto* data = dataset.data_as<float>();
size_t count = dataset.size() / sizeof(float);

// Process data - OS handles paging
for (size_t i = 0; i < count; ++i) {
    process_value(data[i]);
}
```

**Example: Memory Tracking in Debug Builds**
```cpp
#include <core/memory/scoped_memory_tracker.h>

void expensive_operation() {
    fem::core::memory::ScopedMemoryTracker tracker("ExpensiveOp");

    // ... do work that allocates memory ...

    // Automatically logs memory usage on scope exit
}

#ifdef DEBUG
    // Check for memory leaks at program exit
    fem::core::memory::MemoryTracker::instance().check_leaks_and_report();
#endif
```

## Choosing the Right Component

### Decision Tree for Allocators

```
Need fixed-size objects?
  YES → Use ObjectPool or Freelist Allocator
  NO  ↓

Need thread safety?
  YES → Use ConcurrentPool or ThreadPoolAllocator
  NO  ↓

Temporary allocations with known lifetime?
  YES → Use Arena or StackAllocator
  NO  ↓

Need specific alignment?
  YES → Use AlignedAllocator
  NO  ↓

Variable sizes with low fragmentation?
  YES → Use BuddyAllocator
  NO  → Use MallocAllocator (default)
```

### Decision Tree for Containers

```
Need a dynamic array?
  YES → Size usually small? → SmallVector
      → Need stable pointers? → StableVector
      → Append-only? → BumpVector
      → Otherwise → std::vector
  NO  ↓

Need a queue/buffer?
  YES → Fixed size? → RingBuffer/CircularBuffer
      → Otherwise → std::deque
  NO  ↓

Need a map/set?
  YES → Small-medium size? → FlatMap/FlatSet
      → Otherwise → std::map/std::set
  NO  ↓

Need a linked list?
  YES → Can embed node? → IntrusiveList
      → Otherwise → std::list
```

## Best Practices

### 1. Profile First
Always profile before optimizing. The standard allocator is often sufficient.

### 2. Match Lifetime to Allocator
- **Arena**: Frame/scope lifetime
- **Pool**: Reusable objects
- **Stack**: LIFO pattern
- **Standard**: Long-lived or complex lifetime

### 3. Consider Thread Safety
- Use `ConcurrentPool` for shared pools
- Prefer `ThreadPoolAllocator` to eliminate contention
- Use thread-local storage when possible

### 4. Optimize for Cache
- Use `SmallVector` for small collections
- Prefer `FlatMap/FlatSet` for iteration-heavy code
- Align data for SIMD operations

### 5. Debug with Tracking
```cpp
#ifdef DEBUG
    #define TRACK_MEMORY
#endif

#include <core/memory/memory_tracker.h>
```

### 6. Zero-Copy I/O
Use memory-mapped files for large datasets:
```cpp
// Instead of loading entire file
std::vector<byte> data = read_entire_file("large.dat");

// Map it directly
MemoryMappedFile file("large.dat", Mode::ReadOnly);
auto* data = file.data();
```

## Integration Example

Here's a complete example showing how to integrate multiple memory components:

```cpp
#include <core/memory/arena.h>
#include <core/memory/object_pool.h>
#include <core/memory/small_vector.h>
#include <core/memory/flat_map.h>
#include <core/memory/scoped_memory_tracker.h>

class GameWorld {
private:
    // Per-frame temporary allocations
    fem::core::memory::Arena frame_arena{1024 * 1024};

    // Reusable game objects
    fem::core::memory::ObjectPool<Enemy> enemy_pool{100};
    fem::core::memory::ObjectPool<Projectile> projectile_pool{500};

    // Cache-friendly storage
    fem::core::memory::flat_map<EntityID, Transform> transforms;

public:
    void update_frame(float dt) {
        // Track memory usage in debug
        #ifdef DEBUG
        fem::core::memory::ScopedMemoryTracker tracker("GameWorld::update");
        #endif

        // Reset arena for this frame
        auto arena_scope = frame_arena.make_scope();

        // Temporary collision pairs - allocated from arena
        auto* collision_pairs = static_cast<CollisionPair*>(
            frame_arena.allocate(sizeof(CollisionPair) * max_collisions)
        );

        // Find collisions
        int num_collisions = detect_collisions(collision_pairs);

        // Process collisions - may spawn new entities
        for (int i = 0; i < num_collisions; ++i) {
            handle_collision(collision_pairs[i]);
        }

        // Update all entities
        update_enemies(dt);
        update_projectiles(dt);

        // Arena automatically frees collision_pairs here
    }

    void spawn_enemy(const Position& pos) {
        // Get enemy from pool (no heap allocation)
        auto* enemy = enemy_pool.acquire(pos, get_enemy_stats());
        enemies.push_back(enemy);

        // Small vector for nearby targets (stack allocated)
        fem::core::memory::SmallVector<EntityID, 8> nearby_targets;
        find_nearby_entities(pos, 100.0f, nearby_targets);

        enemy->set_initial_targets(nearby_targets);
    }

    void destroy_enemy(Enemy* enemy) {
        // Return to pool for reuse
        auto it = std::find(enemies.begin(), enemies.end(), enemy);
        if (it != enemies.end()) {
            enemies.erase(it);
            enemy_pool.release(enemy);
        }
    }
};
```

## Performance Characteristics

| Component | Allocation | Deallocation | Memory Overhead | Thread Safe |
|-----------|------------|--------------|-----------------|-------------|
| MallocAllocator | O(log n) | O(log n) | Low | No |
| PoolAllocator | O(1) | O(1) | Medium | No |
| ConcurrentPool | O(1)* | O(1)* | Medium | Yes |
| Arena | O(1) | O(1) batch | Very Low | No |
| StackAllocator | O(1) | O(1) LIFO | Very Low | No |
| BuddyAllocator | O(log n) | O(log n) | Low-Medium | No |
| SmallVector | O(1)† | O(1) | Low | No |
| FlatMap | O(n) insert | O(n) | Low | No |
| RingBuffer | O(1) | O(1) | Fixed | No |

\* With mutex contention
† Until exceeding stack capacity

## Dependencies

- C++20 or later
- Platform support for memory mapping (optional)
- Platform support for shared memory (optional)

## Thread Safety

Components marked as thread-safe:
- `ConcurrentPool`
- `ThreadPoolAllocator` (via thread-local storage)

All other components require external synchronization for concurrent access.

## Error Handling

Memory components use the `core/error` module for error handling:

```cpp
auto result = memory_mapped_file.open_result("file.dat", Mode::ReadOnly);
if (result.is_error()) {
    handle_error(result.error());
}
```

## Further Reading

- [AGENT.md](AGENT.md) - Detailed architecture and implementation notes
- [benchmarks/](../benchmarks/) - Performance benchmarks
- [tests/unit/memory/](../tests/unit/memory/) - Unit tests and usage examples