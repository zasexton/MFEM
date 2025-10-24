#ifndef NUMERIC_DECOMPOSITIONS_WORKSPACE_H
#define NUMERIC_DECOMPOSITIONS_WORKSPACE_H

#include <vector>
#include <cstddef>
#include <algorithm>

namespace fem::numeric::decompositions {

// Workspace management structure for decomposition algorithms
// Provides pre-allocated buffers to avoid repeated allocations in hot loops
template <typename T>
struct DecompositionWorkspace {
    std::vector<T> buffer;          // Main workspace buffer
    std::vector<T> secondary_buffer; // Secondary buffer for algorithms needing two workspaces
    std::size_t capacity;
    std::size_t secondary_capacity;

    DecompositionWorkspace() : capacity(0), secondary_capacity(0) {}

    // Ensure the primary buffer has at least 'n' elements
    void ensure_size(std::size_t n) {
        if (n > capacity) {
            buffer.resize(n);
            capacity = n;
        }
    }

    // Ensure the secondary buffer has at least 'n' elements
    void ensure_secondary_size(std::size_t n) {
        if (n > secondary_capacity) {
            secondary_buffer.resize(n);
            secondary_capacity = n;
        }
    }

    // Get a pointer to the primary buffer with at least 'n' elements
    T* get_buffer(std::size_t n) {
        ensure_size(n);
        return buffer.data();
    }

    // Get a pointer to the secondary buffer with at least 'n' elements
    T* get_secondary_buffer(std::size_t n) {
        ensure_secondary_size(n);
        return secondary_buffer.data();
    }

    // Clear all buffers (doesn't deallocate, just resets size tracking)
    void clear() {
        buffer.clear();
        secondary_buffer.clear();
        capacity = 0;
        secondary_capacity = 0;
    }

    // Reserve capacity in advance if the required size is known
    void reserve(std::size_t primary_size, std::size_t secondary_size = 0) {
        if (primary_size > capacity) {
            buffer.reserve(primary_size);
            capacity = primary_size;
        }
        if (secondary_size > secondary_capacity) {
            secondary_buffer.reserve(secondary_size);
            secondary_capacity = secondary_size;
        }
    }
};

// Thread-local workspace for decomposition algorithms
// Each thread gets its own workspace to avoid synchronization overhead
template <typename T>
DecompositionWorkspace<T>& get_thread_local_workspace() {
    static thread_local DecompositionWorkspace<T> workspace;
    return workspace;
}

} // namespace fem::numeric::decompositions

#endif // NUMERIC_DECOMPOSITIONS_WORKSPACE_H