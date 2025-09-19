#pragma once

#ifndef CORE_MEMORY_MEMORY_TRACKER_H
#define CORE_MEMORY_MEMORY_TRACKER_H

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <string>
#include <string_view>
#include <chrono>
#include <optional>
#include <source_location>

#include <config/config.h>
#include <config/debug.h>

#include "memory_stats.h"

namespace fem::core::memory {

#if (CORE_ENABLE_PROFILING || CORE_ENABLE_ASSERTS)
    #ifndef CORE_MEMORY_ENABLE_TRACKING
        #define CORE_MEMORY_ENABLE_TRACKING 1
    #endif
#else
    #ifndef CORE_MEMORY_ENABLE_TRACKING
        #define CORE_MEMORY_ENABLE_TRACKING 0
    #endif
#endif

#if CORE_MEMORY_ENABLE_TRACKING

class MemoryTracker {
public:
    struct AllocationInfo {
        std::size_t size{};
        std::size_t alignment{};
        std::string type_name{};
        std::string label{};
        std::source_location location = std::source_location::current();
        std::chrono::steady_clock::time_point timestamp = std::chrono::steady_clock::now();
    };

    using clock = std::chrono::steady_clock;

    static MemoryTracker& instance() {
        static MemoryTracker inst;
        return inst;
    }

    void on_alloc(void* p,
                  std::size_t bytes,
                  std::size_t alignment,
                  std::string_view type_name,
                  std::string_view label = {},
                  const std::source_location& loc = std::source_location::current()) {
        if (!p || bytes == 0) return;
        std::lock_guard lk(mutex_);
        AllocationInfo info;
        info.size = bytes;
        info.alignment = alignment;
        info.type_name.assign(type_name.begin(), type_name.end());
        info.label.assign(label.begin(), label.end());
        info.location = loc;
        info.timestamp = clock::now();
        allocations_.emplace(p, std::move(info));
        stats_.total_allocated += bytes;
        ++stats_.allocation_count;
        stats_.live_bytes += bytes;
        if (stats_.live_bytes > stats_.peak_bytes) stats_.peak_bytes = stats_.live_bytes;
    }

    void on_free(void* p) {
        if (!p) return;
        std::lock_guard lk(mutex_);
        auto it = allocations_.find(p);
        if (it != allocations_.end()) {
            stats_.total_freed += it->second.size;
            ++stats_.free_count;
            stats_.live_bytes -= it->second.size;
            allocations_.erase(it);
        } else {
            // Unknown pointer; ignore or assert in debug
            FEM_DEBUG_ASSERT_MSG(false, "MemoryTracker: free of unknown pointer");
        }
    }

    [[nodiscard]] MemoryStats stats() const {
        std::lock_guard lk(mutex_);
        return stats_;
    }

    [[nodiscard]] std::size_t live_allocation_count() const {
        std::lock_guard lk(mutex_);
        return allocations_.size();
    }

private:
    MemoryTracker() = default;
    ~MemoryTracker() = default;
    MemoryTracker(const MemoryTracker&) = delete;
    MemoryTracker& operator=(const MemoryTracker&) = delete;

    mutable std::mutex mutex_;
    std::unordered_map<void*, AllocationInfo> allocations_;
    MemoryStats stats_{};
};

#else // CORE_MEMORY_ENABLE_TRACKING == 0

// No-op stub when tracking disabled
class MemoryTracker {
public:
    struct AllocationInfo { };
    static MemoryTracker& instance() {
        static MemoryTracker inst; return inst;
    }
    void on_alloc(void*, std::size_t, std::size_t, std::string_view, std::string_view = {}, const std::source_location& = std::source_location::current()) {}
    void on_free(void*) {}
    [[nodiscard]] MemoryStats stats() const { return {}; }
    [[nodiscard]] std::size_t live_allocation_count() const { return 0; }
};

#endif // CORE_MEMORY_ENABLE_TRACKING

} // namespace fem::core::memory

#endif // CORE_MEMORY_MEMORY_TRACKER_H

