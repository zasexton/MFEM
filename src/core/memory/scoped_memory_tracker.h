#pragma once

#ifndef CORE_MEMORY_SCOPED_MEMORY_TRACKER_H
#define CORE_MEMORY_SCOPED_MEMORY_TRACKER_H

#include <string>
#include <string_view>
#include <utility>

#include <config/config.h>

#include "memory_tracker.h"

namespace fem::core::memory {

// ScopedMemoryTracker: records memory usage delta within a scope, optional label.
class ScopedMemoryTracker {
public:
    explicit ScopedMemoryTracker(std::string label = {})
        : label_(std::move(label))
        , start_(MemoryTracker::instance().stats()) {}

    ~ScopedMemoryTracker() {
        auto end = MemoryTracker::instance().stats();
        delta_.total_allocated = end.total_allocated - start_.total_allocated;
        delta_.total_freed     = end.total_freed     - start_.total_freed;
        // live/peak are instantaneous; keep end snapshot values in delta_.live_bytes and peak_bytes
        delta_.live_bytes = end.live_bytes;
        delta_.peak_bytes = end.peak_bytes;
        delta_.allocation_count = end.allocation_count - start_.allocation_count;
        delta_.free_count       = end.free_count       - start_.free_count;
        // Users can query with delta() post-destruction only if they stored a copy.
    }

    [[nodiscard]] const std::string& label() const noexcept { return label_; }
    [[nodiscard]] const MemoryStats& start() const noexcept { return start_; }
    [[nodiscard]] const MemoryStats& delta() const noexcept { return delta_; }

private:
    std::string label_;
    MemoryStats start_{};
    MemoryStats delta_{};
};

} // namespace fem::core::memory

#endif // CORE_MEMORY_SCOPED_MEMORY_TRACKER_H

