#include <memory>
#include "exception_base.h"

namespace fem::core::error {

// Stub implementations for StackTrace methods
void StackTrace::capture(int skip_frames) {
    // Stub implementation - does nothing
    (void)skip_frames; // Suppress unused parameter warning
}

std::string StackTrace::format() const {
    // Stub implementation - returns empty string
    return "";
}

} // namespace fem::core::error