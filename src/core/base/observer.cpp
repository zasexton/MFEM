#include <format>

#include "observer.h"
namespace fem::core::base {

// === Event Implementation ===

    Event::Event(std::string_view event_type)
            : type_name_(event_type), timestamp_(std::chrono::steady_clock::now()) {}

    std::string Event::to_string() const {
        auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                timestamp_.time_since_epoch()).count();

        return std::format("Event(type={}, source={}, time={}ms, handled={})",
                           type_name_, source_, time_ms, handled_);
    }

} // namespace fem::core