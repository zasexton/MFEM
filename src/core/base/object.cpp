#include "object.h"
#include <atomic>
#include <format>

namespace fem::core::base {

// Static member initialization
std::atomic<Object::id_type> Object::next_id_{1};

Object::Object(std::string_view class_name, const std::source_location& loc)
    : id_(next_id_.fetch_add(1, std::memory_order_relaxed))
    , class_name_(class_name)
    , creation_location_(loc)
    , ref_count_(1)
    , destroyed_(false) {}

Object::Object(const Object& other)
    : id_(next_id_.fetch_add(1, std::memory_order_relaxed))  // New ID for copy
    , class_name_(other.class_name_)
    , creation_location_(std::source_location::current())  // New location
    , ref_count_(1)  // New object starts with ref count 1
    , destroyed_(false) {}

Object::Object(Object&& other) noexcept
    : id_(other.id_)
    , class_name_(std::move(other.class_name_))
    , creation_location_(other.creation_location_)
    , ref_count_(other.ref_count_.load(std::memory_order_acquire))
    , destroyed_(other.destroyed_) {
    // Reset other to valid but moved-from state
    other.id_ = next_id_.fetch_add(1, std::memory_order_relaxed);
    other.ref_count_.store(1, std::memory_order_release);
    other.destroyed_ = false;
}

Object& Object::operator=(const Object& other) {
    if (this != &other) {
        // Don't copy ID or ref count - keep our own identity
        class_name_ = other.class_name_;
        // Don't copy creation_location_ - keep our own
        // Don't copy destroyed_ flag
    }
    return *this;
}

Object& Object::operator=(Object&& other) noexcept {
    if (this != &other) {
        // For move assignment, we could either:
        // 1. Keep our ID (more consistent with copy assignment)
        // 2. Take other's ID (more consistent with move semantics)
        // Let's keep our ID for consistency

        // Don't change id_ - keep our identity
        class_name_ = std::move(other.class_name_);
        // Don't change creation_location_ - keep our own
        // Don't change ref_count_ - keep our own
        // Don't change destroyed_ - keep our own state

        // Reset other to valid but moved-from state
        other.class_name_.clear();
    }
    return *this;
}

Object::~Object() {
    on_destroy();
    destroyed_ = true;
}

void Object::initialize() {
    on_create();
}

} // namespace fem::core::base
