#pragma once

#ifndef CORE_MEMORY_ALLOCATOR_BASE_H
#define CORE_MEMORY_ALLOCATOR_BASE_H

#include <cstddef>
#include <type_traits>
#include <concepts>

#include <config/config.h>

#include "allocator_traits.h"

namespace fem::core::memory {

// Marker for allocator-base lineage (optional, for trait detection)
struct allocator_base_tag {};

// CRTP convenience base for custom allocators (optional).
//
// This base does NOT aim to replace std::allocator_traits-based design. It exists
// to provide:
// - A unified place for typedefs and propagate traits
// - A light CRTP pattern that forwards allocate/deallocate to the derived class
// - A consistent surface for containers that choose to depend on a non-std
//   allocator interface (e.g., internal containers that want a smaller API)
//
// Derive as: class MyAlloc : public AllocatorBase<MyAlloc, T> { ... } and
// implement: pointer do_allocate(size_type n); void do_deallocate(pointer p, size_type n);
template<class Derived, class T>
class AllocatorBase : public allocator_base_tag {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template<class U> struct rebind { using other = AllocatorBase<Derived, U>; };

    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::false_type;

    [[nodiscard]] pointer allocate(size_type n) { return derived().do_allocate(n); }
    void deallocate(pointer p, size_type n) { derived().do_deallocate(p, n); }

protected:
    [[nodiscard]] Derived& derived() { return static_cast<Derived&>(*this); }
    [[nodiscard]] const Derived& derived() const { return static_cast<const Derived&>(*this); }
};

// Concepts for allocators (either std::allocator-compatible or simple allocate/deallocate)
template<class A>
concept StdAllocatorLike = requires(A a, typename std::allocator_traits<A>::size_type n) {
    typename std::allocator_traits<A>::value_type;
    { std::allocator_traits<A>::allocate(a, n) };
    { std::allocator_traits<A>::deallocate(a, std::declval<typename std::allocator_traits<A>::pointer>(), n) };
};

template<class A>
concept SimpleAllocatorLike = requires(A a, std::size_t n) {
    typename A::value_type;
    { a.allocate(n) };
    { a.deallocate(std::declval<typename A::value_type*>(), n) };
};

// Properties/traits discovery
template<class A>
struct allocator_properties {
    static constexpr bool is_std_allocator = StdAllocatorLike<A>;
    static constexpr bool is_simple_allocator = SimpleAllocatorLike<A>;
    static constexpr bool is_always_equal = []{
        if constexpr (StdAllocatorLike<A>) return std::allocator_traits<A>::is_always_equal::value;
        else return false;
    }();
};

// Adapter: wrap a SimpleAllocatorLike that is not std::allocator-compatible
// into a std::allocator-compatible wrapper.
template<class Raw, class T>
class StdAllocatorAdapter {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template<class U> struct rebind { using other = StdAllocatorAdapter<Raw, U>; };

    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::false_type;

    explicit StdAllocatorAdapter(Raw* raw = nullptr) noexcept : raw_(raw) {}

    template<class U>
    explicit StdAllocatorAdapter(const StdAllocatorAdapter<Raw, U>& other) noexcept : raw_(other.raw()) {}

    [[nodiscard]] pointer allocate(size_type n) { return raw_->allocate(n); }
    void deallocate(pointer p, size_type n) { raw_->deallocate(p, n); }

    [[nodiscard]] Raw* raw() const noexcept { return raw_; }

    template<class U>
    bool operator==(const StdAllocatorAdapter<Raw, U>& o) const noexcept { return raw_ == o.raw_; }
    template<class U>
    bool operator!=(const StdAllocatorAdapter<Raw, U>& o) const noexcept { return !(*this == o); }

private:
    template<class, class> friend class StdAllocatorAdapter;
    Raw* raw_{nullptr};
};

} // namespace fem::core::memory

#endif // CORE_MEMORY_ALLOCATOR_BASE_H

