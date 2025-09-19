#pragma once

#ifndef CORE_MEMORY_ALLOCATOR_TRAITS_H
#define CORE_MEMORY_ALLOCATOR_TRAITS_H

#include <memory>
#include <type_traits>
#include <concepts>
#include <cstddef>
#include <cstdint>

#include <config/config.h>

namespace fem::core::memory {

// Convenience aliases over std::allocator_traits
template<class Alloc>
using allocator_value_t = typename std::allocator_traits<Alloc>::value_type;

template<class Alloc>
using allocator_pointer_t = typename std::allocator_traits<Alloc>::pointer;

template<class Alloc>
using allocator_const_pointer_t = typename std::allocator_traits<Alloc>::const_pointer;

template<class Alloc>
using allocator_void_pointer_t = typename std::allocator_traits<Alloc>::void_pointer;

template<class Alloc>
using allocator_size_t = typename std::allocator_traits<Alloc>::size_type;

template<class Alloc>
using allocator_diff_t = typename std::allocator_traits<Alloc>::difference_type;

template<class Alloc, class T>
using rebind_alloc_t = typename std::allocator_traits<Alloc>::template rebind_alloc<T>;

template<class Alloc, class T>
using rebind_traits_t = std::allocator_traits<rebind_alloc_t<Alloc, T>>;

// Basic allocator concept (uses std::allocator_traits)
template<class A>
concept Allocator = requires(A a, allocator_size_t<A> n) {
    typename allocator_value_t<A>;
    { std::allocator_traits<A>::allocate(a, n) };
};

template<class A, class T>
concept AllocatorFor = requires(A a, typename rebind_traits_t<A, T>::pointer p, allocator_size_t<A> n) {
    typename rebind_alloc_t<A, T>;
    { rebind_traits_t<A, T>::allocate(std::declval<rebind_alloc_t<A, T>&>(), n) };
    { rebind_traits_t<A, T>::deallocate(std::declval<rebind_alloc_t<A, T>&>(), p, n) };
};

// Allocate/deallocate N objects of value_type
template<Allocator A>
[[nodiscard]] inline allocator_pointer_t<A>
allocate_n(A& a, allocator_size_t<A> n) {
    return std::allocator_traits<A>::allocate(a, n);
}

template<Allocator A>
inline void deallocate_n(A& a, allocator_pointer_t<A> p, allocator_size_t<A> n) {
    std::allocator_traits<A>::deallocate(a, p, n);
}

// Allocate/deallocate single object of T (rebound allocator)
template<class A, class T>
    requires AllocatorFor<A, T>
[[nodiscard]] inline typename rebind_traits_t<A, T>::pointer
allocate_one(A& a) {
    auto ar = rebind_alloc_t<A, T>(a);
    return rebind_traits_t<A, T>::allocate(ar, 1);
}

template<class A, class T>
    requires AllocatorFor<A, T>
inline void deallocate_one(A& a, typename rebind_traits_t<A, T>::pointer p) {
    auto ar = rebind_alloc_t<A, T>(a);
    rebind_traits_t<A, T>::deallocate(ar, p, 1);
}

// Allocate raw bytes using a rebind to std::byte
template<Allocator A>
[[nodiscard]] inline std::byte*
allocate_bytes(A& a, std::size_t n, std::size_t /*alignment*/ = alignof(std::max_align_t)) {
    using ByteAlloc = rebind_alloc_t<A, std::byte>;
    using BT = std::allocator_traits<ByteAlloc>;
    ByteAlloc ab(a);
    return BT::allocate(ab, static_cast<typename BT::size_type>(n));
}

template<Allocator A>
inline void deallocate_bytes(A& a, std::byte* p, std::size_t n, std::size_t /*alignment*/ = alignof(std::max_align_t)) {
    using ByteAlloc = rebind_alloc_t<A, std::byte>;
    using BT = std::allocator_traits<ByteAlloc>;
    ByteAlloc ab(a);
    BT::deallocate(ab, p, static_cast<typename BT::size_type>(n));
}

// Propagation & equality traits (convenience)
template<class A>
using propagate_on_container_copy_assignment_t = typename std::allocator_traits<A>::propagate_on_container_copy_assignment;

template<class A>
using propagate_on_container_move_assignment_t = typename std::allocator_traits<A>::propagate_on_container_move_assignment;

template<class A>
using propagate_on_container_swap_t = typename std::allocator_traits<A>::propagate_on_container_swap;

template<class A>
inline A select_on_container_copy_construction(const A& a) {
    return std::allocator_traits<A>::select_on_container_copy_construction(a);
}

template<class A>
inline constexpr bool is_always_equal_v = std::allocator_traits<A>::is_always_equal::value;

} // namespace fem::core::memory

#endif // CORE_MEMORY_ALLOCATOR_TRAITS_H

