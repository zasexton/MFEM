#pragma once

#ifndef CORE_MEMORY_MEMORY_RESOURCE_H
#define CORE_MEMORY_MEMORY_RESOURCE_H

#include <cstddef>
#include <new>
#include <type_traits>

#include <config/config.h>

#if __has_include(<memory_resource>)
  #include <memory_resource>
#endif

namespace fem::core::memory {

#if defined(__cpp_lib_memory_resource) || __has_include(<memory_resource>)

// PMR aliases (preferred path)
using memory_resource = std::pmr::memory_resource;

template<class T>
using polymorphic_allocator = std::pmr::polymorphic_allocator<T>;

using monotonic_buffer_resource   = std::pmr::monotonic_buffer_resource;
using unsynchronized_pool_resource = std::pmr::unsynchronized_pool_resource;
using synchronized_pool_resource   = std::pmr::synchronized_pool_resource;

inline memory_resource* default_resource() noexcept {
    return std::pmr::get_default_resource();
}

inline memory_resource* set_default_resource(memory_resource* r) noexcept {
    return std::pmr::set_default_resource(r);
}

inline memory_resource* new_delete_resource() noexcept {
    return std::pmr::new_delete_resource();
}

inline memory_resource* null_memory_resource() noexcept {
    return std::pmr::null_memory_resource();
}

#else // Minimal fallback when <memory_resource> is unavailable

class memory_resource {
public:
    virtual ~memory_resource() = default;

    void* allocate(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) {
        return do_allocate(bytes, alignment);
    }
    void deallocate(void* p, std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) {
        do_deallocate(p, bytes, alignment);
    }
    bool is_equal(const memory_resource& other) const noexcept {
        return do_is_equal(other);
    }

protected:
    virtual void* do_allocate(std::size_t bytes, std::size_t alignment) = 0;
    virtual void  do_deallocate(void* p, std::size_t bytes, std::size_t alignment) = 0;
    virtual bool  do_is_equal(const memory_resource& other) const noexcept = 0;
};

namespace detail {
    class new_delete_resource_impl final : public memory_resource {
    protected:
        void* do_allocate(std::size_t bytes, std::size_t alignment) override {
#if defined(__cpp_aligned_new)
            return ::operator new(bytes, std::align_val_t(alignment));
#else
            (void)alignment;
            return ::operator new(bytes);
#endif
        }
        void do_deallocate(void* p, std::size_t /*bytes*/, std::size_t alignment) override {
#if defined(__cpp_aligned_new)
            ::operator delete(p, std::align_val_t(alignment));
#else
            (void)alignment;
            ::operator delete(p);
#endif
        }
        bool do_is_equal(const memory_resource& other) const noexcept override {
            return this == &other;
        }
    };

    class null_memory_resource_impl final : public memory_resource {
    protected:
        void* do_allocate(std::size_t, std::size_t) override {
            throw std::bad_alloc{};
        }
        void do_deallocate(void*, std::size_t, std::size_t) override {}
        bool do_is_equal(const memory_resource& other) const noexcept override {
            return this == &other;
        }
    };
} // namespace detail

inline memory_resource* new_delete_resource() noexcept {
    static detail::new_delete_resource_impl r;
    return &r;
}

inline memory_resource* null_memory_resource() noexcept {
    static detail::null_memory_resource_impl r;
    return &r;
}

inline memory_resource*& default_resource_ref() noexcept {
    static memory_resource* current = new_delete_resource();
    return current;
}

inline memory_resource* default_resource() noexcept { return default_resource_ref(); }

inline memory_resource* set_default_resource(memory_resource* r) noexcept {
    memory_resource*& cur = default_resource_ref();
    memory_resource* prev = cur;
    if (r) cur = r;
    return prev;
}

// Simple polymorphic_allocator fallback
template<class T>
class polymorphic_allocator {
public:
    using value_type = T;

    explicit polymorphic_allocator(memory_resource* r = default_resource()) noexcept : res_(r) {}

    template<class U>
    explicit polymorphic_allocator(const polymorphic_allocator<U>& other) noexcept : res_(other.resource()) {}

    [[nodiscard]] T* allocate(std::size_t n) { return static_cast<T*>(res_->allocate(n * sizeof(T), alignof(T))); }
    void deallocate(T* p, std::size_t n) { res_->deallocate(p, n * sizeof(T), alignof(T)); }

    [[nodiscard]] memory_resource* resource() const noexcept { return res_; }

    template<class U>
    bool operator==(const polymorphic_allocator<U>& other) const noexcept { return res_->is_equal(*other.resource()); }
    template<class U>
    bool operator!=(const polymorphic_allocator<U>& other) const noexcept { return !(*this == other); }

private:
    memory_resource* res_;
};

// Minimal monotonic_buffer_resource fallback: delegates to new/delete_resource
class monotonic_buffer_resource {
public:
    explicit monotonic_buffer_resource(memory_resource* upstream = default_resource()) noexcept : upstream_(upstream) {}
    explicit monotonic_buffer_resource(std::size_t /*initial_size*/, memory_resource* upstream = default_resource()) noexcept : upstream_(upstream) {}

    void* allocate(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) { return upstream_->allocate(bytes, alignment); }
    void  deallocate(void* p, std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) { upstream_->deallocate(p, bytes, alignment); }

    void release() noexcept {}

private:
    memory_resource* upstream_;
};

// Pool resources are omitted in the fallback
using unsynchronized_pool_resource = monotonic_buffer_resource;
using synchronized_pool_resource   = monotonic_buffer_resource;

#endif // fallback

} // namespace fem::core::memory

#endif // CORE_MEMORY_MEMORY_RESOURCE_H
