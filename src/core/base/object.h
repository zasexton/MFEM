#pragma once

#ifndef BASE_OBJECT_H
#define BASE_OBJECT_H

#include <atomic>
#include <memory>
#include <string>
#include <string_view>
#include <typeinfo>
#include <concepts>
#include <format>
#include <source_location>
#include <utility>
#include <functional>

#include <config/config.h>
#include <config/debug.h>

namespace fem::core::base {
    // Type trait to check if a type is derived from Object
    template<typename T>
    concept ObjectDerived = std::is_base_of_v<class Object, T>;

    // Forward declarations

    /**
     * @brief Root base class for all objects in the FEM solver
     *
     * Provides:
     * - Unique object identification
     * - Reference counting for memory management
     * - Type information and runtime type checking
     * - Object lifecycle tracking
     * - Debug and logging support
     */
    class Object {
    public:
        using id_type = fem::config::id_type;
        using ref_count_type = std::atomic<std::size_t>;

        /**
         * @brief Default constructor
         */
        explicit Object(std::string_view class_name = "",
                        const std::source_location& loc = std::source_location::current());

        /**
         * @brief Copy constructor - creates new object with new ID
         */
        Object(const Object& other);

        /**
         * @brief Move constructor
         */
        Object(Object&& other) noexcept;

        /**
         * @brief Copy assignment - does NOT copy ID or ref count
         */
        Object& operator=(const Object& other);

        /**
         * @brief Move assignment
         */
        Object& operator=(Object&& other) noexcept;

        /**
         * @brief Virtual destructor
         */
        virtual ~Object();

        // === Object Identity ===

        /**
         * @brief Get unique object ID
         */
        [[nodiscard]] constexpr id_type id() const noexcept { return id_; }

        /**
         * @brief Get object class name
         */
        [[nodiscard]] std::string_view class_name() const noexcept { return class_name_; }

        /**
         * @brief Get runtime type information
         */
        [[nodiscard]] const std::type_info& type_info() const noexcept { return typeid(*this); }

        /**
         * @brief Check if object is of specific type
         */
        template<ObjectDerived T>
        [[nodiscard]] bool is_type() const noexcept {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnonnull-compare"
            return dynamic_cast<const T*>(this) != nullptr;
#pragma GCC diagnostic pop
        }

        /**
         * @brief Safe cast to derived type
         */
        template<ObjectDerived T>
        [[nodiscard]] T* as() noexcept {
            return dynamic_cast<T*>(this);
        }

        template<ObjectDerived T>
        [[nodiscard]] const T* as() const noexcept {
            return dynamic_cast<const T*>(this);
        }

        /**
         * @brief Safe cast that throws on failure.
         *
         * Performs a `dynamic_cast` and throws `std::bad_cast` if the cast
         * fails, providing exception-based error handling.
         */
        template<ObjectDerived T>
        [[nodiscard]] T& as_ref() {
            auto* ptr = dynamic_cast<T*>(this);
#if CORE_ENABLE_ASSERTS
            FEM_NUMERIC_ASSERT_MSG(ptr != nullptr, "Invalid object cast");
            return *ptr;
#else
            if (!ptr) { throw std::bad_cast{}; }
            return *ptr;
#endif
        }

        /**
         * @brief Const-safe cast that throws on failure.
         */
        template<ObjectDerived T>
        [[nodiscard]] const T& as_ref() const {
            auto* ptr = dynamic_cast<const T*>(this);
#if CORE_ENABLE_ASSERTS
            FEM_NUMERIC_ASSERT_MSG(ptr != nullptr, "Invalid object cast");
            return *ptr;
#else
            if (!ptr) { throw std::bad_cast{}; }
            return *ptr;
#endif
        }

        // === Reference Counting ===

        /**
         * @brief Increment reference count
         */
        void add_ref() const noexcept {
            ref_count_.fetch_add(1, std::memory_order_relaxed);
        }

        /**
         * @brief Decrement reference count
         */
        void release() const noexcept {
            if (ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                delete this;
            }
        }

        /**
         * @brief Get current reference count
         */
        [[nodiscard]] std::size_t ref_count() const noexcept {
            return ref_count_.load(std::memory_order_acquire);
        }

        // === Object Lifecycle ===

        /**
         * @brief Check if object is valid/alive
         */
        [[nodiscard]] virtual bool is_valid() const noexcept { return !destroyed_; }

        /**
         * @brief Mark object as destroyed (for debugging)
         */
        virtual void destroy() noexcept { destroyed_ = true; }

        // === Debugging and Utilities ===

        /**
         * @brief Get string representation of object
         */
        [[nodiscard]] virtual std::string to_string() const {
            return std::format("{}(id={}, refs={})", class_name_, id_, ref_count());
        }

        /**
         * @brief Get debug information
         */
        [[nodiscard]] virtual std::string debug_info() const {
            return std::format("Object Debug Info:\n"
                               "  Class: {}\n"
                               "  Type: {}\n"
                               "  ID: {}\n"
                               "  Refs: {}\n"
                               "  Valid: {}\n"
                               "  Created at: {}:{}",
                               class_name_,
                               type_info().name(),
                               id_,
                               ref_count(),
                               is_valid(),
                               creation_location_.file_name(),
                               creation_location_.line());
        }

        /**
         * @brief Compare objects by ID
         */
        [[nodiscard]] constexpr bool operator==(const Object& other) const noexcept {
            return id_ == other.id_;
        }

        [[nodiscard]] constexpr auto operator<=>(const Object& other) const noexcept {
            return id_ <=> other.id_;
        }

    protected:
        /**
         * @brief Initialize the object after construction.
         *
         * Derived classes must call this at the end of their constructors
         * to trigger lifecycle callbacks.
         */
        void initialize();

        /**
         * @brief Called when object is first created (override for initialization)
         */
        virtual void on_create() {}

        /**
         * @brief Called when object is being destroyed (override for cleanup)
         */
        virtual void on_destroy() {}

    private:
        static std::atomic<id_type> next_id_;

        id_type id_;
        std::string class_name_;
        std::source_location creation_location_;
        mutable ref_count_type ref_count_{1}; // Start with 1 reference
        bool destroyed_{false};

        template<ObjectDerived T, typename KeyType> friend class Registry;
    };

    // === Smart Pointer Support ===

    /**
     * @brief Intrusive smart pointer for Object-derived classes
     */
    template<ObjectDerived T>
    class object_ptr {
    public:
        constexpr object_ptr() noexcept = default;
        constexpr object_ptr(std::nullptr_t) noexcept : ptr_(nullptr) {}

        explicit object_ptr(T* ptr) noexcept : ptr_(ptr) {}

        object_ptr(const object_ptr& other) noexcept : ptr_(other.ptr_) {
            if (ptr_) ptr_->add_ref();
        }

        object_ptr(object_ptr&& other) noexcept : ptr_(std::exchange(other.ptr_, nullptr)) {}

        template<ObjectDerived U> requires std::convertible_to<U*, T*>
        object_ptr(const object_ptr<U>& other) noexcept : ptr_(other.get()) {
            if (ptr_) ptr_->add_ref();
        }

        ~object_ptr() {
            if (ptr_) ptr_->release();
        }

        object_ptr& operator=(const object_ptr& other) noexcept {
            if (this != &other) {
                if (ptr_) ptr_->release();
                ptr_ = other.ptr_;
                if (ptr_) ptr_->add_ref();
            }
            return *this;
        }

        object_ptr& operator=(object_ptr&& other) noexcept {
            if (this != &other) {
                if (ptr_) ptr_->release();
                ptr_ = std::exchange(other.ptr_, nullptr);
            }
            return *this;
        }

        [[nodiscard]] T* get() const noexcept { return ptr_; }
        [[nodiscard]] T* operator->() const noexcept {
            FEM_NUMERIC_ASSERT_MSG(ptr_ != nullptr, "Dereferencing null object_ptr");
            return ptr_;
        }
        [[nodiscard]] T& operator*() const noexcept {
            FEM_NUMERIC_ASSERT_MSG(ptr_ != nullptr, "Dereferencing null object_ptr");
            return *ptr_;
        }
        [[nodiscard]] explicit operator bool() const noexcept { return ptr_ != nullptr; }

        void reset(T* ptr = nullptr) noexcept {
            if (ptr_) ptr_->release();
            ptr_ = ptr;
        }

        [[nodiscard]] T* release() noexcept {
            return std::exchange(ptr_, nullptr);
        }

        // Equality operators for containers
        [[nodiscard]] bool operator==(const object_ptr& other) const noexcept {
            return ptr_ == other.ptr_;
        }

        [[nodiscard]] bool operator!=(const object_ptr& other) const noexcept {
            return ptr_ != other.ptr_;
        }

        template<ObjectDerived U>
        [[nodiscard]] bool operator==(const object_ptr<U>& other) const noexcept {
            return ptr_ == other.ptr_;
        }

        template<ObjectDerived U>
        [[nodiscard]] bool operator!=(const object_ptr<U>& other) const noexcept {
            return ptr_ != other.ptr_;
        }

    private:
        T* ptr_{nullptr};

        template<ObjectDerived U> friend class object_ptr;
    };

    // === Factory Function ===

    /**
     * @brief Create object with intrusive pointer
     */
    template<ObjectDerived T, typename... Args>
    [[nodiscard]] object_ptr<T> make_object(Args&&... args) {
        // object_ptr adopts ownership without increasing the reference count
        return object_ptr<T>{new T(std::forward<Args>(args)...)};
    }

} // namespace fem::core::base

// === Hash Support ===
namespace std {
template<>
struct hash<fem::core::base::Object> {
    std::size_t operator()(const fem::core::base::Object& obj) const noexcept {
        return hash<fem::core::base::Object::id_type>{}(obj.id());
    }
};
template<fem::core::base::ObjectDerived T>
struct hash<fem::core::base::object_ptr<T>> {
    std::size_t operator()(const fem::core::base::object_ptr<T>& ptr) const noexcept {
        return ptr ? hash<fem::core::base::Object>{}(*ptr) : 0;
    }
};
} // namespace std

#endif //BASE_OBJECT_H
