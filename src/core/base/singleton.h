#pragma once

#ifndef BASE_SINGLETON_H
#define BASE_SINGLETON_H

#include <mutex>
#include <memory>
#include <atomic>
#include <type_traits>

namespace fem::core::base {

/**
 * @brief Thread-safe singleton template using CRTP (Curiously Recurring Template Pattern)
 *
 * Features:
 * - Thread-safe initialization using std::call_once
 * - Lazy initialization (created on first access)
 * - Non-copyable and non-movable
 * - Automatic cleanup on program termination
 * - Exception-safe construction
 *
 * Usage:
 *   class MyManager : public Singleton<MyManager> {
 *       friend class Singleton<MyManager>;  // Required for private constructor access
 *   private:
 *       MyManager() = default;  // Private constructor
 *   public:
 *       void do_something();
 *   };
 *
 *   // Usage:
 *   MyManager::instance().do_something();
 */
    template<typename T>
    class Singleton {
    public:
        /**
         * @brief Get the singleton instance
         * @return Reference to the singleton instance
         */
        static T& instance() {
            std::call_once(init_flag_, &Singleton::create_instance);
            return *instance_;
        }

        /**
         * @brief Check if the singleton has been created
         * @return true if instance exists, false otherwise
         */
        static bool is_created() noexcept {
            return instance_ != nullptr;
        }

        /**
         * @brief Destroy the singleton instance (for testing purposes)
         * Warning: This should only be used in unit tests or special circumstances
         */
        static void destroy() {
            std::lock_guard<std::mutex> lock(destruction_mutex_);
            instance_.reset();
            init_flag_ = {};  // Reset the once_flag for potential recreation
        }

    protected:
        /**
         * @brief Protected constructor - derived classes should have private constructors
         */
        Singleton() = default;

        /**
         * @brief Protected destructor - prevents deletion through base pointer
         */
        virtual ~Singleton() = default;

        // Delete copy and move operations
        Singleton(const Singleton&) = delete;
        Singleton& operator=(const Singleton&) = delete;
        Singleton(Singleton&&) = delete;
        Singleton& operator=(Singleton&&) = delete;

    private:
        /**
         * @brief Create the singleton instance
         */
        static void create_instance() {
            instance_ = std::make_unique<T>();
        }

        static std::unique_ptr<T> instance_;
        static std::once_flag init_flag_;
        static std::mutex destruction_mutex_;  // For safe destruction in tests
    };

// Static member definitions
    template<typename T>
    std::unique_ptr<T> Singleton<T>::instance_ = nullptr;

    template<typename T>
    std::once_flag Singleton<T>::init_flag_;

    template<typename T>
    std::mutex Singleton<T>::destruction_mutex_;

/**
 * @brief Eager singleton - creates instance immediately (not lazy)
 *
 * Use when you need the singleton to be created at program startup
 * rather than on first access.
 */
    template<typename T>
    class EagerSingleton {
    public:
        static T& instance() {
            return instance_;
        }

        static bool is_created() noexcept {
            return true;  // Always created
        }

    protected:
        EagerSingleton() = default;
        virtual ~EagerSingleton() = default;

        // Delete copy and move operations
        EagerSingleton(const EagerSingleton&) = delete;
        EagerSingleton& operator=(const EagerSingleton&) = delete;
        EagerSingleton(EagerSingleton&&) = delete;
        EagerSingleton& operator=(EagerSingleton&&) = delete;

    private:
        static T instance_;
    };

    template<typename T>
    T EagerSingleton<T>::instance_;

/**
 * @brief Singleton with custom deleter support
 *
 * Useful when the singleton needs special cleanup logic
 */
    template<typename T, typename Deleter = std::default_delete<T>>
    class CustomSingleton {
    public:
        static T& instance() {
            std::call_once(init_flag_, &CustomSingleton::create_instance);
            return *instance_;
        }

        static bool is_created() noexcept {
            return instance_ != nullptr;
        }

        static void destroy() {
            std::lock_guard<std::mutex> lock(destruction_mutex_);
            instance_.reset();
            init_flag_ = {};
        }

    protected:
        CustomSingleton() = default;
        virtual ~CustomSingleton() = default;

        CustomSingleton(const CustomSingleton&) = delete;
        CustomSingleton& operator=(const CustomSingleton&) = delete;
        CustomSingleton(CustomSingleton&&) = delete;
        CustomSingleton& operator=(CustomSingleton&&) = delete;

    private:
        static void create_instance() {
            instance_ = std::unique_ptr<T, Deleter>(new T(), Deleter{});
        }

        static std::unique_ptr<T, Deleter> instance_;
        static std::once_flag init_flag_;
        static std::mutex destruction_mutex_;
    };

    template<typename T, typename Deleter>
    std::unique_ptr<T, Deleter> CustomSingleton<T, Deleter>::instance_ = nullptr;

    template<typename T, typename Deleter>
    std::once_flag CustomSingleton<T, Deleter>::init_flag_;

    template<typename T, typename Deleter>
    std::mutex CustomSingleton<T, Deleter>::destruction_mutex_;

// === Convenience Macros ===

/**
 * @brief Macro to easily declare a singleton class
 */
#define FEM_DECLARE_SINGLETON(ClassName) \
    class ClassName : public fem::core::Singleton<ClassName> { \
        friend class fem::core::Singleton<ClassName>; \
    private: \
        ClassName() = default; \
    public:

/**
 * @brief Macro to end singleton class declaration
 */
#define FEM_END_SINGLETON() \
    };

/**
 * @brief Macro for singleton method forwarding
 */
#define FEM_SINGLETON_METHOD(ClassName, method) \
    static auto method() -> decltype(ClassName::instance().method()) { \
        return ClassName::instance().method(); \
    }

} // namespace fem::core

#endif //BASE_SINGLETON_H
