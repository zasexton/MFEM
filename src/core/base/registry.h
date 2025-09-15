#pragma once

#ifndef MFEM_REGISTRY_H
#define MFEM_REGISTRY_H

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <string_view>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <functional>
#include <type_traits>
#include <algorithm>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <utility>

#include "object.h"
#include "singleton.h"

namespace fem::core::base {

/**
 * @brief Registry error codes
 */
enum class RegistryError : int {
    AlreadyRegistered = 1,
    KeyInUse = 2,
    NotFound = 3,
    InvalidInput = 4,
    CapacityExceeded = 5
};

/**
 * @brief Thread-safe registry for managing collections of objects
 *
 * Provides:
 * - Fast lookup by ID or name
 * - Automatic cleanup of destroyed objects
 * - Type-safe object storage and retrieval
 * - Thread-safe operations
 * - Iteration and filtering capabilities
 * - Event notifications for registration/unregistration
 *
 * Template Parameters:
 * - T: Type of objects to store (must derive from Object)
 * - KeyType: Type used for named lookups (default: std::string)
 */
    template<ObjectDerived T, typename KeyType = std::string>
    class Registry {
    public:
        using object_type = T;
        using key_type = KeyType;
        using id_type = Object::id_type;
        using object_ptr = fem::core::base::object_ptr<T>;
        using weak_ptr = std::weak_ptr<T>;

        // Event callback types
        using registration_callback = std::function<void(object_ptr)>;
        using unregistration_callback = std::function<void(id_type, std::string_view)>;

        /**
         * @brief Default constructor
         */
        Registry() {
            // Reserve reasonable capacity to avoid rehashing
            objects_by_id_.reserve(100);
            objects_by_key_.reserve(100);
        }

        /**
         * @brief Constructor with name
         */
        explicit Registry(std::string_view name) : name_(name) {
            // Reserve reasonable capacity to avoid rehashing
            objects_by_id_.reserve(100);
            objects_by_key_.reserve(100);
        }

        /**
         * @brief Destructor
         */
        ~Registry() = default;

        // Non-copyable but movable
        Registry(const Registry&) = delete;
        Registry& operator=(const Registry&) = delete;
        Registry(Registry&&) = default;
        Registry& operator=(Registry&&) = default;

        // === Registration Methods ===

        /**
         * @brief Register an object with automatic ID-based lookup
         * @return true if successfully registered, false if already registered or null
         */
        [[nodiscard]] bool register_object(object_ptr obj) noexcept {
            if (!obj) return false;

            auto id = obj->id();
            
            // Use shared lock for read check first
            {
                std::shared_lock lock(mutex_);
                if (objects_by_id_.contains(id)) {
                    return false; // Already registered
                }
            }
            
            // Now take exclusive lock for modification
            {
                std::lock_guard lock(mutex_);
                
                // Double-check after acquiring exclusive lock
                if (objects_by_id_.contains(id)) {
                    return false;
                }

                objects_by_id_[id] = obj;
            }
            
            // Notify without holding lock to prevent deadlocks
            notify_registration(obj);

            return true;
        }

        /**
         * @brief Register an object with a named key
         * @return true if successfully registered, false if already registered, null, or key in use
         */
        [[nodiscard]] bool register_object(const key_type& key, object_ptr obj) noexcept {
            if (!obj) return false;

            auto id = obj->id();
            
            // Use shared lock for read checks first
            {
                std::shared_lock lock(mutex_);
                if (objects_by_id_.contains(id) || objects_by_key_.contains(key)) {
                    return false; // Already registered or key in use
                }
            }
            
            // Now take exclusive lock for modification
            {
                std::lock_guard lock(mutex_);
                
                // Double-check after acquiring exclusive lock
                if (objects_by_id_.contains(id) || objects_by_key_.contains(key)) {
                    return false;
                }

                objects_by_id_[id] = obj;
                objects_by_key_[key] = obj;
                key_to_id_[key] = id;
                id_to_key_[id] = key;
            }

            // Notify without holding lock to prevent deadlocks
            notify_registration(obj);

            return true;
        }

        /**
         * @brief Register an object with automatic name generation
         */
        [[nodiscard]] bool register_object_auto_name(object_ptr obj, std::string_view prefix = "obj") noexcept {
            if (!obj) return false;

            std::lock_guard lock(mutex_);

            // Generate unique name
            key_type auto_name;
            int counter = 1;
            constexpr int max_attempts = 10000; // Prevent infinite loop
            
            do {
                if (counter > max_attempts) {
                    return false; // Failed to generate unique name
                }
                
                if constexpr (std::is_same_v<key_type, std::string>) {
                    auto_name = std::string(prefix) + "_" + std::to_string(counter++);
                } else {
                    auto_name = static_cast<key_type>(counter++); // For numeric keys
                }
            } while (objects_by_key_.contains(auto_name));

            return register_object_impl(auto_name, obj);
        }

        // === Lookup Methods ===

        /**
         * @brief Find object by ID
         */
        [[nodiscard]] object_ptr find_by_id(id_type id) const noexcept {
            std::shared_lock lock(mutex_);

            auto it = objects_by_id_.find(id);
            return it != objects_by_id_.end() ? it->second : nullptr;
        }

        /**
         * @brief Find object by key
         */
        [[nodiscard]] object_ptr find_by_key(const key_type& key) const noexcept {
            std::shared_lock lock(mutex_);

            auto it = objects_by_key_.find(key);
            return it != objects_by_key_.end() ? it->second : nullptr;
        }

        /**
         * @brief Find object by key with default value
         */
        [[nodiscard]] object_ptr find_or_default(const key_type& key, object_ptr default_obj = nullptr) const {
            auto obj = find_by_key(key);
            return obj ? obj : default_obj;
        }

        /**
         * @brief Check if object is registered by ID
         */
        [[nodiscard]] bool contains_id(id_type id) const noexcept {
            std::shared_lock lock(mutex_);
            return objects_by_id_.contains(id);
        }

        /**
         * @brief Check if key is registered
         */
        [[nodiscard]] bool contains_key(const key_type& key) const noexcept {
            std::shared_lock lock(mutex_);
            return objects_by_key_.contains(key);
        }

        /**
         * @brief Get key for object ID (if it has one)
         */
        [[nodiscard]] std::optional<key_type> get_key_for_id(id_type id) const {
            std::shared_lock lock(mutex_);

            auto it = id_to_key_.find(id);
            return it != id_to_key_.end() ? std::make_optional(it->second) : std::nullopt;
        }

        // === Removal Methods ===

        /**
         * @brief Unregister object by ID
         */
        [[nodiscard]] bool unregister_by_id(id_type id) noexcept {
            std::lock_guard lock(mutex_);

            auto it = objects_by_id_.find(id);
            if (it == objects_by_id_.end()) {
                return false;
            }

            // Get key if it exists
            std::string key_str;
            auto key_it = id_to_key_.find(id);
            if (key_it != id_to_key_.end()) {
                if constexpr (std::is_same_v<key_type, std::string>) {
                    key_str = key_it->second;
                } else {
                    key_str = std::to_string(key_it->second);
                }
                objects_by_key_.erase(key_it->second);
                key_to_id_.erase(key_it->second);
                id_to_key_.erase(key_it);
            }

            objects_by_id_.erase(it);
            notify_unregistration(id, key_str);

            return true;
        }

        /**
         * @brief Unregister object by key
         */
        [[nodiscard]] bool unregister_by_key(const key_type& key) noexcept {
            std::lock_guard lock(mutex_);

            auto it = objects_by_key_.find(key);
            if (it == objects_by_key_.end()) {
                return false;
            }

            auto id = key_to_id_[key];
            objects_by_id_.erase(id);
            objects_by_key_.erase(it);
            key_to_id_.erase(key);
            id_to_key_.erase(id);

            std::string key_str;
            if constexpr (std::is_same_v<key_type, std::string>) {
                key_str = key;
            } else {
                key_str = std::to_string(key);
            }
            notify_unregistration(id, key_str);

            return true;
        }

        /**
         * @brief Unregister an object (by the object itself)
         */
        [[nodiscard]] bool unregister_object(const object_ptr& obj) noexcept {
            if (!obj) return false;
            return unregister_by_id(obj->id());
        }

        /**
         * @brief Clear all registrations
         */
        void clear() {
            std::lock_guard lock(mutex_);

            // Notify about all unregistrations
            for (const auto& [id, obj] : objects_by_id_) {
                std::string key_str;
                auto key_it = id_to_key_.find(id);
                if (key_it != id_to_key_.end()) {
                    if constexpr (std::is_same_v<key_type, std::string>) {
                        key_str = key_it->second;
                    } else {
                        key_str = std::to_string(key_it->second);
                    }
                }
                notify_unregistration(id, key_str);
            }

            objects_by_id_.clear();
            objects_by_key_.clear();
            key_to_id_.clear();
            id_to_key_.clear();
        }

        // === Query Methods ===

        /**
         * @brief Get number of registered objects
         */
        [[nodiscard]] std::size_t size() const noexcept {
            std::shared_lock lock(mutex_);
            return objects_by_id_.size();
        }

        /**
         * @brief Check if registry is empty
         */
        [[nodiscard]] bool empty() const noexcept {
            std::shared_lock lock(mutex_);
            return objects_by_id_.empty();
        }

        /**
         * @brief Get all registered object IDs
         */
        [[nodiscard]] std::vector<id_type> get_all_ids() const {
            std::shared_lock lock(mutex_);

            std::vector<id_type> ids;
            ids.reserve(objects_by_id_.size());

            for (const auto& [id, obj] : objects_by_id_) {
                ids.push_back(id);
            }

            return ids;
        }

        /**
         * @brief Get all registered keys
         */
        [[nodiscard]] std::vector<key_type> get_all_keys() const {
            std::shared_lock lock(mutex_);

            std::vector<key_type> keys;
            keys.reserve(objects_by_key_.size());

            for (const auto& [key, obj] : objects_by_key_) {
                keys.push_back(key);
            }

            return keys;
        }

        /**
         * @brief Get all registered objects
         */
        [[nodiscard]] std::vector<object_ptr> get_all_objects() const {
            std::shared_lock lock(mutex_);

            std::vector<object_ptr> objects;
            objects.reserve(objects_by_id_.size());

            for (const auto& [id, obj] : objects_by_id_) {
                objects.push_back(obj);
            }

            return objects;
        }

        /**
         * @brief Find objects matching a predicate
         * @note Consider using find_first_if for early termination
         */
        template<typename Predicate>
        [[nodiscard]] std::vector<object_ptr> find_if(Predicate pred) const {
            std::shared_lock lock(mutex_);

            std::vector<object_ptr> results;
            results.reserve(objects_by_id_.size() / 10); // Reasonable initial capacity

            for (const auto& [id, obj] : objects_by_id_) {
                if (pred(obj)) {
                    results.push_back(obj);
                }
            }

            return results;
        }

        /**
         * @brief Find first object matching a predicate
         * @return Object if found, nullptr otherwise
         */
        template<typename Predicate>
        [[nodiscard]] object_ptr find_first_if(Predicate pred) const noexcept {
            std::shared_lock lock(mutex_);

            for (const auto& [id, obj] : objects_by_id_) {
                if (pred(obj)) {
                    return obj;
                }
            }

            return nullptr;
        }

        /**
         * @brief Count objects matching a predicate
         */
        template<typename Predicate>
        [[nodiscard]] std::size_t count_if(Predicate pred) const noexcept {
            std::shared_lock lock(mutex_);

            std::size_t count = 0;
            for (const auto& [id, obj] : objects_by_id_) {
                if (pred(obj)) {
                    ++count;
                }
            }

            return count;
        }

        /**
         * @brief Apply function to all registered objects
         * @note Function is called with registry lock held - keep operations brief
         */
        template<typename Function>
        void for_each(Function func) const {
            // Make a snapshot of objects to avoid holding lock during function execution
            std::vector<object_ptr> snapshot;
            {
                std::shared_lock lock(mutex_);
                snapshot.reserve(objects_by_id_.size());
                for (const auto& [id, obj] : objects_by_id_) {
                    snapshot.push_back(obj);
                }
            }
            
            // Execute function on snapshot without holding lock
            for (const auto& obj : snapshot) {
                func(obj);
            }
        }

        // === Event Callbacks ===

        /**
         * @brief Add callback for object registration events
         */
        void add_registration_callback(registration_callback callback) {
            std::lock_guard lock(callback_mutex_);
            registration_callbacks_.push_back(std::move(callback));
        }

        /**
         * @brief Add callback for object unregistration events
         */
        void add_unregistration_callback(unregistration_callback callback) {
            std::lock_guard lock(callback_mutex_);
            unregistration_callbacks_.push_back(std::move(callback));
        }

        /**
         * @brief Clear all callbacks
         */
        void clear_callbacks() {
            std::lock_guard lock(callback_mutex_);
            registration_callbacks_.clear();
            unregistration_callbacks_.clear();
        }

        // === Batch Operations ===

        /**
         * @brief Register multiple objects at once
         * @return Number of successfully registered objects
         */
        template<typename Container>
        std::size_t register_batch(const Container& objects) noexcept {
            std::size_t registered = 0;
            
            std::lock_guard lock(mutex_);
            for (const auto& obj : objects) {
                if (!obj) continue;
                
                auto id = obj->id();
                if (!objects_by_id_.contains(id)) {
                    objects_by_id_[id] = obj;
                    ++registered;
                }
            }
            
            return registered;
        }

        /**
         * @brief Register multiple objects with keys
         * @return Number of successfully registered objects
         */
        template<typename Container>
        std::size_t register_batch_with_keys(const Container& pairs) noexcept {
            std::size_t registered = 0;
            
            std::lock_guard lock(mutex_);
            for (const auto& [key, obj] : pairs) {
                if (!obj) continue;
                
                auto id = obj->id();
                if (!objects_by_id_.contains(id) && !objects_by_key_.contains(key)) {
                    objects_by_id_[id] = obj;
                    objects_by_key_[key] = obj;
                    key_to_id_[key] = id;
                    id_to_key_[id] = key;
                    ++registered;
                }
            }
            
            return registered;
        }

        /**
         * @brief Unregister multiple objects by ID
         * @return Number of successfully unregistered objects
         */
        template<typename Container>
        std::size_t unregister_batch(const Container& ids) noexcept {
            std::size_t unregistered = 0;
            
            std::lock_guard lock(mutex_);
            for (const auto& id : ids) {
                if (objects_by_id_.contains(id)) {
                    // Remove from key mappings if exists
                    auto key_it = id_to_key_.find(id);
                    if (key_it != id_to_key_.end()) {
                        objects_by_key_.erase(key_it->second);
                        key_to_id_.erase(key_it->second);
                        id_to_key_.erase(key_it);
                    }
                    objects_by_id_.erase(id);
                    ++unregistered;
                }
            }
            
            return unregistered;
        }

        // === Statistics and Debug ===

        /**
         * @brief Get registry statistics
         */
        struct Statistics {
            std::size_t total_objects{0};
            std::size_t named_objects{0};
            std::size_t unnamed_objects{0};
            std::string registry_name;
        };

        [[nodiscard]] Statistics get_statistics() const {
            std::shared_lock lock(mutex_);

            return Statistics{
                    .total_objects = objects_by_id_.size(),
                    .named_objects = objects_by_key_.size(),
                    .unnamed_objects = objects_by_id_.size() - objects_by_key_.size(),
                    .registry_name = name_
            };
        }

        /**
         * @brief Get registry name
         */
        [[nodiscard]] const std::string& name() const { return name_; }

        /**
         * @brief Set registry name
         */
        void set_name(std::string_view name) { name_ = name; }

        /**
         * @brief Clean up any dangling weak references (maintenance)
         * @note This operation is potentially expensive and should be called periodically
         */
        std::size_t cleanup_weak_references() {
            std::lock_guard lock(mutex_);

            std::size_t removed = 0;
            std::vector<id_type> to_remove;

            // First pass: identify objects to remove (snapshot under lock)
            for (const auto& [id, obj] : objects_by_id_) {
                if (!obj) {
                    to_remove.push_back(id);
                } else {
                    // Use atomic load for thread-safe ref count check
                    // Note: This is a heuristic - ref_count could change
                    // But we're only cleaning up "likely dead" objects
                    auto count = obj->ref_count();
                    if (count == 1) { // Only registry holds reference
                        to_remove.push_back(id);
                    }
                }
            }

            // Second pass: remove identified objects
            for (auto id : to_remove) {
                // Remove from all maps
                auto key_it = id_to_key_.find(id);
                if (key_it != id_to_key_.end()) {
                    objects_by_key_.erase(key_it->second);
                    key_to_id_.erase(key_it->second);
                    id_to_key_.erase(key_it);
                }
                objects_by_id_.erase(id);
                ++removed;
            }

            return removed;
        }

    private:
        // Internal registration without locking
        bool register_object_impl(const key_type& key, object_ptr obj) {
            auto id = obj->id();

            objects_by_id_[id] = obj;
            objects_by_key_[key] = obj;
            key_to_id_[key] = id;
            id_to_key_[id] = key;

            notify_registration(obj);
            return true;
        }

        // Notification helpers
        void notify_registration(object_ptr obj) {
            // Copy callbacks under lock to avoid holding lock during execution
            std::vector<registration_callback> callbacks_copy;
            {
                std::shared_lock lock(callback_mutex_);
                callbacks_copy = registration_callbacks_;
            }
            
            // Execute callbacks without holding any locks
            for (const auto& callback : callbacks_copy) {
                try {
                    callback(obj);
                } catch (...) {
                    // Swallow exceptions from callbacks to maintain strong guarantee
                    // In production, log this error
                }
            }
        }

        void notify_unregistration(id_type id, std::string_view key) {
            // Copy callbacks under lock to avoid holding lock during execution
            std::vector<unregistration_callback> callbacks_copy;
            {
                std::shared_lock lock(callback_mutex_);
                callbacks_copy = unregistration_callbacks_;
            }
            
            // Execute callbacks without holding any locks
            for (const auto& callback : callbacks_copy) {
                try {
                    callback(id, key);
                } catch (...) {
                    // Swallow exceptions from callbacks to maintain strong guarantee
                    // In production, log this error
                }
            }
        }

        // Data members
        std::string name_{"Registry"};

        mutable std::shared_mutex mutex_;
        std::unordered_map<id_type, object_ptr> objects_by_id_;
        std::unordered_map<key_type, object_ptr> objects_by_key_;
        std::unordered_map<key_type, id_type> key_to_id_;
        std::unordered_map<id_type, key_type> id_to_key_;

        mutable std::shared_mutex callback_mutex_;
        std::vector<registration_callback> registration_callbacks_;
        std::vector<unregistration_callback> unregistration_callbacks_;
    };

// === Specialized Registry Types ===

/**
 * @brief Registry specialized for Object base class
 */
    using ObjectRegistry = Registry<Object, std::string>;

/**
 * @brief Global object registry singleton
 */
    class GlobalObjectRegistry : public Singleton<GlobalObjectRegistry> {
        friend class Singleton<GlobalObjectRegistry>;

    private:
        GlobalObjectRegistry() : registry_("GlobalObjects") {}

    public:
        Registry<Object, std::string>& get_registry() { return registry_; }
        const Registry<Object, std::string>& get_registry() const { return registry_; }

        // Convenience forwarding methods
        bool register_object(object_ptr<Object> obj) {
            return registry_.register_object(obj);
        }

        bool register_object(const std::string& name, object_ptr<Object> obj) {
            return registry_.register_object(name, obj);
        }

        object_ptr<Object> find_by_name(const std::string& name) const {
            return registry_.find_by_key(name);
        }

        object_ptr<Object> find_by_id(Object::id_type id) const {
            return registry_.find_by_id(id);
        }

    private:
        Registry<Object, std::string> registry_;
    };

// === Convenience Functions ===

/**
 * @brief Register object with global registry
 */
    inline bool register_global_object(object_ptr<Object> obj) {
        return GlobalObjectRegistry::instance().register_object(obj);
    }

/**
 * @brief Register named object with global registry
 */
    inline bool register_global_object(const std::string& name, object_ptr<Object> obj) {
        return GlobalObjectRegistry::instance().register_object(name, obj);
    }

/**
 * @brief Find object in global registry by name
 */
    inline object_ptr<Object> find_global_object(const std::string& name) {
        return GlobalObjectRegistry::instance().find_by_name(name);
    }

} // namespace fem::core

#endif //MFEM_REGISTRY_H
