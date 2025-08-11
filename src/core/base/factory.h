#pragma once

#ifndef BASE_FACTORY_H
#define BASE_FACTORY_H

#include <functional>
#include <unordered_map>
#include <string>
#include <string_view>
#include <memory>
#include <type_traits>
#include <concepts>
#include <source_location>

#include "object.h"

namespace fem::core::base {

/**
 * @brief Generic factory for creating objects by name/type
 *
 * Supports:
 * - Registration of creator functions
 * - Type-safe object creation
 * - Automatic type name registration
 * - Parameter forwarding to constructors
 * - Thread-safe registration and creation
 */
    template<ObjectDerived BaseType>
    class Factory {
    public:
        using base_type = BaseType;
        using creator_function = std::function<object_ptr<BaseType>()>;
        using parameterized_creator = std::function<object_ptr<BaseType>(
                const std::unordered_map<std::string, std::string> &)>;

        /**
         * @brief Get the singleton factory instance
         */
        static Factory &instance() {
            static Factory factory;
            return factory;
        }

        /**
         * @brief Register a type with the factory
         * @param name Type name for creation
         * @param creator Function to create instances
         */
        template<std::derived_from<BaseType> ConcreteType>
        bool register_type(std::string_view name) {
            std::lock_guard lock(mutex_);

            auto creator = []() -> object_ptr<BaseType> {
                return object_ptr<BaseType>(new ConcreteType());
            };

            auto param_creator = [](
                    const std::unordered_map<std::string, std::string> &params) -> object_ptr<BaseType> {
                if constexpr(
                            std::is_constructible_v<ConcreteType, const std::unordered_map<std::string, std::string> &>)
                {
                    return object_ptr<BaseType>(new ConcreteType(params));
                } else {
                return object_ptr<BaseType>(new ConcreteType());
            }
            };

            creators_[std::string(name)] = creator;
            param_creators_[std::string(name)] = param_creator;
            type_names_[std::type_index(typeid(ConcreteType))] = name;

            return true;
        }

        /**
         * @brief Register a type with custom creator function
         */
        bool register_creator(std::string_view name, creator_function creator) {
            std::lock_guard lock(mutex_);
            creators_[std::string(name)] = std::move(creator);
            return true;
        }

        /**
         * @brief Register a type with parameterized creator
         */
        bool register_parameterized_creator(std::string_view name, parameterized_creator creator) {
            std::lock_guard lock(mutex_);
            param_creators_[std::string(name)] = std::move(creator);
            return true;
        }

        /**
         * @brief Create an object by type name
         */
        [[nodiscard]] object_ptr<BaseType> create(std::string_view name) const {
            std::shared_lock lock(mutex_);

            auto it = creators_.find(std::string(name));
            if (it != creators_.end()) {
                return it->second();
            }

            throw std::invalid_argument(std::format("Unknown type '{}' in factory", name));
        }

        /**
         * @brief Create an object with parameters
         */
        [[nodiscard]] object_ptr<BaseType> create(std::string_view name,
                                                  const std::unordered_map<std::string, std::string> &params) const {
            std::shared_lock lock(mutex_);

            auto it = param_creators_.find(std::string(name));
            if (it != param_creators_.end()) {
                return it->second(params);
            }

            // Fallback to simple creator if parameterized not available
            return create(name);
        }

        /**
         * @brief Create an object by type
         */
        template<std::derived_from<BaseType> ConcreteType>
        [[nodiscard]] object_ptr<BaseType> create() const {
            auto type_name = get_type_name<ConcreteType>();
            if (!type_name.empty()) {
                return create(type_name);
            }

            // Direct creation if not registered
            return object_ptr<BaseType>(new ConcreteType());
        }

        /**
         * @brief Check if a type is registered
         */
        [[nodiscard]] bool is_registered(std::string_view name) const {
            std::shared_lock lock(mutex_);
            return creators_.contains(std::string(name));
        }

        /**
         * @brief Check if a type is registered by type
         */
        template<std::derived_from<BaseType> ConcreteType>
        [[nodiscard]] bool is_registered() const {
            std::shared_lock lock(mutex_);
            return type_names_.contains(std::type_index(typeid(ConcreteType)));
        }

        /**
         * @brief Get registered type name for a type
         */
        template<std::derived_from<BaseType> ConcreteType>
        [[nodiscard]] std::string get_type_name() const {
            std::shared_lock lock(mutex_);
            auto it = type_names_.find(std::type_index(typeid(ConcreteType)));
            return it != type_names_.end() ? it->second : "";
        }

        /**
         * @brief Get all registered type names
         */
        [[nodiscard]] std::vector<std::string> get_registered_types() const {
            std::shared_lock lock(mutex_);
            std::vector<std::string> types;
            types.reserve(creators_.size());

            for (const auto &[name, _]: creators_) {
                types.push_back(name);
            }

            return types;
        }

        /**
         * @brief Unregister a type
         */
        bool unregister(std::string_view name) {
            std::lock_guard lock(mutex_);

            auto name_str = std::string(name);
            auto removed = creators_.erase(name_str) > 0;
            param_creators_.erase(name_str);

            // Find and remove from type_names_
            for (auto it = type_names_.begin(); it != type_names_.end(); ++it) {
                if (it->second == name_str) {
                    type_names_.erase(it);
                    break;
                }
            }

            return removed;
        }

        /**
         * @brief Clear all registrations
         */
        void clear() {
            std::lock_guard lock(mutex_);
            creators_.clear();
            param_creators_.clear();
            type_names_.clear();
        }

        /**
         * @brief Get factory statistics
         */
        struct Statistics {
            std::size_t registered_types{0};
            std::size_t simple_creators{0};
            std::size_t parameterized_creators{0};
        };

        [[nodiscard]] Statistics get_statistics() const {
            std::shared_lock lock(mutex_);
            return Statistics{
                    .registered_types = creators_.size(),
                    .simple_creators = creators_.size(),
                    .parameterized_creators = param_creators_.size()
            };
        }

    private:
        Factory() = default;

        ~Factory() = default;

        Factory(const Factory &) = delete;

        Factory &operator=(const Factory &) = delete;

        Factory(Factory &&) = delete;

        Factory &operator=(Factory &&) = delete;

        mutable std::shared_mutex mutex_;
        std::unordered_map<std::string, creator_function> creators_;
        std::unordered_map<std::string, parameterized_creator> param_creators_;
        std::unordered_map<std::type_index, std::string> type_names_;
    };

// === Factory Registration Helper ===

/**
 * @brief RAII factory registration helper
 */
    template<ObjectDerived BaseType, std::derived_from<BaseType> ConcreteType>
    class FactoryRegistrar {
    public:
        explicit FactoryRegistrar(std::string_view name) : name_(name) {
            registered_ = Factory<BaseType>::instance().template register_type<ConcreteType>(name);
        }

        ~FactoryRegistrar() {
            if (registered_) {
                Factory<BaseType>::instance().unregister(name_);
            }
        }

        [[nodiscard]] bool is_registered() const

        noexcept { return registered_; }

    private:
        std::string name_;
        bool registered_{false};
    };

// === Convenience Macros ===

/**
 * @brief Register a type with automatic name detection
 */
#define FEM_REGISTER_TYPE(BaseType, ConcreteType) \
    static const fem::core::FactoryRegistrar<BaseType, ConcreteType> \
    registrar_##ConcreteType(#ConcreteType)

/**
 * @brief Register a type with custom name
 */
#define FEM_REGISTER_TYPE_AS(BaseType, ConcreteType, name) \
    static const fem::core::FactoryRegistrar<BaseType, ConcreteType> \
    registrar_##ConcreteType(name)

// === Type Aliases for Common Factories ===

    using ObjectFactory = Factory<Object>;

} // namespace fem::core
#endif //BASE_FACTORY_H
