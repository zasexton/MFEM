#pragma once

#ifndef BASE_COMPONENT_H
#define BASE_COMPONENT_H

#include "Object.hpp"
#include <typeindex>
#include <unordered_map>
#include <memory>
#include <vector>
#include <string>
#include <type_traits>
#include <concepts>
#include <functional>
#include <optional>

namespace fem::core::base {

// Forward declarations
    class ComponentManager;
    class Entity;

/**
 * @brief Base class for all components in the ECS (Entity-Component-System) architecture
 *
 * Components represent data and behavior that can be attached to entities.
 * Unlike inheritance-based design, components enable composition of functionality:
 * - An element can have MechanicalComponent + ThermalComponent + ElectromagneticComponent
 * - A node can have DOFComponent + ConstraintComponent + ContactComponent
 * - A material can have ElasticComponent + PlasticComponent + DamageComponent
 *
 * Design principles:
 * - Lightweight: Components should be simple data/behavior holders
 * - Composable: Multiple components can work together
 * - Reusable: Same component type can be used by different entities
 * - Type-safe: Component types are checked at compile time
 */
    class Component {
    public:
        /**
         * @brief Constructor with optional component name
         */
        explicit Component(std::string_view name = "");

        /**
         * @brief Virtual destructor
         */
        virtual ~Component() = default;

        /**
         * @brief Get component type information
         */
        [[nodiscard]] virtual std::type_index get_type() const = 0;

        /**
         * @brief Get human-readable component type name
         */
        [[nodiscard]] virtual std::string get_type_name() const = 0;

        /**
         * @brief Get component instance name
         */
        [[nodiscard]] const std::string& get_name() const { return name_; }

        /**
         * @brief Set component instance name
         */
        void set_name(std::string_view name) { name_ = name; }

        /**
         * @brief Get the entity that owns this component
         */
        [[nodiscard]] Entity* get_owner() const { return owner_; }

        /**
         * @brief Called when component is attached to an entity
         */
        virtual void on_attach(Entity* entity) { owner_ = entity; }

        /**
         * @brief Called when component is detached from an entity
         */
        virtual void on_detach() { owner_ = nullptr; }

        /**
         * @brief Called when component needs to update
         */
        virtual void update(double dt) {}

        /**
         * @brief Called when component should reset to initial state
         */
        virtual void reset() {}

        /**
         * @brief Check if component is active
         */
        [[nodiscard]] bool is_active() const { return active_; }

        /**
         * @brief Set component active state
         */
        void set_active(bool active) { active_ = active; }

        /**
         * @brief Serialize component data
         */
        virtual std::string serialize() const { return "{}"; }

        /**
         * @brief Deserialize component data
         */
        virtual bool deserialize(const std::string& data) { return true; }

        /**
         * @brief Get component dependencies (other component types this depends on)
         */
        virtual std::vector<std::type_index> get_dependencies() const { return {}; }

        /**
         * @brief Check if component is compatible with another component type
         */
        virtual bool is_compatible_with(std::type_index other_type) const { return true; }

    protected:
        std::string name_;
        Entity* owner_{nullptr};
        bool active_{true};
    };

/**
 * @brief CRTP helper for typed components
 *
 * Provides automatic type information and type-safe casting
 */
    template<typename Derived>
    class TypedComponent : public Component {
    public:
        explicit TypedComponent(std::string_view name = "") : Component(name) {}

        /**
         * @brief Get component type (automatically derived)
         */
        [[nodiscard]] std::type_index get_type() const override {
            return std::type_index(typeid(Derived));
        }

        /**
         * @brief Get component type name (automatically derived)
         */
        [[nodiscard]] std::string get_type_name() const override {
            return typeid(Derived).name();
        }

        /**
         * @brief Safe cast to derived type
         */
        template<typename T>
        [[nodiscard]] T* as() {
            return dynamic_cast<T*>(this);
        }

        template<typename T>
        [[nodiscard]] const T* as() const {
            return dynamic_cast<const T*>(this);
        }
    };

/**
 * @brief Entity class that can hold multiple components
 *
 * Entities are lightweight containers for components. They don't inherit
 * from Object by default to keep them lightweight, but can optionally
 * do so if needed for specific use cases.
 */
    class Entity {
    public:
        using ComponentPtr = std::unique_ptr<Component>;
        using ComponentMap = std::unordered_map<std::type_index, ComponentPtr>;

        /**
         * @brief Constructor with optional entity name
         */
        explicit Entity(std::string_view name = "");

        /**
         * @brief Destructor
         */
        virtual ~Entity();

        /**
         * @brief Add a component to this entity
         */
        template<typename T, typename... Args>
        T* add_component(Args&&... args) {
            static_assert(std::is_base_of_v<Component, T>, "T must be derived from Component");

            auto component = std::make_unique<T>(std::forward<Args>(args)...);
            T* component_ptr = component.get();

            std::type_index type_id = component->get_type();

            // Check dependencies
            if (!check_dependencies<T>()) {
                return nullptr;
            }

            // Check compatibility
            if (!check_compatibility(type_id)) {
                return nullptr;
            }

            // Remove existing component of same type
            remove_component<T>();

            // Add new component
            component->on_attach(this);
            components_[type_id] = std::move(component);

            return component_ptr;
        }

        /**
         * @brief Get component of specific type
         */
        template<typename T>
        [[nodiscard]] T* get_component() {
            static_assert(std::is_base_of_v<Component, T>, "T must be derived from Component");

            std::type_index type_id(typeid(T));
            auto it = components_.find(type_id);

            if (it != components_.end()) {
                return static_cast<T*>(it->second.get());
            }

            return nullptr;
        }

        template<typename T>
        [[nodiscard]] const T* get_component() const {
            return const_cast<Entity*>(this)->get_component<T>();
        }

        /**
         * @brief Check if entity has component of specific type
         */
        template<typename T>
        [[nodiscard]] bool has_component() const {
            std::type_index type_id(typeid(T));
            return components_.contains(type_id);
        }

        /**
         * @brief Remove component of specific type
         */
        template<typename T>
        bool remove_component() {
            std::type_index type_id(typeid(T));
            auto it = components_.find(type_id);

            if (it != components_.end()) {
                it->second->on_detach();
                components_.erase(it);
                return true;
            }

            return false;
        }

        /**
         * @brief Get all components
         */
        [[nodiscard]] const ComponentMap& get_all_components() const {
            return components_;
        }

        /**
         * @brief Update all components
         */
        virtual void update(double dt) {
            for (auto& [type, component] : components_) {
                if (component->is_active()) {
                    component->update(dt);
                }
            }
        }

        /**
         * @brief Reset all components
         */
        virtual void reset() {
            for (auto& [type, component] : components_) {
                component->reset();
            }
        }

        /**
         * @brief Get entity name
         */
        [[nodiscard]] const std::string& get_name() const { return name_; }

        /**
         * @brief Set entity name
         */
        void set_name(std::string_view name) { name_ = name; }

        /**
         * @brief Get number of components
         */
        [[nodiscard]] size_t get_component_count() const { return components_.size(); }

        /**
         * @brief Check if entity is active
         */
        [[nodiscard]] bool is_active() const { return active_; }

        /**
         * @brief Set entity active state
         */
        void set_active(bool active) { active_ = active; }

        /**
         * @brief Apply function to all components of specific type
         */
        template<typename T, typename Func>
        void for_each_component(Func func) {
            if (auto* component = get_component<T>()) {
                func(component);
            }
        }

        /**
         * @brief Apply function to all components
         */
        template<typename Func>
        void for_each_component(Func func) {
            for (auto& [type, component] : components_) {
                func(component.get());
            }
        }

    protected:
        /**
         * @brief Check if all dependencies for component type T are satisfied
         */
        template<typename T>
        bool check_dependencies() {
            // Create temporary instance to get dependencies
            T temp_component;
            auto dependencies = temp_component.get_dependencies();

            for (auto dep_type : dependencies) {
                if (components_.find(dep_type) == components_.end()) {
                    return false; // Missing dependency
                }
            }

            return true;
        }

        /**
         * @brief Check if component type is compatible with existing components
         */
        bool check_compatibility(std::type_index new_type) {
            for (auto& [existing_type, component] : components_) {
                if (!component->is_compatible_with(new_type)) {
                    return false;
                }
            }
            return true;
        }

    private:
        std::string name_;
        ComponentMap components_;
        bool active_{true};
    };

/**
 * @brief Enhanced Entity that inherits from Object
 *
 * Use this when you need Object features (ID, ref counting, etc.)
 * along with component-based architecture
 */
    class ObjectEntity : public Object, public Entity {
    public:
        explicit ObjectEntity(std::string_view name = "")
                : Object("ObjectEntity"), Entity(name) {}

        /**
         * @brief Update both Object and Entity
         */
        void update(double dt) override {
            Entity::update(dt);
        }

        /**
         * @brief Reset both Object and Entity
         */
        void reset() override {
            Entity::reset();
        }
    };

// === Component System Manager ===

/**
 * @brief System for managing component lifecycles and interactions
 */
    class ComponentSystem {
    public:
        /**
         * @brief Register a component type with the system
         */
        template<typename T>
        void register_component_type() {
            static_assert(std::is_base_of_v<Component, T>, "T must be derived from Component");

            std::type_index type_id(typeid(T));
            component_factories_[type_id] = []() -> std::unique_ptr<Component> {
                return std::make_unique<T>();
            };
        }

        /**
         * @brief Create component by type name
         */
        std::unique_ptr<Component> create_component(const std::string& type_name) {
            for (auto& [type_id, factory] : component_factories_) {
                auto component = factory();
                if (component->get_type_name() == type_name) {
                    return component;
                }
            }
            return nullptr;
        }

        /**
         * @brief Get all registered component types
         */
        [[nodiscard]] std::vector<std::string> get_registered_types() const {
            std::vector<std::string> types;
            for (auto& [type_id, factory] : component_factories_) {
                auto component = factory();
                types.push_back(component->get_type_name());
            }
            return types;
        }

        /**
         * @brief Update all entities in the system
         */
        void update_all_entities(double dt) {
            for (auto* entity : managed_entities_) {
                if (entity->is_active()) {
                    entity->update(dt);
                }
            }
        }

        /**
         * @brief Register entity with system for management
         */
        void manage_entity(Entity* entity) {
            managed_entities_.insert(entity);
        }

        /**
         * @brief Unregister entity from system
         */
        void unmanage_entity(Entity* entity) {
            managed_entities_.erase(entity);
        }

    private:
        std::unordered_map<std::type_index, std::function<std::unique_ptr<Component>()>> component_factories_;
        std::unordered_set<Entity*> managed_entities_;
    };

} // namespace fem::core

#endif //BASE_COMPONENT_H
