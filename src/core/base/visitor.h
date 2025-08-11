#pragma once

#ifndef MFEM_VISITOR_H
#define MFEM_VISITOR_H

#include <type_traits>
#include <concepts>
#include <memory>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <string>
#include <vector>
#include <any>
#include <algorithm>

namespace fem::core::base {

// Forward declarations
    template<typename T> class Visitor;
    template<typename T> class Visitable;

/**
 * @brief Base visitor interface for type-erased visiting
 *
 * This allows storing visitors of different types in containers
 * and provides a common interface for visitor management.
 */
    class BaseVisitor {
    public:
        virtual ~BaseVisitor() = default;

        /**
         * @brief Get visitor name for debugging/identification
         */
        [[nodiscard]] virtual std::string get_name() const = 0;

        /**
         * @brief Get visitor description
         */
        [[nodiscard]] virtual std::string get_description() const { return get_name(); }

        /**
         * @brief Check if visitor can visit objects of given type
         */
        [[nodiscard]] virtual bool can_visit(std::type_index type) const = 0;

    protected:
        BaseVisitor() = default;
    };

/**
 * @brief Generic visitor interface using CRTP
 *
 * Template parameter T should be the base type of objects this visitor can visit.
 * Concrete visitors should inherit from this and implement visit methods for
 * specific derived types.
 */
    template<typename T>
    class Visitor : public BaseVisitor {
    public:
        using visitable_type = T;

        /**
         * @brief Visit an object of the base type
         * Default implementation tries to dispatch to specific visit methods
         */
        virtual void visit(T& object) {
            // Default implementation - can be overridden for custom behavior
            visit_impl(object);
        }

        /**
         * @brief Visit a const object of the base type
         */
        virtual void visit(const T& object) {
            // Default implementation - can be overridden for custom behavior
            visit_impl(object);
        }

        /**
         * @brief Check if this visitor can visit objects of type T
         */
        [[nodiscard]] bool can_visit(std::type_index type) const override {
            return type == std::type_index(typeid(T)) ||
                   derived_types_.find(type) != derived_types_.end();
        }

    protected:
        /**
         * @brief Register that this visitor can handle a derived type
         */
        template<typename Derived>
        void register_visitable_type() {
            static_assert(std::is_base_of_v<T, Derived>, "Derived must inherit from T");
            derived_types_.insert(std::type_index(typeid(Derived)));
        }

        /**
         * @brief Internal visit implementation - can be specialized by derived visitors
         */
        virtual void visit_impl(T& object) {}
        virtual void visit_impl(const T& object) {}

    private:
        std::unordered_set<std::type_index> derived_types_;
    };

/**
 * @brief Interface for objects that can be visited
 *
 * Objects implementing this interface can accept visitors.
 * Uses CRTP to provide type-safe visitor acceptance.
 */
    template<typename T>
    class Visitable {
    public:
        /**
         * @brief Accept a visitor
         */
        virtual void accept(Visitor<T>& visitor) {
            visitor.visit(static_cast<T&>(*this));
        }

        /**
         * @brief Accept a const visitor
         */
        virtual void accept(Visitor<T>& visitor) const {
            visitor.visit(static_cast<const T&>(*this));
        }

        /**
         * @brief Accept a visitor through base visitor interface
         */
        virtual void accept_base(BaseVisitor& visitor) {
            if (auto* typed_visitor = dynamic_cast<Visitor<T>*>(&visitor)) {
                accept(*typed_visitor);
            }
        }

    protected:
        Visitable() = default;
        virtual ~Visitable() = default;
    };

/**
 * @brief Hierarchical visitor that can visit object hierarchies
 *
 * Provides pre-order, post-order, and in-order traversal capabilities
 * for objects that have hierarchical relationships.
 */
    template<typename T>
    class HierarchicalVisitor : public Visitor<T> {
    public:
        enum class TraversalOrder {
            PRE_ORDER,   // Visit parent before children
            POST_ORDER,  // Visit children before parent
            IN_ORDER     // Visit first child, parent, then remaining children
        };

        explicit HierarchicalVisitor(TraversalOrder order = TraversalOrder::PRE_ORDER)
                : traversal_order_(order) {}

        /**
         * @brief Visit an object and its hierarchy
         */
        void visit(T& object) override {
            switch (traversal_order_) {
                case TraversalOrder::PRE_ORDER:
                    visit_pre_order(object);
                    break;
                case TraversalOrder::POST_ORDER:
                    visit_post_order(object);
                    break;
                case TraversalOrder::IN_ORDER:
                    visit_in_order(object);
                    break;
            }
        }

        /**
         * @brief Set traversal order
         */
        void set_traversal_order(TraversalOrder order) { traversal_order_ = order; }

        /**
         * @brief Get current traversal order
         */
        [[nodiscard]] TraversalOrder get_traversal_order() const { return traversal_order_; }

    protected:
        /**
         * @brief Visit the object itself (to be implemented by derived classes)
         */
        virtual void visit_object(T& object) = 0;

        /**
         * @brief Get children of an object (to be implemented by derived classes)
         */
        virtual std::vector<T*> get_children(T& object) { return {}; }

    private:
        void visit_pre_order(T& object) {
            visit_object(object);
            for (auto* child : get_children(object)) {
                if (child) visit(*child);
            }
        }

        void visit_post_order(T& object) {
            for (auto* child : get_children(object)) {
                if (child) visit(*child);
            }
            visit_object(object);
        }

        void visit_in_order(T& object) {
            auto children = get_children(object);
            if (!children.empty() && children[0]) {
                visit(*children[0]);
            }
            visit_object(object);
            for (size_t i = 1; i < children.size(); ++i) {
                if (children[i]) visit(*children[i]);
            }
        }

        TraversalOrder traversal_order_;
    };

/**
 * @brief Composite visitor that applies multiple visitors in sequence
 *
 * Useful for applying multiple operations to the same set of objects
 * without multiple traversals.
 */
    template<typename T>
    class CompositeVisitor : public Visitor<T> {
    public:
        /**
         * @brief Add a visitor to the composite
         */
        void add_visitor(std::unique_ptr<Visitor<T>> visitor) {
            visitors_.push_back(std::move(visitor));
        }

        /**
         * @brief Add a visitor by reference (visitor lifetime managed externally)
         */
        void add_visitor(Visitor<T>& visitor) {
            visitor_refs_.push_back(&visitor);
        }

        /**
         * @brief Visit object with all contained visitors
         */
        void visit(T& object) override {
            for (auto& visitor : visitors_) {
                visitor->visit(object);
            }
            for (auto* visitor : visitor_refs_) {
                visitor->visit(object);
            }
        }

        /**
         * @brief Visit const object with all contained visitors
         */
        void visit(const T& object) override {
            for (auto& visitor : visitors_) {
                visitor->visit(object);
            }
            for (auto* visitor : visitor_refs_) {
                visitor->visit(object);
            }
        }

        /**
         * @brief Get number of contained visitors
         */
        [[nodiscard]] size_t get_visitor_count() const {
            return visitors_.size() + visitor_refs_.size();
        }

        /**
         * @brief Clear all visitors
         */
        void clear() {
            visitors_.clear();
            visitor_refs_.clear();
        }

        // BaseVisitor interface
        [[nodiscard]] std::string get_name() const override {
            return "CompositeVisitor(" + std::to_string(get_visitor_count()) + " visitors)";
        }

    private:
        std::vector<std::unique_ptr<Visitor<T>>> visitors_;
        std::vector<Visitor<T>*> visitor_refs_;
    };

/**
 * @brief Conditional visitor that only visits objects meeting certain criteria
 *
 * Wraps another visitor and only applies it to objects that satisfy a predicate.
 */
    template<typename T>
    class ConditionalVisitor : public Visitor<T> {
    public:
        using predicate_type = std::function<bool(const T&)>;

        ConditionalVisitor(std::unique_ptr<Visitor<T>> wrapped_visitor, predicate_type predicate)
                : wrapped_visitor_(std::move(wrapped_visitor)), predicate_(std::move(predicate)) {}

        void visit(T& object) override {
            if (predicate_(object)) {
                wrapped_visitor_->visit(object);
            }
        }

        void visit(const T& object) override {
            if (predicate_(object)) {
                wrapped_visitor_->visit(object);
            }
        }

        [[nodiscard]] std::string get_name() const override {
            return "ConditionalVisitor(" + wrapped_visitor_->get_name() + ")";
        }

    private:
        std::unique_ptr<Visitor<T>> wrapped_visitor_;
        predicate_type predicate_;
    };

/**
 * @brief Function visitor that wraps a simple function as a visitor
 *
 * Convenient for simple operations that don't require a full visitor class.
 */
    template<typename T>
    class FunctionVisitor : public Visitor<T> {
    public:
        using visit_function = std::function<void(T&)>;
        using const_visit_function = std::function<void(const T&)>;

        explicit FunctionVisitor(visit_function func, std::string_view name = "FunctionVisitor")
                : visit_func_(std::move(func)), name_(name) {}

        FunctionVisitor(visit_function func, const_visit_function const_func, std::string_view name = "FunctionVisitor")
                : visit_func_(std::move(func)), const_visit_func_(std::move(const_func)), name_(name) {}

        void visit(T& object) override {
            if (visit_func_) {
                visit_func_(object);
            }
        }

        void visit(const T& object) override {
            if (const_visit_func_) {
                const_visit_func_(object);
            } else if (visit_func_) {
                // Fallback to non-const version if const version not provided
                visit_func_(const_cast<T&>(object));
            }
        }

        [[nodiscard]] std::string get_name() const override {
            return name_;
        }

    private:
        visit_function visit_func_;
        const_visit_function const_visit_func_;
        std::string name_;
    };

/**
 * @brief Visitor registry for managing and discovering visitors
 *
 * Allows registration of visitor factories and creation of visitors by name.
 * Useful for plugin architectures and configuration-driven visitor selection.
 */
    template<typename T>
    class VisitorRegistry {
    public:
        using visitor_factory = std::function<std::unique_ptr<Visitor<T>>()>;

        /**
         * @brief Register a visitor factory
         */
        void register_visitor(const std::string& name, visitor_factory factory) {
            factories_[name] = std::move(factory);
        }

        /**
         * @brief Register a visitor type (must have default constructor)
         */
        template<typename VisitorType>
        void register_visitor_type(const std::string& name) {
            static_assert(std::is_base_of_v<Visitor<T>, VisitorType>,
                          "VisitorType must inherit from Visitor<T>");

            register_visitor(name, []() -> std::unique_ptr<Visitor<T>> {
                return std::make_unique<VisitorType>();
            });
        }

        /**
         * @brief Create a visitor by name
         */
        [[nodiscard]] std::unique_ptr<Visitor<T>> create_visitor(const std::string& name) const {
            auto it = factories_.find(name);
            return it != factories_.end() ? it->second() : nullptr;
        }

        /**
         * @brief Get all registered visitor names
         */
        [[nodiscard]] std::vector<std::string> get_registered_names() const {
            std::vector<std::string> names;
            names.reserve(factories_.size());
            for (const auto& [name, factory] : factories_) {
                names.push_back(name);
            }
            return names;
        }

        /**
         * @brief Check if a visitor is registered
         */
        [[nodiscard]] bool is_registered(const std::string& name) const {
            return factories_.find(name) != factories_.end();
        }

        /**
         * @brief Clear all registered visitors
         */
        void clear() {
            factories_.clear();
        }

    private:
        std::unordered_map<std::string, visitor_factory> factories_;
    };

/**
 * @brief Visitor coordinator for managing visitor application to object collections
 *
 * Provides convenient methods for applying visitors to collections of objects
 * with various filtering and ordering options.
 */
    template<typename T>
    class VisitorCoordinator {
    public:
        /**
         * @brief Apply visitor to a collection of objects
         */
        template<typename Container>
        static void apply_visitor(Visitor<T>& visitor, Container& objects) {
            for (auto& obj : objects) {
                if constexpr (std::is_pointer_v<typename Container::value_type>) {
                    if (obj) visitor.visit(*obj);
                } else {
                    visitor.visit(obj);
                }
            }
        }

        /**
         * @brief Apply visitor to a collection with a predicate filter
         */
        template<typename Container, typename Predicate>
        static void apply_visitor_if(Visitor<T>& visitor, Container& objects, Predicate pred) {
            for (auto& obj : objects) {
                if constexpr (std::is_pointer_v<typename Container::value_type>) {
                    if (obj && pred(*obj)) visitor.visit(*obj);
                } else {
                    if (pred(obj)) visitor.visit(obj);
                }
            }
        }

        /**
         * @brief Apply multiple visitors to a collection
         */
        template<typename Container>
        static void apply_visitors(const std::vector<std::unique_ptr<Visitor<T>>>& visitors, Container& objects) {
            for (auto& visitor : visitors) {
                apply_visitor(*visitor, objects);
            }
        }

        /**
         * @brief Apply visitor to collection in parallel (requires thread-safe visitor)
         */
        template<typename Container>
        static void apply_visitor_parallel(Visitor<T>& visitor, Container& objects) {
            // Note: This is a simplified parallel implementation
            // In a real implementation, you'd use std::execution or threading library
#pragma omp parallel for
            for (size_t i = 0; i < objects.size(); ++i) {
                if constexpr (std::is_pointer_v<typename Container::value_type>) {
                    if (objects[i]) visitor.visit(*objects[i]);
                } else {
                    visitor.visit(objects[i]);
                }
            }
        }
    };

// === Convenience Factory Functions ===

/**
 * @brief Create a function visitor from a lambda or function
 */
    template<typename T, typename Func>
    [[nodiscard]] std::unique_ptr<FunctionVisitor<T>> make_function_visitor(Func func, std::string_view name = "Lambda") {
        return std::make_unique<FunctionVisitor<T>>(std::move(func), name);
    }

/**
 * @brief Create a conditional visitor
 */
    template<typename T, typename Predicate>
    [[nodiscard]] std::unique_ptr<ConditionalVisitor<T>>
    make_conditional_visitor(std::unique_ptr<Visitor<T>> visitor, Predicate pred) {
        return std::make_unique<ConditionalVisitor<T>>(std::move(visitor), std::move(pred));
    }

/**
 * @brief Create a composite visitor
 */
    template<typename T>
    [[nodiscard]] std::unique_ptr<CompositeVisitor<T>> make_composite_visitor() {
        return std::make_unique<CompositeVisitor<T>>();
    }

// === Visitor Pattern Macros ===

/**
 * @brief Macro to define visitor acceptance in visitable classes
 */
#define FEM_DECLARE_VISITABLE(BaseType) \
    void accept(fem::core::Visitor<BaseType>& visitor) override { \
        visitor.visit(*this); \
    } \
    void accept(fem::core::Visitor<BaseType>& visitor) const override { \
        visitor.visit(*this); \
    }

/**
 * @brief Macro to define a simple visitor class
 */
#define FEM_DECLARE_VISITOR(VisitorName, BaseType) \
    class VisitorName : public fem::core::Visitor<BaseType> { \
    public: \
        std::string get_name() const override { return #VisitorName; } \
    private: \
        void visit_impl(BaseType& object) override;

#define FEM_END_VISITOR() \
    };

} // namespace fem::core

#endif //MFEM_VISITOR_H
