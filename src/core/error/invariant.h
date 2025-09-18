#pragma once

#ifndef CORE_ERROR_INVARIANT_H
#define CORE_ERROR_INVARIANT_H

#include <string>
#include <functional>
#include <vector>
#include <memory>
#include <format>
#include <source_location>
#include <chrono>
#include "logic_error.h"
#include "contract.h"
#include "error_context.h"

namespace fem::core::error {

/**
 * @brief Invariant checking utilities
 *
 * Provides comprehensive invariant validation for:
 * - Class invariants
 * - Loop invariants
 * - Data structure invariants
 * - System-wide invariants
 */

/**
 * @brief Base class for invariant checkers
 */
class InvariantChecker {
public:
    using CheckFunction = std::function<bool()>;
    using DiagnosticFunction = std::function<std::string()>;

    /**
     * @brief Register an invariant
     */
    void add_invariant(const std::string& name,
                      CheckFunction check,
                      DiagnosticFunction diagnostic = nullptr) {
        invariants_.push_back({name, check, diagnostic});
    }

    /**
     * @brief Check all invariants
     */
    void check_all(const std::source_location& loc = std::source_location::current()) {
        std::vector<std::string> violations;
        
        for (const auto& [name, check, diagnostic] : invariants_) {
            if (!check()) {
                std::string violation = name;
                if (diagnostic) {
                    violation += ": " + diagnostic();
                }
                violations.push_back(violation);
            }
        }

        if (!violations.empty()) {
            std::string message = "Invariant violations:\n";
            for (const auto& violation : violations) {
                message += "  - " + violation + "\n";
            }
            throw InvariantError(get_context(), "multiple invariants",
                               message, loc);
        }
    }

    /**
     * @brief Check specific invariant
     */
    void check(const std::string& name,
              const std::source_location& loc = std::source_location::current()) {
        for (const auto& [inv_name, check, diagnostic] : invariants_) {
            if (inv_name == name) {
                if (!check()) {
                    std::string message = "Invariant '" + name + "' violated";
                    if (diagnostic) {
                        message += ": " + diagnostic();
                    }
                    throw InvariantError(get_context(), name, message, loc);
                }
                return;
            }
        }
        throw std::logic_error("Unknown invariant: " + name);
    }

    /**
     * @brief Check invariants in debug mode only
     */
    void debug_check(const std::source_location& loc = std::source_location::current()) {
#ifndef NDEBUG
        check_all(loc);
#endif
    }

    /**
     * @brief Clear all invariants
     */
    void clear() {
        invariants_.clear();
    }

    /**
     * @brief Get number of registered invariants
     */
    size_t count() const {
        return invariants_.size();
    }

protected:
    virtual std::string get_context() const {
        return "<unknown>";
    }

private:
    struct Invariant {
        std::string name;
        CheckFunction check;
        DiagnosticFunction diagnostic;
    };
    
    std::vector<Invariant> invariants_;
};

/**
 * @brief Class invariant checker
 */
template<typename T>
class ClassInvariant : public InvariantChecker {
public:
    explicit ClassInvariant(const T* object, const std::string& class_name)
        : object_(object), class_name_(class_name) {}

    /**
     * @brief Add member invariant
     */
    template<typename Member>
    ClassInvariant& require(Member T::*member,
                           std::function<bool(const Member&)> predicate,
                           const std::string& description) {
        add_invariant(
            description,
            [this, member, predicate]() {
                return predicate(object_->*member);
            },
            [this, member]() {
                return std::format("Current value: {}",
                                 object_->*member);
            });
        return *this;
    }

    /**
     * @brief Add method-based invariant
     */
    ClassInvariant& require_method(std::function<bool(const T*)> predicate,
                                  const std::string& description) {
        add_invariant(
            description,
            [this, predicate]() { return predicate(object_); });
        return *this;
    }

    /**
     * @brief RAII guard that checks on construction and destruction
     */
    class Guard {
    public:
        explicit Guard(ClassInvariant& invariant)
            : invariant_(invariant) {
            invariant_.check_all();
        }

        ~Guard() {
            try {
                invariant_.check_all();
            } catch (...) {
                // Suppress exceptions in destructor
            }
        }

    private:
        ClassInvariant& invariant_;
    };

    /**
     * @brief Create RAII guard
     */
    Guard guard() {
        return Guard(*this);
    }

protected:
    std::string get_context() const override {
        return class_name_;
    }

private:
    const T* object_;
    std::string class_name_;
};

/**
 * @brief Loop invariant checker
 */
class LoopInvariant {
public:
    explicit LoopInvariant(const std::string& loop_name)
        : loop_name_(loop_name), iteration_(0) {}

    /**
     * @brief Add invariant condition
     */
    LoopInvariant& require(std::function<bool()> condition,
                          const std::string& description) {
        conditions_.emplace_back(condition, description);
        return *this;
    }

    /**
     * @brief Check at loop entry
     */
    void enter(const std::source_location& loc = std::source_location::current()) {
        iteration_ = 0;
        check("loop entry", loc);
    }

    /**
     * @brief Check during iteration
     */
    void iterate(const std::source_location& loc = std::source_location::current()) {
        iteration_++;
        check(std::format("iteration {}", iteration_), loc);
    }

    /**
     * @brief Check at loop exit
     */
    void exit(const std::source_location& loc = std::source_location::current()) {
        check("loop exit", loc);
    }

    /**
     * @brief Get current iteration
     */
    size_t current_iteration() const {
        return iteration_;
    }

private:
    void check(const std::string& phase,
              const std::source_location& loc) {
        for (const auto& [condition, description] : conditions_) {
            if (!condition()) {
                throw InvariantError(
                    loop_name_,
                    description,
                    std::format("Loop invariant violated at {}", phase),
                    loc);
            }
        }
    }

    std::string loop_name_;
    size_t iteration_;
    std::vector<std::pair<std::function<bool()>, std::string>> conditions_;
};

/**
 * @brief Data structure invariant checker
 */
template<typename Container>
class DataStructureInvariant : public InvariantChecker {
public:
    explicit DataStructureInvariant(const Container* container,
                                   const std::string& name)
        : container_(container), name_(name) {}

    /**
     * @brief Check size constraints
     */
    DataStructureInvariant& size_in_range(size_t min, size_t max) {
        add_invariant(
            "size constraint",
            [this, min, max]() {
                size_t size = container_->size();
                return size >= min && size <= max;
            },
            [this]() {
                return std::format("Current size: {}", container_->size());
            });
        return *this;
    }

    /**
     * @brief Check ordering
     */
    DataStructureInvariant& is_sorted(
            std::function<bool(const typename Container::value_type&,
                              const typename Container::value_type&)> comparator
                = std::less<typename Container::value_type>()) {
        add_invariant(
            "sorted order",
            [this, comparator]() {
                return std::is_sorted(container_->begin(),
                                    container_->end(),
                                    comparator);
            });
        return *this;
    }

    /**
     * @brief Check uniqueness
     */
    DataStructureInvariant& all_unique() {
        add_invariant(
            "uniqueness",
            [this]() {
                std::set<typename Container::value_type> seen;
                for (const auto& item : *container_) {
                    if (!seen.insert(item).second) {
                        return false;
                    }
                }
                return true;
            });
        return *this;
    }

    /**
     * @brief Check element constraint
     */
    DataStructureInvariant& all_elements(
            std::function<bool(const typename Container::value_type&)> predicate,
            const std::string& description) {
        add_invariant(
            description,
            [this, predicate]() {
                return std::all_of(container_->begin(),
                                 container_->end(),
                                 predicate);
            });
        return *this;
    }

protected:
    std::string get_context() const override {
        return name_;
    }

private:
    const Container* container_;
    std::string name_;
};

/**
 * @brief Numeric invariant checker
 */
class NumericInvariant : public InvariantChecker {
public:
    explicit NumericInvariant(const std::string& context)
        : context_(context) {}

    /**
     * @brief Check conservation law
     */
    template<typename T>
    NumericInvariant& conserves(std::function<T()> quantity,
                               const std::string& name,
                               T initial_value,
                               T tolerance) {
        add_invariant(
            std::format("{} conservation", name),
            [quantity, initial_value, tolerance]() {
                return std::abs(quantity() - initial_value) <= tolerance;
            },
            [quantity, initial_value]() {
                return std::format("Initial: {}, Current: {}",
                                 initial_value, quantity());
            });
        return *this;
    }

    /**
     * @brief Check positive definiteness
     */
    template<typename Matrix>
    NumericInvariant& positive_definite(std::function<Matrix()> get_matrix,
                                       const std::string& name) {
        add_invariant(
            std::format("{} positive definite", name),
            [get_matrix]() {
                // Simplified check - would need eigenvalue computation
                auto mat = get_matrix();
                for (size_t i = 0; i < mat.rows(); ++i) {
                    if (mat(i, i) <= 0) return false;
                }
                return true;
            });
        return *this;
    }

    /**
     * @brief Check bounds
     */
    template<typename T>
    NumericInvariant& bounded(std::function<T()> value,
                             const std::string& name,
                             T min, T max) {
        add_invariant(
            std::format("{} bounds", name),
            [value, min, max]() {
                T val = value();
                return val >= min && val <= max;
            },
            [value]() {
                return std::format("Current value: {}", value());
            });
        return *this;
    }

protected:
    std::string get_context() const override {
        return context_;
    }

private:
    std::string context_;
};

/**
 * @brief System-wide invariant manager
 */
class SystemInvariantManager {
public:
    /**
     * @brief Get singleton instance
     */
    static SystemInvariantManager& instance() {
        static SystemInvariantManager manager;
        return manager;
    }

    /**
     * @brief Register an invariant
     */
    void register_invariant(const std::string& name,
                          std::shared_ptr<InvariantChecker> checker) {
        invariants_[name] = checker;
    }

    /**
     * @brief Unregister an invariant
     */
    void unregister_invariant(const std::string& name) {
        invariants_.erase(name);
    }

    /**
     * @brief Check all system invariants
     */
    void check_all() {
        for (const auto& [name, checker] : invariants_) {
            checker->check_all();
        }
    }

    /**
     * @brief Check specific invariant
     */
    void check(const std::string& name) {
        auto it = invariants_.find(name);
        if (it != invariants_.end()) {
            it->second->check_all();
        }
    }

    /**
     * @brief Get all invariant names
     */
    std::vector<std::string> get_names() const {
        std::vector<std::string> names;
        for (const auto& [name, _] : invariants_) {
            names.push_back(name);
        }
        return names;
    }

private:
    SystemInvariantManager() = default;
    std::map<std::string, std::shared_ptr<InvariantChecker>> invariants_;
};

/**
 * @brief RAII invariant scope
 */
class InvariantScope {
public:
    explicit InvariantScope(std::function<bool()> check,
                           const std::string& description)
        : check_(check)
        , description_(description) {
        if (!check_()) {
            throw InvariantError("<scope>", description_,
                               "Invariant violated at scope entry");
        }
    }

    ~InvariantScope() {
        if (std::uncaught_exceptions() == 0) {
            try {
                if (!check_()) {
                    // Can't throw from destructor
                    // Log the violation
                }
            } catch (...) {
                // Suppress exceptions
            }
        }
    }

private:
    std::function<bool()> check_;
    std::string description_;
};

/**
 * @brief Helper functions
 */

// Create class invariant
template<typename T>
auto make_class_invariant(const T* object, const std::string& class_name) {
    return ClassInvariant<T>(object, class_name);
}

// Create data structure invariant
template<typename Container>
auto make_container_invariant(const Container* container,
                             const std::string& name) {
    return DataStructureInvariant<Container>(container, name);
}

// Create invariant scope
inline auto invariant_scope(std::function<bool()> check,
                           const std::string& description) {
    return InvariantScope(check, description);
}

} // namespace fem::core::error

#endif // CORE_ERROR_INVARIANT_H