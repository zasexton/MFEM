#pragma once

#ifndef CORE_ERROR_ERROR_CATEGORY_H
#define CORE_ERROR_ERROR_CATEGORY_H

#include <string>
#include <system_error>
#include <memory>
#include <vector>
#include "../base/object.h"
#include "../base/registry.h"

namespace fem::core::error {

/**
 * @brief Extended error category with additional features
 *
 * Provides a registry-based error category system that extends
 * std::error_category with additional functionality and inherits from Object
 * to enable registry management.
 */
class ExtendedErrorCategory : public std::error_category, public base::Object {
public:
    /**
     * @brief Constructor
     */
    ExtendedErrorCategory(std::string_view class_name = "ExtendedErrorCategory")
        : base::Object(class_name) {}

    /**
     * @brief Virtual destructor
     */
    virtual ~ExtendedErrorCategory() = default;

    // Resolve ambiguity by explicitly using std::error_category's comparison
    using std::error_category::operator==;

    /**
     * @brief Get the domain of this error category
     */
    virtual std::string domain() const noexcept {
        return "fem.core";
    }

    /**
     * @brief Check if this error code is recoverable
     */
    virtual bool is_recoverable([[maybe_unused]] int code) const noexcept {
        return false;  // By default, errors are not recoverable
    }

    /**
     * @brief Get suggested action for this error
     */
    virtual std::string suggested_action([[maybe_unused]] int code) const {
        return "Check error documentation";
    }

    /**
     * @brief Get severity level (0=info, 1=warning, 2=error, 3=fatal)
     */
    virtual int severity(int code) const noexcept {
        return code == 0 ? 0 : 2;  // Success is info, others are errors
    }

    // Pure virtual functions from std::error_category
    // These must be implemented by derived classes
    const char* name() const noexcept override = 0;
    std::string message(int code) const override = 0;
};

/**
 * @brief Registry for error categories
 *
 * Allows registration and lookup of error categories by name
 */
class ErrorCategoryRegistry : public base::Registry<ExtendedErrorCategory> {
public:
    static ErrorCategoryRegistry& instance() {
        static ErrorCategoryRegistry registry;
        return registry;
    }

    /**
     * @brief Register an error category
     */
    bool register_category(const std::string& name,
                          base::object_ptr<ExtendedErrorCategory> category) {
        return register_object(name, std::move(category));
    }

    /**
     * @brief Get an error category by name
     */
    base::object_ptr<ExtendedErrorCategory> get_category(const std::string& name) {
        return find_by_key(name);
    }

    /**
     * @brief Check if a category exists
     */
    bool has_category(const std::string& name) const {
        return contains_key(name);
    }

    /**
     * @brief Unregister an error category
     */
    bool unregister_category(const std::string& name) {
        return unregister_by_key(name);
    }

    /**
     * @brief List all registered categories
     */
    std::vector<std::string> list_categories() const {
        return get_all_keys();
    }

    /**
     * @brief Clear all registered categories
     */
    void clear() {
        base::Registry<ExtendedErrorCategory>::clear();
    }

private:
    ErrorCategoryRegistry() = default;
};

/**
 * @brief Domain-specific error category
 *
 * Template for creating domain-specific error categories
 */
template<typename DomainTag>
class DomainErrorCategory : public ExtendedErrorCategory {
public:
    const char* name() const noexcept override {
        return DomainTag::name();
    }

    std::string domain() const noexcept override {
        return DomainTag::domain();
    }
};

/**
 * @brief Numeric error category
 *
 * For numerical computation errors
 */
class NumericErrorCategory : public ExtendedErrorCategory {
public:
    const char* name() const noexcept override {
        return "fem.numeric";
    }

    std::string domain() const noexcept override {
        return "numeric";
    }

    std::string message(int code) const override;
    bool is_recoverable(int code) const noexcept override;
    int severity(int code) const noexcept override;
};

/**
 * @brief I/O error category
 *
 * For file and stream I/O errors
 */
class IOErrorCategory : public ExtendedErrorCategory {
public:
    const char* name() const noexcept override {
        return "fem.io";
    }

    std::string domain() const noexcept override {
        return "io";
    }

    std::string message(int code) const override;
    bool is_recoverable(int code) const noexcept override;
    std::string suggested_action(int code) const override;
};

/**
 * @brief Memory error category
 *
 * For memory allocation and management errors
 */
class MemoryErrorCategory : public ExtendedErrorCategory {
public:
    const char* name() const noexcept override {
        return "fem.memory";
    }

    std::string domain() const noexcept override {
        return "memory";
    }

    std::string message(int code) const override;
    int severity(int code) const noexcept override;
};

// Global category accessors
const NumericErrorCategory& numeric_category() noexcept;
const IOErrorCategory& io_category() noexcept;
const MemoryErrorCategory& memory_category() noexcept;

} // namespace fem::core::error

#endif // CORE_ERROR_ERROR_CATEGORY_H