#pragma once

#ifndef CORE_ERROR_ERROR_MESSAGE_H
#define CORE_ERROR_ERROR_MESSAGE_H

#include <string>
#include <format>
#include <vector>
#include <map>
#include <optional>
#include <chrono>
#include <sstream>
#include <iomanip>
#include "error_code.h"

namespace fem::core::error {

/**
 * @brief Error message formatting and templating utilities
 *
 * Provides consistent formatting for error messages with support for:
 * - Template-based messages with placeholders
 * - Structured error details
 * - Localization support (future)
 * - Message severity levels
 */
class ErrorMessage {
public:
    enum class Severity {
        Debug,
        Info,
        Warning,
        Error,
        Critical,
        Fatal
    };

    /**
     * @brief Message template with named placeholders
     */
    class Template {
    public:
        explicit Template(const std::string& pattern)
            : pattern_(pattern) {
            parse_placeholders();
        }

        /**
         * @brief Format message with provided values
         */
        std::string format(const std::map<std::string, std::string>& values) const {
            std::string result = pattern_;
            
            for (const auto& [key, value] : values) {
                std::string placeholder = "{" + key + "}";
                size_t pos = 0;
                while ((pos = result.find(placeholder, pos)) != std::string::npos) {
                    result.replace(pos, placeholder.length(), value);
                    pos += value.length();
                }
            }
            
            return result;
        }

        /**
         * @brief Get required placeholder names
         */
        const std::vector<std::string>& placeholders() const {
            return placeholders_;
        }

        /**
         * @brief Check if all required placeholders are provided
         */
        bool validate(const std::map<std::string, std::string>& values) const {
            for (const auto& placeholder : placeholders_) {
                if (values.find(placeholder) == values.end()) {
                    return false;
                }
            }
            return true;
        }

    private:
        void parse_placeholders() {
            size_t pos = 0;
            while ((pos = pattern_.find('{', pos)) != std::string::npos) {
                size_t end = pattern_.find('}', pos);
                if (end != std::string::npos) {
                    std::string placeholder = pattern_.substr(pos + 1, end - pos - 1);
                    if (!placeholder.empty() && 
                        std::find(placeholders_.begin(), placeholders_.end(), placeholder) == placeholders_.end()) {
                        placeholders_.push_back(placeholder);
                    }
                    pos = end + 1;
                } else {
                    break;
                }
            }
        }

        std::string pattern_;
        std::vector<std::string> placeholders_;
    };

    /**
     * @brief Structured error message builder
     */
    class Builder {
    public:
        Builder() = default;

        Builder& set_code(ErrorCode code) {
            code_ = code;
            return *this;
        }

        Builder& set_severity(Severity severity) {
            severity_ = severity;
            return *this;
        }

        Builder& set_message(const std::string& message) {
            message_ = message;
            return *this;
        }

        Builder& set_details(const std::string& details) {
            details_ = details;
            return *this;
        }

        Builder& add_context(const std::string& key, const std::string& value) {
            context_[key] = value;
            return *this;
        }

        Builder& set_suggestion(const std::string& suggestion) {
            suggestion_ = suggestion;
            return *this;
        }

        Builder& set_timestamp(std::chrono::system_clock::time_point timestamp) {
            timestamp_ = timestamp;
            return *this;
        }

        /**
         * @brief Build formatted error message
         */
        std::string build() const {
            std::ostringstream oss;
            
            // Header with severity and code
            oss << "[" << severity_to_string(severity_) << "]";
            if (code_ != ErrorCode::Success) {
                oss << " [" << error_code_to_string(code_) << "]";
            }
            
            // Timestamp
            if (timestamp_) {
                auto time_t = std::chrono::system_clock::to_time_t(*timestamp_);
                oss << " [" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << "]";
            }
            
            oss << ": " << message_;
            
            // Details
            if (!details_.empty()) {
                oss << "\n  Details: " << details_;
            }
            
            // Context
            if (!context_.empty()) {
                oss << "\n  Context:";
                for (const auto& [key, value] : context_) {
                    oss << "\n    " << key << ": " << value;
                }
            }
            
            // Suggestion
            if (!suggestion_.empty()) {
                oss << "\n  Suggestion: " << suggestion_;
            }
            
            return oss.str();
        }

        /**
         * @brief Build structured data
         */
        std::map<std::string, std::string> build_structured() const {
            std::map<std::string, std::string> result;
            
            result["severity"] = severity_to_string(severity_);
            result["code"] = error_code_to_string(code_);
            result["message"] = message_;
            
            if (!details_.empty()) {
                result["details"] = details_;
            }
            
            if (!suggestion_.empty()) {
                result["suggestion"] = suggestion_;
            }
            
            if (timestamp_) {
                auto time_t = std::chrono::system_clock::to_time_t(*timestamp_);
                std::ostringstream oss;
                oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
                result["timestamp"] = oss.str();
            }
            
            // Add context with prefixed keys
            for (const auto& [key, value] : context_) {
                result["context." + key] = value;
            }
            
            return result;
        }

    private:
        static std::string severity_to_string(Severity severity) {
            switch (severity) {
                case Severity::Debug: return "DEBUG";
                case Severity::Info: return "INFO";
                case Severity::Warning: return "WARNING";
                case Severity::Error: return "ERROR";
                case Severity::Critical: return "CRITICAL";
                case Severity::Fatal: return "FATAL";
                default: return "UNKNOWN";
            }
        }

        static std::string error_code_to_string(ErrorCode code) {
            // This would ideally use ErrorInfo::get_name
            return std::format("E{:04d}", static_cast<int>(code));
        }

        ErrorCode code_ = ErrorCode::Success;
        Severity severity_ = Severity::Error;
        std::string message_;
        std::string details_;
        std::map<std::string, std::string> context_;
        std::string suggestion_;
        std::optional<std::chrono::system_clock::time_point> timestamp_;
    };

    /**
     * @brief Message formatter with common patterns
     */
    class Formatter {
    public:
        /**
         * @brief Format file operation error
         */
        static std::string file_error(const std::string& operation,
                                     const std::string& path,
                                     const std::string& reason) {
            return std::format("Failed to {} file '{}': {}",
                             operation, path, reason);
        }

        /**
         * @brief Format validation error
         */
        static std::string validation_error(const std::string& field,
                                          const std::string& constraint,
                                          const std::string& actual_value) {
            return std::format("Validation failed for '{}': {} (got: {})",
                             field, constraint, actual_value);
        }

        /**
         * @brief Format range error
         */
        template<typename T>
        static std::string range_error(const std::string& parameter,
                                      T value, T min, T max) {
            return std::format("{} out of range: {} not in [{}, {}]",
                             parameter, value, min, max);
        }

        /**
         * @brief Format null pointer error
         */
        static std::string null_pointer_error(const std::string& pointer_name) {
            return std::format("Null pointer: '{}'", pointer_name);
        }

        /**
         * @brief Format type mismatch error
         */
        static std::string type_mismatch_error(const std::string& context,
                                              const std::string& expected_type,
                                              const std::string& actual_type) {
            return std::format("Type mismatch in {}: expected '{}', got '{}'",
                             context, expected_type, actual_type);
        }

        /**
         * @brief Format timeout error
         */
        static std::string timeout_error(const std::string& operation,
                                        std::chrono::milliseconds timeout) {
            return std::format("Operation '{}' timed out after {}ms",
                             operation, timeout.count());
        }

        /**
         * @brief Format resource error
         */
        static std::string resource_error(const std::string& resource_type,
                                         const std::string& operation,
                                         const std::string& reason) {
            return std::format("{} {} failed: {}",
                             resource_type, operation, reason);
        }

        /**
         * @brief Format configuration error
         */
        static std::string config_error(const std::string& parameter,
                                       const std::string& value,
                                       const std::string& reason) {
            return std::format("Configuration error for '{}' = '{}': {}",
                             parameter, value, reason);
        }

        /**
         * @brief Format numerical error
         */
        static std::string numerical_error(const std::string& operation,
                                          const std::string& issue) {
            return std::format("Numerical error in {}: {}",
                             operation, issue);
        }

        /**
         * @brief Format dependency error
         */
        static std::string dependency_error(const std::string& component,
                                           const std::string& dependency,
                                           const std::string& reason) {
            return std::format("'{}' dependency on '{}' failed: {}",
                             component, dependency, reason);
        }
    };

    /**
     * @brief Message catalog for localization support
     */
    class Catalog {
    public:
        /**
         * @brief Register a message template
         */
        void register_template(const std::string& key,
                              const std::string& pattern,
                              const std::string& language = "en") {
            templates_[language][key] = Template(pattern);
        }

        /**
         * @brief Get message template
         */
        std::optional<Template> get_template(const std::string& key,
                                            const std::string& language = "en") const {
            auto lang_it = templates_.find(language);
            if (lang_it == templates_.end()) {
                return std::nullopt;
            }
            
            auto template_it = lang_it->second.find(key);
            if (template_it == lang_it->second.end()) {
                return std::nullopt;
            }
            
            return template_it->second;
        }

        /**
         * @brief Format message using template
         */
        std::string format(const std::string& key,
                         const std::map<std::string, std::string>& values,
                         const std::string& language = "en") const {
            auto tmpl = get_template(key, language);
            if (!tmpl) {
                // Fallback to key if template not found
                return key;
            }
            
            return tmpl->format(values);
        }

        /**
         * @brief Load catalog from configuration
         */
        void load_from_config(const std::map<std::string, std::map<std::string, std::string>>& config) {
            for (const auto& [key, translations] : config) {
                for (const auto& [language, pattern] : translations) {
                    register_template(key, pattern, language);
                }
            }
        }

    private:
        std::map<std::string, std::map<std::string, Template>> templates_;
    };

    /**
     * @brief Global message catalog instance
     */
    static Catalog& catalog() {
        static Catalog instance;
        return instance;
    }
};

/**
 * @brief Helper functions for common error messages
 */

/**
 * @brief Create a formatted error message
 */
inline std::string format_error(ErrorCode code,
                               const std::string& message,
                               const std::map<std::string, std::string>& context = {}) {
    ErrorMessage::Builder builder;
    builder.set_code(code)
           .set_message(message)
           .set_timestamp(std::chrono::system_clock::now());
    
    for (const auto& [key, value] : context) {
        builder.add_context(key, value);
    }
    
    return builder.build();
}

/**
 * @brief Create a detailed error message
 */
inline std::string detailed_error(ErrorCode code,
                                 const std::string& message,
                                 const std::string& details,
                                 const std::string& suggestion = "") {
    ErrorMessage::Builder builder;
    builder.set_code(code)
           .set_message(message)
           .set_details(details)
           .set_timestamp(std::chrono::system_clock::now());
    
    if (!suggestion.empty()) {
        builder.set_suggestion(suggestion);
    }
    
    return builder.build();
}

} // namespace fem::core::error

#endif // CORE_ERROR_ERROR_MESSAGE_H