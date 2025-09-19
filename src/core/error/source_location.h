#pragma once

#ifndef CORE_ERROR_SOURCE_LOCATION_H
#define CORE_ERROR_SOURCE_LOCATION_H

#include <source_location>
#include <string>
#include <format>
#include <vector>
#include <optional>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>

namespace fem::core::error {

/**
 * @brief Enhanced source location tracking
 *
 * Extends std::source_location with additional utilities for:
 * - Path manipulation and normalization
 * - Relative path calculation
 * - Source context extraction
 * - Location formatting
 */
class SourceLocation {
public:
    /**
     * @brief Construct from std::source_location
     */
    explicit SourceLocation(const std::source_location& loc = std::source_location::current())
        : location_(loc) {
    }

    /**
     * @brief Construct with explicit values
     */
    SourceLocation(const char* file, int line, const char* function, int column = 0)
        : has_explicit_values_(true)
        , file_name_(file)
        , function_name_(function)
        , line_(line)
        , column_(column) {
    }

    // Accessors matching std::source_location interface
    const char* file_name() const noexcept {
        return has_explicit_values_ ? file_name_ : location_.file_name();
    }

    const char* function_name() const noexcept {
        return has_explicit_values_ ? function_name_ : location_.function_name();
    }

    std::uint_least32_t line() const noexcept {
        return has_explicit_values_ ? line_ : location_.line();
    }

    std::uint_least32_t column() const noexcept {
        return has_explicit_values_ ? column_ : location_.column();
    }

    /**
     * @brief Get filename only (without path)
     */
    std::string filename() const {
        std::filesystem::path path(file_name());
        return path.filename().string();
    }

    /**
     * @brief Get directory path
     */
    std::string directory() const {
        std::filesystem::path path(file_name());
        return path.parent_path().string();
    }

    /**
     * @brief Get relative path from a base directory
     */
    std::string relative_path(const std::string& base_dir) const {
        try {
            std::filesystem::path file_path(file_name());
            std::filesystem::path base_path(base_dir);

            // Check if paths are valid before attempting relative
            if (!file_path.is_absolute() || !base_path.is_absolute()) {
                return file_name();
            }

            return std::filesystem::relative(file_path, base_path).string();
        } catch (...) {
            return file_name();
        }
    }

    /**
     * @brief Get simplified function name (without template parameters)
     */
    std::string simple_function_name() const {
        std::string full_name = function_name();
        
        // Find template bracket
        auto pos = full_name.find('<');
        if (pos != std::string::npos) {
            return full_name.substr(0, pos);
        }
        
        // Find parameter list
        pos = full_name.find('(');
        if (pos != std::string::npos) {
            return full_name.substr(0, pos);
        }
        
        return full_name;
    }

    /**
     * @brief Get namespace from function name
     */
    std::string namespace_name() const {
        std::string full_name = function_name();
        
        // Find last :: before any template or parameter
        auto template_pos = full_name.find('<');
        auto param_pos = full_name.find('(');
        auto search_end = std::min(template_pos, param_pos);
        
        auto pos = full_name.rfind("::", search_end);
        if (pos != std::string::npos && pos > 0) {
            return full_name.substr(0, pos);
        }
        
        return "";
    }

    /**
     * @brief Format as string with various styles
     */
    enum class Format {
        Short,      // file:line
        Medium,     // file:line:function
        Long,       // file:line:column:function
        Full        // /full/path/file:line:column:function
    };

    std::string to_string(Format format = Format::Medium) const {
        switch (format) {
            case Format::Short:
                return std::format("{}:{}", filename(), line());
            
            case Format::Medium:
                return std::format("{}:{}:{}", 
                                 filename(), line(), simple_function_name());
            
            case Format::Long:
                if (column() > 0) {
                    return std::format("{}:{}:{}:{}",
                                     filename(), line(), column(), 
                                     simple_function_name());
                } else {
                    return std::format("{}:{}:{}",
                                     filename(), line(), 
                                     simple_function_name());
                }
            
            case Format::Full:
                if (column() > 0) {
                    return std::format("{}:{}:{}:{}",
                                     file_name(), line(), column(),
                                     function_name());
                } else {
                    return std::format("{}:{}:{}",
                                     file_name(), line(),
                                     function_name());
                }
            
            default:
                return to_string(Format::Medium);
        }
    }

    /**
     * @brief Format for logging
     */
    std::string log_format() const {
        return std::format("[{}:{}] in {}",
                         filename(), line(), simple_function_name());
    }

    /**
     * @brief Format for debugging
     */
    std::string debug_format() const {
        return std::format("File: {}\nLine: {}\nColumn: {}\nFunction: {}\nNamespace: {}",
                         file_name(), line(), column(), 
                         function_name(), namespace_name());
    }

    /**
     * @brief Compare locations
     */
    bool operator==(const SourceLocation& other) const {
        return std::string(file_name()) == std::string(other.file_name()) &&
               line() == other.line() &&
               column() == other.column();
    }

    bool operator!=(const SourceLocation& other) const {
        return !(*this == other);
    }

    /**
     * @brief Check if location is valid
     */
    bool is_valid() const {
        return file_name() != nullptr && *file_name() != '\0' && line() > 0;
    }

private:
    std::source_location location_;
    
    // For explicit construction
    bool has_explicit_values_ = false;
    const char* file_name_ = "";
    const char* function_name_ = "";
    std::uint_least32_t line_ = 0;
    std::uint_least32_t column_ = 0;
};

/**
 * @brief Source location with context
 */
class SourceContext {
public:
    /**
     * @brief Construct with location and context lines
     */
    SourceContext(const SourceLocation& location,
                 int context_lines = 2)
        : location_(location)
        , context_lines_(context_lines) {
        if (location.is_valid()) {
            load_context();
        }
    }

    /**
     * @brief Get the source location
     */
    const SourceLocation& location() const { return location_; }

    /**
     * @brief Get lines before the error
     */
    const std::vector<std::string>& lines_before() const { return lines_before_; }

    /**
     * @brief Get the error line
     */
    const std::string& error_line() const { return error_line_; }

    /**
     * @brief Get lines after the error
     */
    const std::vector<std::string>& lines_after() const { return lines_after_; }

    /**
     * @brief Format context for display
     */
    std::string format(bool show_line_numbers = true,
                      bool highlight_error = true) const {
        if (error_line_.empty()) {
            return location_.to_string();
        }

        std::ostringstream oss;
        oss << "Source: " << location_.to_string() << "\n";
        
        int start_line = static_cast<int>(location_.line() - lines_before_.size());
        
        // Lines before
        for (size_t i = 0; i < lines_before_.size(); ++i) {
            if (show_line_numbers) {
                oss << std::format("{:4d} | ", start_line + static_cast<int>(i));
            }
            oss << lines_before_[i] << "\n";
        }
        
        // Error line
        if (show_line_numbers) {
            if (highlight_error) {
                oss << std::format(">{:3d} | ", location_.line());
            } else {
                oss << std::format("{:4d} | ", location_.line());
            }
        }
        oss << error_line_;
        
        // Add column indicator if available
        if (highlight_error && location_.column() > 0) {
            oss << "\n";
            if (show_line_numbers) {
                oss << "      | ";
            }
            oss << std::string(location_.column() - 1, ' ') << "^";
        }
        oss << "\n";
        
        // Lines after
        for (size_t i = 0; i < lines_after_.size(); ++i) {
            if (show_line_numbers) {
                oss << std::format("{:4d} | ", location_.line() + 1 + static_cast<std::uint_least32_t>(i));
            }
            oss << lines_after_[i] << "\n";
        }
        
        return oss.str();
    }

private:
    void load_context() {
        // Try to read the source file
        const char* fname = location_.file_name();
        if (!fname || fname[0] == '\0') {
            return;
        }
        std::ifstream file(fname);
        if (!file) {
            return;
        }
        
        std::vector<std::string> all_lines;
        std::string line;
        while (std::getline(file, line)) {
            all_lines.push_back(line);
        }
        
        if (all_lines.empty() || location_.line() == 0) {
            return;
        }
        
        // Get the error line (1-indexed)
        size_t error_index = location_.line() - 1;
        if (error_index >= all_lines.size()) {
            return;
        }
        
        error_line_ = all_lines[error_index];
        
        // Get lines before
        int start = std::max(0, static_cast<int>(error_index) - context_lines_);
        for (int i = start; i < static_cast<int>(error_index); ++i) {
            lines_before_.push_back(all_lines[i]);
        }
        
        // Get lines after
        int end = std::min(static_cast<int>(all_lines.size()),
                          static_cast<int>(error_index) + context_lines_ + 1);
        for (int i = static_cast<int>(error_index) + 1; i < end; ++i) {
            lines_after_.push_back(all_lines[i]);
        }
    }

    SourceLocation location_;
    int context_lines_;
    std::vector<std::string> lines_before_;
    std::string error_line_;
    std::vector<std::string> lines_after_;
};

/**
 * @brief Call site tracker for debugging
 */
class CallSite {
public:
    explicit CallSite(const std::source_location& loc = std::source_location::current())
        : location_(loc)
        , timestamp_(std::chrono::steady_clock::now()) {
    }

    const SourceLocation& location() const { return location_; }
    
    std::chrono::steady_clock::time_point timestamp() const { return timestamp_; }
    
    /**
     * @brief Get elapsed time since call site was created
     */
    std::chrono::milliseconds elapsed() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now - timestamp_);
    }

    /**
     * @brief Format call site information
     */
    std::string format() const {
        return std::format("{}@{}ms",
                         location_.to_string(SourceLocation::Format::Short),
                         elapsed().count());
    }

private:
    SourceLocation location_;
    std::chrono::steady_clock::time_point timestamp_;
};

/**
 * @brief Macro helpers for source location
 */

// Get current source location
#define FEM_CURRENT_LOCATION() \
    fem::core::error::SourceLocation(std::source_location::current())

// Get current source context
#define FEM_SOURCE_CONTEXT(lines) \
    fem::core::error::SourceContext(FEM_CURRENT_LOCATION(), lines)

// Track call site
#define FEM_CALL_SITE() \
    fem::core::error::CallSite(std::source_location::current())

} // namespace fem::core::error

#endif // CORE_ERROR_SOURCE_LOCATION_H