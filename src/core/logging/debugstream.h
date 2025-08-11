#pragma once

#ifndef LOGGING_DEBUGSTREAM_H
#define LOGGING_DEBUGSTREAM_H

#include <iostream>
#include <sstream>
#include <iomanip>

#include "logger.h"

namespace fem::core::logging {

/**
 * @brief Stream-based debug output that compiles away in release builds
 *
 * Provides std::cout-like interface for debug logging that has zero
 * overhead in release builds. Integrates with the logging system.
 *
 * Usage context:
 * - Quick debugging without format strings
 * - Temporary debug output during development
 * - Matrix/vector printing with formatting
 * - Performance-critical code where even checking log levels is too expensive
 *
 * Example:
 * ```cpp
 * fem::dbg() << "Matrix A:\n" << matrix << "\n";
 * fem::dbg() << "Iteration " << i << ": residual = " << residual << "\n";
 * ```
 */
    class DebugStream {
    public:
        explicit DebugStream(std::shared_ptr<Logger> logger,
                             LogLevel level = LogLevel::DEBUG)
                : logger_(std::move(logger)), level_(level) {}

        ~DebugStream() {
            if (logger_ && logger_->should_log(level_)) {
                logger_->log(level_, "{}", buffer_.str());
            }
        }

        // Output operators
        template<typename T>
        DebugStream& operator<<(const T& value) {
            buffer_ << value;
            return *this;
        }

        // Special handling for manipulators
        DebugStream& operator<<(std::ostream& (*manip)(std::ostream&)) {
            buffer_ << manip;
            return *this;
        }

        // Special handling for std::endl
        DebugStream& operator<<(std::ostream& (*manip)(std::ostream&)) {
            buffer_ << manip;
            return *this;
        }

    private:
        std::shared_ptr<Logger> logger_;
        LogLevel level_;
        std::ostringstream buffer_;
    };

/**
 * @brief Null stream that discards all output
 */
    class NullStream {
    public:
        template<typename T>
        NullStream& operator<<(const T&) { return *this; }

        NullStream& operator<<(std::ostream& (*)(std::ostream&)) { return *this; }
    };

// Global debug stream factory functions

#ifdef NDEBUG
    // In release builds, return null stream
    inline NullStream dbg() { return NullStream{}; }
    inline NullStream dbg(const std::string&) { return NullStream{}; }
#else
    // In debug builds, return real debug stream
    inline DebugStream dbg() {
        return DebugStream(get_logger("fem.debug"));
    }

    inline DebugStream dbg(const std::string& logger_name) {
        return DebugStream(get_logger(logger_name));
    }
#endif

/**
 * @brief Conditional debug stream
 */
    template<bool Condition>
    class ConditionalStream {
    public:
        explicit ConditionalStream(std::shared_ptr<Logger> logger,
                                   LogLevel level = LogLevel::DEBUG)
                : logger_(std::move(logger)), level_(level) {}

        ~ConditionalStream() {
            if constexpr (Condition) {
                if (logger_ && logger_->should_log(level_)) {
                    logger_->log(level_, "{}", buffer_.str());
                }
            }
        }

        template<typename T>
        ConditionalStream& operator<<(const T& value) {
            if constexpr (Condition) {
                buffer_ << value;
            }
            return *this;
        }

    private:
        std::shared_ptr<Logger> logger_;
        LogLevel level_;
        std::ostringstream buffer_;
    };

/**
 * @brief Matrix/Vector debug printer with formatting
 */
    class MatrixDebugPrinter {
    public:
        explicit MatrixDebugPrinter(std::shared_ptr<Logger> logger)
                : logger_(std::move(logger)) {}

        /**
         * @brief Print a matrix with formatting
         */
        template<typename Matrix>
        void print_matrix(const Matrix& mat, const std::string& name = "Matrix") {
            if (!logger_->is_debug_enabled()) return;

            std::ostringstream oss;
            oss << name << " [" << mat.rows() << "x" << mat.cols() << "]:\n";

            // Configure precision
            oss << std::fixed << std::setprecision(precision_);

            for (size_t i = 0; i < mat.rows(); ++i) {
                oss << "  [";
                for (size_t j = 0; j < mat.cols(); ++j) {
                    if (j > 0) oss << ", ";
                    oss << std::setw(width_) << mat(i, j);
                }
                oss << "]\n";
            }

            logger_->debug("{}", oss.str());
        }

        /**
         * @brief Print a vector with formatting
         */
        template<typename Vector>
        void print_vector(const Vector& vec, const std::string& name = "Vector") {
            if (!logger_->is_debug_enabled()) return;

            std::ostringstream oss;
            oss << name << " [" << vec.size() << "]:\n  [";

            oss << std::fixed << std::setprecision(precision_);

            for (size_t i = 0; i < vec.size(); ++i) {
                if (i > 0) oss << ", ";
                if (i > 0 && i % elements_per_line_ == 0) {
                    oss << "\n   ";
                }
                oss << std::setw(width_) << vec(i);
            }
            oss << "]\n";

            logger_->debug("{}", oss.str());
        }

        /**
         * @brief Print sparse matrix pattern
         */
        template<typename SparseMatrix>
        void print_sparsity_pattern(const SparseMatrix& mat,
                                    const std::string& name = "Sparsity") {
            if (!logger_->is_debug_enabled()) return;

            std::ostringstream oss;
            oss << name << " pattern [" << mat.rows() << "x" << mat.cols()
                << ", nnz=" << mat.nonZeros() << "]:\n";

            // Print pattern using characters
            for (size_t i = 0; i < std::min(mat.rows(), max_display_size_); ++i) {
                oss << "  ";
                for (size_t j = 0; j < std::min(mat.cols(), max_display_size_); ++j) {
                    oss << (mat.coeff(i, j) != 0 ? '*' : '.');
                }
                if (mat.cols() > max_display_size_) oss << "...";
                oss << '\n';
            }
            if (mat.rows() > max_display_size_) {
                oss << "  ...\n";
            }

            logger_->debug("{}", oss.str());
        }

        // Configuration
        void set_precision(int precision) { precision_ = precision; }
        void set_width(int width) { width_ = width; }
        void set_elements_per_line(int n) { elements_per_line_ = n; }
        void set_max_display_size(size_t size) { max_display_size_ = size; }

    private:
        std::shared_ptr<Logger> logger_;
        int precision_{4};
        int width_{10};
        int elements_per_line_{8};
        size_t max_display_size_{10};
    };

/**
 * @brief Progress indicator for long operations
 */
    class DebugProgressIndicator {
    public:
        DebugProgressIndicator(std::shared_ptr<Logger> logger,
                               const std::string& operation,
                               size_t total_steps)
                : logger_(std::move(logger))
                , operation_(operation)
                , total_steps_(total_steps)
                , start_time_(std::chrono::steady_clock::now()) {

            if (logger_ && logger_->is_debug_enabled()) {
                logger_->debug("Starting {}: {} steps", operation_, total_steps_);
            }
        }

        void update(size_t current_step) {
            if (!logger_ || !logger_->is_debug_enabled()) return;

            // Only log at certain intervals
            if (current_step % report_interval_ != 0 &&
                current_step != total_steps_ - 1) {
                return;
            }

            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    now - start_time_).count();

            double progress = static_cast<double>(current_step + 1) / total_steps_;
            double rate = elapsed > 0 ? (current_step + 1) / static_cast<double>(elapsed) : 0;

            logger_->debug("{}: {:.1f}% ({}/{}) - {:.1f} steps/sec",
                           operation_, progress * 100,
                           current_step + 1, total_steps_, rate);
        }

        void set_report_interval(size_t interval) { report_interval_ = interval; }

    private:
        std::shared_ptr<Logger> logger_;
        std::string operation_;
        size_t total_steps_;
        size_t report_interval_{100};
        std::chrono::steady_clock::time_point start_time_;
    };

// Convenience macros for debug output

/**
 * @brief Debug print with automatic variable name
 */
#define FEM_DBG_VAR(var) \
    fem::core::logging::dbg() << #var << " = " << (var) << "\n"

/**
 * @brief Debug print multiple variables
 */
#define FEM_DBG_VARS(...) \
    do { \
        auto& _dbg = fem::core::logging::dbg(); \
        ((_dbg << #__VA_ARGS__ << " = " << (__VA_ARGS__) << " "), ...); \
        _dbg << "\n"; \
    } while(0)

/**
 * @brief Conditional debug output
 */
#ifdef NDEBUG
    #define FEM_DBG_IF(condition) if (false)
#else
    #define FEM_DBG_IF(condition) if (condition)
#endif

} // namespace fem::core::logging

#endif //LOGGING_DEBUGSTREAM_H
