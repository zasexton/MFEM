#pragma once

#ifndef LOGGING_SCOPEDLOGGER_H
#define LOGGING_SCOPEDLOGGER_H

#include <chrono>
#include <memory>
#include <sstream>
#include <vector>
#include <utility>
#include <exception>
#include <atomic>

#include "logger.h"
#include "logcontext.h"

#include "../base/object.h"
#include "../base/component.h"
#include "../base/observer.h"
#include "../base/interface.h"
#include "../base/visitor.h"
#include "../base/policy.h"
#include "../base/singleton.h"

namespace fem::core::logging {

// Forward declarations
    class ScopeEvent;
    class ScopeStatistics;

/**
 * @brief Interface for scope timing and tracing
 */
    class IScoped : public base::TypedInterface<IScoped> {
    public:
        virtual void mark_failed(const std::string& reason = "") = 0;
        virtual void mark_success(const std::string& message = "") = 0;
        virtual void checkpoint(const std::string& label) = 0;
        virtual int64_t elapsed_ms() const = 0;
        virtual int64_t elapsed_us() const = 0;
        virtual const std::string& get_scope_name() const = 0;
    };

/**
 * @brief Event for scope lifecycle notifications
 */
    class ScopeEvent : public base::TypedEvent<ScopeEvent> {
    public:
        enum Type {
            SCOPE_ENTERED,
            SCOPE_EXITED,
            SCOPE_CHECKPOINT,
            SCOPE_FAILED,
            SCOPE_SLOW_DETECTED
        };

        ScopeEvent(Type type, std::string scope_name, int64_t duration_us = 0)
                : TypedEvent("ScopeEvent")
                , type_(type)
                , scope_name_(std::move(scope_name))
                , duration_us_(duration_us) {}

        Type get_event_type() const { return type_; }
        const std::string& get_scope_name() const { return scope_name_; }
        int64_t get_duration_us() const { return duration_us_; }

        void set_checkpoint_label(const std::string& label) { checkpoint_label_ = label; }
        const std::string& get_checkpoint_label() const { return checkpoint_label_; }

        void set_failure_reason(const std::string& reason) { failure_reason_ = reason; }
        const std::string& get_failure_reason() const { return failure_reason_; }

    private:
        Type type_;
        std::string scope_name_;
        int64_t duration_us_;
        std::string checkpoint_label_;
        std::string failure_reason_;
    };

/**
 * @brief Statistics collector for scope performance
 */
    class ScopeStatistics : public base::Object,
                            public base::Singleton<ScopeStatistics>,
                            public base::Subject<ScopeEvent> {
        friend class base::Singleton<ScopeStatistics>;

    public:
        struct ScopeStats {
            std::string name;
            uint64_t call_count{0};
            uint64_t total_duration_us{0};
            uint64_t min_duration_us{UINT64_MAX};
            uint64_t max_duration_us{0};
            uint64_t failure_count{0};
            std::chrono::system_clock::time_point last_called;

            double average_duration_us() const {
                return call_count > 0 ? static_cast<double>(total_duration_us) / call_count : 0.0;
            }
        };

        void record_scope_entry(const std::string& scope_name) {
            std::lock_guard lock(stats_mutex_);
            active_scopes_[scope_name]++;
        }

        void record_scope_exit(const std::string& scope_name, int64_t duration_us, bool success) {
            std::lock_guard lock(stats_mutex_);

            auto& stats = scope_stats_[scope_name];
            stats.name = scope_name;
            stats.call_count++;
            stats.total_duration_us += duration_us;
            stats.min_duration_us = std::min(stats.min_duration_us, static_cast<uint64_t>(duration_us));
            stats.max_duration_us = std::max(stats.max_duration_us, static_cast<uint64_t>(duration_us));
            stats.last_called = std::chrono::system_clock::now();

            if (!success) {
                stats.failure_count++;
            }

            if (active_scopes_[scope_name] > 0) {
                active_scopes_[scope_name]--;
            }
        }

        std::vector<ScopeStats> get_all_stats() const {
            std::lock_guard lock(stats_mutex_);
            std::vector<ScopeStats> result;
            result.reserve(scope_stats_.size());

            for (const auto& [name, stats] : scope_stats_) {
                result.push_back(stats);
            }

            return result;
        }

        ScopeStats get_stats(const std::string& scope_name) const {
            std::lock_guard lock(stats_mutex_);
            auto it = scope_stats_.find(scope_name);
            return it != scope_stats_.end() ? it->second : ScopeStats{};
        }

        void clear_stats() {
            std::lock_guard lock(stats_mutex_);
            scope_stats_.clear();
            active_scopes_.clear();
        }

        int get_active_scope_count(const std::string& scope_name) const {
            std::lock_guard lock(stats_mutex_);
            auto it = active_scopes_.find(scope_name);
            return it != active_scopes_.end() ? it->second : 0;
        }

    private:
        ScopeStatistics() : base::Object("ScopeStatistics") {}

        mutable std::mutex stats_mutex_;
        std::unordered_map<std::string, ScopeStats> scope_stats_;
        std::unordered_map<std::string, int> active_scopes_;
    };

/**
 * @brief Component for adding scope timing to any Entity
 */
    class ScopeTimingComponent : public base::TypedComponent<ScopeTimingComponent> {
    public:
        struct Config {
            int64_t slow_threshold_ms = 1000;
            bool enable_statistics = true;
            bool enable_events = true;
            LogLevel default_level = LogLevel::TRACE;
        };

        explicit ScopeTimingComponent(const Config& config = {})
                : base::TypedComponent("ScopeTiming")
                , config_(config) {}

        void start_scope(const std::string& name) {
            std::lock_guard lock(scope_mutex_);
            scope_stack_.emplace_back(name, std::chrono::steady_clock::now());

            if (config_.enable_statistics) {
                ScopeStatistics::instance().record_scope_entry(name);
            }

            if (config_.enable_events) {
                base::emit_event<ScopeEvent>(ScopeEvent::SCOPE_ENTERED, name);
            }
        }

        void end_scope(const std::string& name, bool success = true) {
            std::lock_guard lock(scope_mutex_);

            if (scope_stack_.empty() || scope_stack_.back().first != name) {
                // Scope mismatch - log warning but continue
                return;
            }

            auto start_time = scope_stack_.back().second;
            scope_stack_.pop_back();

            auto duration = std::chrono::steady_clock::now() - start_time;
            auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

            if (config_.enable_statistics) {
                ScopeStatistics::instance().record_scope_exit(name, duration_us, success);
            }

            if (config_.enable_events) {
                auto event_type = success ? ScopeEvent::SCOPE_EXITED : ScopeEvent::SCOPE_FAILED;
                base::emit_event<ScopeEvent>(event_type, name, duration_us);

                if (duration_ms > config_.slow_threshold_ms) {
                    base::emit_event<ScopeEvent>(ScopeEvent::SCOPE_SLOW_DETECTED, name, duration_us);
                }
            }
        }

        void add_checkpoint(const std::string& scope_name, const std::string& label) {
            if (config_.enable_events) {
                ScopeEvent event(ScopeEvent::SCOPE_CHECKPOINT, scope_name);
                event.set_checkpoint_label(label);
                base::emit_event<ScopeEvent>(event);
            }
        }

        void update(double dt) override {
            // Could perform periodic cleanup or statistics updates
        }

        void reset() override {
            std::lock_guard lock(scope_mutex_);
            scope_stack_.clear();
        }

        const Config& get_config() const { return config_; }
        void set_config(const Config& config) { config_ = config; }

    private:
        Config config_;
        std::mutex scope_mutex_;
        std::vector<std::pair<std::string, std::chrono::steady_clock::time_point>> scope_stack_;
    };

/**
 * @brief RAII logger using base infrastructure
 */
    class ScopedLogger : public base::Object,
                         public IScoped,
                         public base::Subject<ScopeEvent>,
                         public base::NonCopyableNonMovable<ScopedLogger> {
    public:
        /**
         * @brief Constructor with automatic function name detection
         */
        ScopedLogger(object_ptr<Logger> logger,
                     std::string scope_name = "",
                     LogLevel level = LogLevel::TRACE,
                     std::source_location loc = std::source_location::current())
                : base::Object("ScopedLogger")
                , logger_(std::move(logger))
                , scope_name_(scope_name.empty() ? extract_function_name(loc.function_name()) : std::move(scope_name))
                , level_(level)
                , location_(loc)
                , start_time_(std::chrono::steady_clock::now())
                , initial_exception_count_(std::uncaught_exceptions()) {

            initialize();
        }

        /**
         * @brief Constructor with custom entry message
         */
        template<typename... Args>
        ScopedLogger(object_ptr<Logger> logger,
                     std::string scope_name,
                     LogLevel level,
                     std::format_string<Args...> entry_fmt,
                     Args&&... args,
                     std::source_location loc = std::source_location::current())
                : base::Object("ScopedLogger")
                , logger_(std::move(logger))
                , scope_name_(std::move(scope_name))
                , level_(level)
                , location_(loc)
                , start_time_(std::chrono::steady_clock::now())
                , initial_exception_count_(std::uncaught_exceptions())
                , entry_message_(std::format(entry_fmt.get(), std::forward<Args>(args)...)) {

            initialize();
        }

        ~ScopedLogger() override {
            try {
                finalize();
            } catch (...) {
                // Don't throw from destructor
            }
        }

        // IScoped interface implementation
        void mark_failed(const std::string& reason = "") override {
            success_ = false;
            failure_reason_ = reason;

            // Emit failure event
            ScopeEvent event(ScopeEvent::SCOPE_FAILED, scope_name_);
            event.set_failure_reason(reason);
            notify_observers(event);
            base::emit_event<ScopeEvent>(event);
        }

        void mark_success(const std::string& message = "") override {
            success_ = true;
            success_message_ = message;
        }

        void checkpoint(const std::string& label) override {
            auto now = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time_);

            checkpoints_.emplace_back(label, duration.count());

            // Log the checkpoint
            if (logger_ && logger_->should_log(LogLevel::TRACE)) {
                std::string indent = get_indent();
                logger_->log(LogLevel::TRACE, "{}  ✓ {} ({}μs)",
                             indent, label, duration.count());
            }

            // Emit checkpoint event
            ScopeEvent event(ScopeEvent::SCOPE_CHECKPOINT, scope_name_, duration.count());
            event.set_checkpoint_label(label);
            notify_observers(event);
            base::emit_event<ScopeEvent>(event);

            // Update timing component if available
            if (auto* timing = get_timing_component()) {
                timing->add_checkpoint(scope_name_, label);
            }
        }

        int64_t elapsed_ms() const override {
            auto duration = std::chrono::steady_clock::now() - start_time_;
            return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        }

        int64_t elapsed_us() const override {
            auto duration = std::chrono::steady_clock::now() - start_time_;
            return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
        }

        const std::string& get_scope_name() const override {
            return scope_name_;
        }

        // Enhanced logging with automatic indentation
        template<typename... Args>
        void log(LogLevel level, std::format_string<Args...> fmt, Args&&... args) {
            if (logger_) {
                std::string indent = get_indent();
                std::string message = std::format(fmt.get(), std::forward<Args>(args)...);
                logger_->log(level, "{}  {}", indent, message);
            }
        }

        // Convenience logging methods
        template<typename... Args>
        void trace(std::format_string<Args...> fmt, Args&&... args) {
            log(LogLevel::TRACE, fmt, std::forward<Args>(args)...);
        }

        template<typename... Args>
        void debug(std::format_string<Args...> fmt, Args&&... args) {
            log(LogLevel::DEBUG, fmt, std::forward<Args>(args)...);
        }

        template<typename... Args>
        void info(std::format_string<Args...> fmt, Args&&... args) {
            log(LogLevel::INFO, fmt, std::forward<Args>(args)...);
        }

        template<typename... Args>
        void warn(std::format_string<Args...> fmt, Args&&... args) {
            log(LogLevel::WARN, fmt, std::forward<Args>(args)...);
        }

        template<typename... Args>
        void error(std::format_string<Args...> fmt, Args&&... args) {
            log(LogLevel::ERROR, fmt, std::forward<Args>(args)...);
        }

        // Configuration
        void set_slow_threshold(std::chrono::milliseconds threshold) {
            slow_threshold_ms_ = threshold.count();
        }

        // Visitor pattern support for scope analysis
        void accept(base::Visitor<ScopedLogger>& visitor) {
            visitor.visit(*this);
        }

        // Override from Object
        std::string to_string() const override {
            return std::format("ScopedLogger(scope={}, id={}, elapsed={}ms, success={})",
                               scope_name_, id(), elapsed_ms(), success_);
        }

    private:
        object_ptr<Logger> logger_;
        std::string scope_name_;
        LogLevel level_;
        std::source_location location_;
        std::chrono::steady_clock::time_point start_time_;
        int initial_exception_count_;
        bool success_{true};
        std::string failure_reason_;
        std::string success_message_;
        std::string entry_message_;
        int64_t slow_threshold_ms_{1000};

        std::vector<std::pair<std::string, int64_t>> checkpoints_;

        static thread_local std::atomic<int> current_depth_;

        void initialize() {
            increment_depth();

            // Set context using LogContext
            LogContext::set("scope", scope_name_);
            LogContext::set("scope_depth", get_current_depth());
            LogContext::set("scope_id", id());

            // Log entry
            if (logger_ && logger_->should_log(level_)) {
                std::string indent = get_indent();
                std::string msg = entry_message_.empty() ? "" : ": " + entry_message_;
                logger_->log(level_, "{}→ Entering {}{}", indent, scope_name_, msg, location_);
            }

            // Record in statistics
            ScopeStatistics::instance().record_scope_entry(scope_name_);

            // Emit entry event
            ScopeEvent event(ScopeEvent::SCOPE_ENTERED, scope_name_);
            notify_observers(event);
            base::emit_event<ScopeEvent>(event);

            // Start timing component if logger has one
            if (auto* timing = get_timing_component()) {
                timing->start_scope(scope_name_);
            }
        }

        void finalize() {
            auto duration = std::chrono::steady_clock::now() - start_time_;
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
            auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(duration);

            // Check for exceptions
            bool has_exception = std::uncaught_exceptions() > initial_exception_count_;
            bool actual_success = success_ && !has_exception;

            // Log exit
            if (logger_ && logger_->should_log(level_)) {
                std::string indent = get_indent();
                std::string duration_str = format_duration(duration);

                if (has_exception) {
                    logger_->log(LogLevel::ERROR,
                                 "{}← Exiting {} with EXCEPTION after {}",
                                 indent, scope_name_, duration_str);
                } else if (!success_) {
                    std::string reason = failure_reason_.empty() ? "" : " - " + failure_reason_;
                    logger_->log(LogLevel::WARN,
                                 "{}← Exiting {} with FAILURE after {}{}",
                                 indent, scope_name_, duration_str, reason);
                } else {
                    LogLevel exit_level = (duration_ms.count() > slow_threshold_ms_) ? LogLevel::WARN : level_;
                    std::string msg = success_message_.empty() ? "" : " - " + success_message_;
                    logger_->log(exit_level,
                                 "{}← Exiting {} successfully after {}{}",
                                 indent, scope_name_, duration_str, msg);
                }

                // Log checkpoint summary if any
                if (!checkpoints_.empty()) {
                    log_checkpoint_summary();
                }
            }

            // Record in statistics
            ScopeStatistics::instance().record_scope_exit(scope_name_, duration_us.count(), actual_success);

            // Emit exit event
            auto event_type = actual_success ? ScopeEvent::SCOPE_EXITED : ScopeEvent::SCOPE_FAILED;
            ScopeEvent event(event_type, scope_name_, duration_us.count());
            if (!actual_success) {
                event.set_failure_reason(has_exception ? "Exception" : failure_reason_);
            }
            notify_observers(event);
            base::emit_event<ScopeEvent>(event);

            // Check for slow operation
            if (duration_ms.count() > slow_threshold_ms_) {
                ScopeEvent slow_event(ScopeEvent::SCOPE_SLOW_DETECTED, scope_name_, duration_us.count());
                notify_observers(slow_event);
                base::emit_event<ScopeEvent>(slow_event);
            }

            // End timing component if available
            if (auto* timing = get_timing_component()) {
                timing->end_scope(scope_name_, actual_success);
            }

            decrement_depth();
        }

        ScopeTimingComponent* get_timing_component() const {
            if (logger_ && logger_->template has_component<ScopeTimingComponent>()) {
                return logger_->template get_component<ScopeTimingComponent>();
            }
            return nullptr;
        }

        void increment_depth() {
            current_depth_.fetch_add(1, std::memory_order_relaxed);
        }

        void decrement_depth() {
            current_depth_.fetch_sub(1, std::memory_order_relaxed);
        }

        int get_current_depth() const {
            return current_depth_.load(std::memory_order_relaxed);
        }

        std::string get_indent() const {
            int depth = current_depth_.load(std::memory_order_relaxed);
            return std::string(depth * 2, ' ');
        }

        std::string extract_function_name(const char* full_name) const {
            if (!full_name) return "unknown";

            std::string name(full_name);
            auto template_pos = name.find('<');
            if (template_pos != std::string::npos) {
                name = name.substr(0, template_pos);
            }

            auto last_scope = name.find_last_of("::");
            if (last_scope != std::string::npos && last_scope + 1 < name.length()) {
                name = name.substr(last_scope + 1);
            }

            return name;
        }

        std::string format_duration(std::chrono::steady_clock::duration duration) const {
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(duration);
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
            auto s = std::chrono::duration_cast<std::chrono::seconds>(duration);

            if (s.count() >= 1) {
                return std::format("{:.2f}s", s.count() + (ms.count() % 1000) / 1000.0);
            } else if (ms.count() >= 1) {
                return std::format("{}ms", ms.count());
            } else {
                return std::format("{}μs", us.count());
            }
        }

        void log_checkpoint_summary() const {
            std::ostringstream oss;
            oss << "Checkpoints for " << scope_name_ << ":\n";

            int64_t last_time = 0;
            for (const auto& [label, time_us] : checkpoints_) {
                int64_t delta = time_us - last_time;
                oss << "    " << label << ": " << time_us << "μs (+=" << delta << "μs)\n";
                last_time = time_us;
            }

            if (logger_) {
                logger_->debug("{}", oss.str());
            }
        }
    };

// Initialize thread-local depth counter
    thread_local std::atomic<int> ScopedLogger::current_depth_{0};

/**
 * @brief Enhanced timer using base infrastructure
 */
class ScopedTimer : public base::Object,
                    public base::NonCopyableNonMovable<ScopedTimer> {
public:
    ScopedTimer(object_ptr<Logger> logger,
                std::string operation,
                LogLevel level = LogLevel::INFO)
            : base::Object("ScopedTimer")
            , logger_(std::move(logger))
            , operation_(std::move(operation))
            , level_(level)
            , start_time_(std::chrono::steady_clock::now()) {}

    template<typename... Args>
    ScopedTimer(object_ptr<Logger> logger,
                LogLevel level,
                std::format_string<Args...> fmt,
                Args&&... args)
            : base::Object("ScopedTimer")
            , logger_(std::move(logger))
            , operation_(std::format(fmt.get(), std::forward<Args>(args)...))
            , level_(level)
            , start_time_(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() override {
        if (logger_ && logger_->should_log(level_)) {
            auto duration = std::chrono::steady_clock::now() - start_time_;
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();

            if (ms >= 1) {
                logger_->log(level_, "⏱ {} took {}ms", operation_, ms);
            } else {
                logger_->log(level_, "⏱ {} took {}μs", operation_, us);
            }
        }
    }

    int64_t elapsed_ms() const {
        auto duration = std::chrono::steady_clock::now() - start_time_;
        return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    }

    std::string to_string() const override {
        return std::format("ScopedTimer(operation={}, id={}, elapsed={}ms)",
                           operation_, id(), elapsed_ms());
    }

private:
    object_ptr<Logger> logger_;
    std::string operation_;
    LogLevel level_;
    std::chrono::steady_clock::time_point start_time_;
};

/**
 * @brief Scope analysis visitor for performance debugging
 */
class ScopeAnalysisVisitor : public base::Visitor<ScopedLogger> {
public:
    struct Analysis {
        int total_scopes{0};
        int failed_scopes{0};
        int64_t total_duration_ms{0};
        int64_t max_duration_ms{0};
        std::string slowest_scope;
        std::vector<std::string> failed_scope_names;
    };

    void visit_impl(ScopedLogger& scope) override {
        analysis_.total_scopes++;

        auto duration_ms = scope.elapsed_ms();
        analysis_.total_duration_ms += duration_ms;

        if (duration_ms > analysis_.max_duration_ms) {
            analysis_.max_duration_ms = duration_ms;
            analysis_.slowest_scope = scope.get_scope_name();
        }
    }

    std::string get_name() const override { return "ScopeAnalysisVisitor"; }

    const Analysis& get_analysis() const { return analysis_; }

    void reset() { analysis_ = Analysis{}; }

private:
    Analysis analysis_;
};

/**
 * @brief Factory for creating scoped loggers with different configurations
 */
class ScopedLoggerFactory : public base::Factory<ScopedLogger> {
public:
    static ScopedLoggerFactory& instance() {
        static ScopedLoggerFactory factory;
        return factory;
    }

    object_ptr<ScopedLogger> create_function_tracer(object_ptr<Logger> logger,
                                                    std::source_location loc = std::source_location::current()) {
        return base::make_object<ScopedLogger>(std::move(logger), "", LogLevel::TRACE, loc);
    }

    object_ptr<ScopedLogger> create_performance_tracer(object_ptr<Logger> logger,
                                                       const std::string& scope_name) {
        auto scoped = base::make_object<ScopedLogger>(std::move(logger), scope_name, LogLevel::INFO);
        scoped->set_slow_threshold(std::chrono::milliseconds(100)); // More sensitive for performance
        return scoped;
    }

    object_ptr<ScopedLogger> create_debug_tracer(object_ptr<Logger> logger,
                                                 const std::string& scope_name) {
        return base::make_object<ScopedLogger>(std::move(logger), scope_name, LogLevel::DEBUG);
    }

private:
    ScopedLoggerFactory() = default;
};

// Enhanced convenience macros using base infrastructure

/**
 * @brief Create a scoped logger for current function
 */
#define FEM_LOG_SCOPE(logger) \
    auto FEM_CONCAT(_scope_, __LINE__) = \
        fem::core::logging::ScopedLoggerFactory::instance().create_function_tracer(logger)

/**
 * @brief Create a performance-focused scoped logger
 */
#define FEM_LOG_PERF_SCOPE(logger, name) \
    auto FEM_CONCAT(_scope_, __LINE__) = \
        fem::core::logging::ScopedLoggerFactory::instance().create_performance_tracer(logger, name)

/**
 * @brief Create a scoped logger with custom name and level
 */
#define FEM_LOG_SCOPE_NAMED(logger, name, level) \
    auto FEM_CONCAT(_scope_, __LINE__) = \
        fem::core::base::make_object<fem::core::logging::ScopedLogger>(logger, name, level)

/**
 * @brief Create a scoped timer using base infrastructure
 */
#define FEM_LOG_TIME(logger, operation) \
    auto FEM_CONCAT(_timer_, __LINE__) = \
        fem::core::base::make_object<fem::core::logging::ScopedTimer>(logger, operation)

/**
 * @brief Add scope timing component to a logger
 */
#define FEM_ADD_SCOPE_TIMING(logger) \
    (logger)->template add_component<fem::core::logging::ScopeTimingComponent>()

/**
 * @brief Subscribe to scope events
 */
#define FEM_SUBSCRIBE_SCOPE_EVENTS(handler) \
    fem::core::base::subscribe_to_events<fem::core::logging::ScopeEvent>(handler)

// Helper macro for concatenation
#define FEM_CONCAT(a, b) FEM_CONCAT_IMPL(a, b)
#define FEM_CONCAT_IMPL(a, b) a##b

} // namespace fem::core::logging

#endif //LOGGING_SCOPEDLOGGER_H
