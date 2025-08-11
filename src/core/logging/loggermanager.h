#pragma once

#ifndef LOGGING_LOGGERMANAGER_H
#define LOGGING_LOGGERMANAGER_H

#include <regex>

#include "logger.h"

#include "../base/singleton.h"
#include "../base/registry.h"
#include "../base/factory.h"
#include "../base/observer.h"

namespace fem::core::logging {

/**
 * @brief Configuration for a logger
 */
    struct LoggerConfig {
        LogLevel level{LogLevelConfig::DEFAULT_LEVEL};
        bool enabled{true};
        std::vector<std::string> sink_names;
        std::unordered_map<std::string, std::string> properties;
    };

/**
 * @brief Centralized manager for all loggers using base infrastructure
 *
 * Uses:
 * - Singleton: Single global instance
 * - Registry: Manages logger collection
 * - Factory: Creates sinks by configuration
 * - Observer: Notifies about logger changes
 */
    class LoggerManager : public Singleton<LoggerManager>,
                          public Subject<LoggerEvent> {
        friend class Singleton<LoggerManager>;

    public:
        using LoggerPtr = object_ptr<Logger>;
        using SinkPtr = std::shared_ptr<LogSink>;

        /**
         * @brief Get or create a logger with the given name
         */
        LoggerPtr get_logger(const std::string& name) {
            // Check if logger exists in registry
            if (auto logger = logger_registry_.find_by_key(name)) {
                return logger;
            }

            // Create new logger
            auto logger = make_object<Logger>(name);

            // Apply configuration
            configure_logger(logger);

            // Register it
            logger_registry_.register_object(name, logger);

            // Notify observers
            LoggerEvent event(LoggerEvent::LOGGER_ENABLED, name);
            notify_observers(event);
            emit_event<LoggerEvent>(event);

            return logger;
        }

        /**
         * @brief Get root logger
         */
        LoggerPtr get_root_logger() {
            return get_logger("root");
        }

        /**
         * @brief Set level for a logger and optionally its children
         */
        void set_level(const std::string& name, LogLevel level, bool include_children = true) {
            // Store configuration
            logger_configs_[name].level = level;

            // Apply to existing loggers
            auto all_loggers = logger_registry_.get_all_objects();
            for (auto& logger : all_loggers) {
                if (logger->name() == name ||
                    (include_children && is_child_logger(logger->name(), name))) {
                    logger->set_level(level);
                }
            }
        }

        /**
         * @brief Create and register a sink
         */
        SinkPtr create_sink(const std::string& type, const std::string& name,
                            const std::unordered_map<std::string, std::string>& params = {}) {
            // Use factory to create sink
            auto sink = sink_factory_.create(type, params);

            // Register it
            named_sinks_[name] = sink;

            return sink;
        }

        /**
         * @brief Add a sink to all loggers
         */
        void add_global_sink(SinkPtr sink) {
            global_sinks_.push_back(sink);

            // Add to all existing loggers
            auto all_loggers = logger_registry_.get_all_objects();
            for (auto& logger : all_loggers) {
                logger->add_sink(sink);
            }
        }

        /**
         * @brief Remove a global sink
         */
        void remove_global_sink(const SinkPtr& sink) {
            global_sinks_.erase(
                    std::remove(global_sinks_.begin(), global_sinks_.end(), sink),
                    global_sinks_.end()
            );

            // Remove from all existing loggers
            auto all_loggers = logger_registry_.get_all_objects();
            for (auto& logger : all_loggers) {
                logger->remove_sink(sink);
            }
        }

        /**
         * @brief Clear all global sinks
         */
        void clear_global_sinks() {
            global_sinks_.clear();

            // Clear from all existing loggers
            auto all_loggers = logger_registry_.get_all_objects();
            for (auto& logger : all_loggers) {
                logger->clear_sinks();
            }
        }

        /**
         * @brief Add a sink to loggers matching pattern
         */
        void add_pattern_sink(const std::string& pattern, SinkPtr sink) {
            pattern_sinks_[pattern].push_back(sink);

            // Apply to matching existing loggers
            std::regex re(pattern);
            auto all_loggers = logger_registry_.get_all_objects();
            for (auto& logger : all_loggers) {
                if (std::regex_match(logger->name(), re)) {
                    logger->add_sink(sink);
                }
            }
        }

        /**
         * @brief Set default level for new loggers
         */
        void set_default_level(LogLevel level) {
            default_level_ = level;
        }

        /**
         * @brief Get all logger names
         */
        std::vector<std::string> get_logger_names() const {
            return logger_registry_.get_all_keys();
        }

        /**
         * @brief Get logger by name (returns nullptr if not found)
         */
        LoggerPtr find_logger(const std::string& name) const {
            return logger_registry_.find_by_key(name);
        }

        /**
         * @brief Visit all loggers with a visitor
         */
        void visit_all_loggers(Visitor<Logger>& visitor) {
            auto all_loggers = logger_registry_.get_all_objects();
            for (auto& logger : all_loggers) {
                visitor.visit(*logger);
            }
        }

        /**
         * @brief Flush all loggers
         */
        void flush_all() {
            auto all_loggers = logger_registry_.get_all_objects();
            for (auto& logger : all_loggers) {
                logger->flush();
            }
        }

        /**
         * @brief Reset all loggers to a specific level
         */
        void reset_all_levels(LogLevel level) {
            auto all_loggers = logger_registry_.get_all_objects();
            for (auto& logger : all_loggers) {
                logger->set_level(level);
            }
        }

        /**
         * @brief Enable/disable all loggers
         */
        void set_all_enabled(bool enabled) {
            auto all_loggers = logger_registry_.get_all_objects();
            for (auto& logger : all_loggers) {
                logger->set_enabled(enabled);
            }
        }

        /**
         * @brief Configure logging from a configuration string
         * Format: "logger_name=level;logger_name2=level2;..."
         */
        void configure_from_string(const std::string& config) {
            std::istringstream stream(config);
            std::string item;

            while (std::getline(stream, item, ';')) {
                auto pos = item.find('=');
                if (pos != std::string::npos) {
                    std::string logger_name = item.substr(0, pos);
                    std::string level_str = item.substr(pos + 1);

                    // Trim whitespace
                    logger_name.erase(0, logger_name.find_first_not_of(" \t"));
                    logger_name.erase(logger_name.find_last_not_of(" \t") + 1);
                    level_str.erase(0, level_str.find_first_not_of(" \t"));
                    level_str.erase(level_str.find_last_not_of(" \t") + 1);

                    try {
                        LogLevel level = from_string(level_str);
                        set_level(logger_name, level);
                    } catch (const std::exception& e) {
                        if (auto root = get_root_logger()) {
                            root->error("Invalid log configuration: {}", e.what());
                        }
                    }
                }
            }
        }

        /**
         * @brief Load configuration from environment variable
         */
        void configure_from_env(const char* env_var = "FEM_LOG_LEVELS") {
            if (const char* config = std::getenv(env_var)) {
                configure_from_string(config);
            }
        }

        /**
         * @brief Register a custom sink type with the factory
         */
        template<typename SinkType>
        void register_sink_type(const std::string& name) {
            sink_factory_.register_type<SinkType>(name);
        }

        /**
         * @brief Get the logger registry
         */
        Registry<Logger, std::string>& get_registry() {
            return logger_registry_;
        }

        /**
         * @brief Get the sink factory
         */
        Factory<LogSink>& get_sink_factory() {
            return sink_factory_;
        }

        struct Statistics {
            size_t total_loggers;
            size_t active_loggers;
            size_t global_sinks;
            size_t named_sinks;
            std::unordered_map<LogLevel, size_t> level_counts;
        };

        /**
         * @brief Get logging system statistics
         */
        Statistics get_statistics() const {
            Statistics stats{};
            stats.total_loggers = logger_registry_.size();
            stats.global_sinks = global_sinks_.size();
            stats.named_sinks = named_sinks_.size();

            auto all_loggers = logger_registry_.get_all_objects();
            for (const auto& logger : all_loggers) {
                if (logger->is_enabled()) {
                    stats.active_loggers++;
                }
                stats.level_counts[logger->get_level()]++;
            }

            return stats;
        }

    private:
        LoggerManager() : default_level_(LogLevelConfig::DEFAULT_LEVEL) {
            // Register default sink types
            register_default_sink_types();

            // Initialize with default console sink
            auto console_sink = std::make_shared<ConsoleSink>();
            console_sink->set_formatter(std::make_unique<BasicLogFormatter>());
            global_sinks_.push_back(console_sink);

            // Set up registry callbacks
            setup_registry_callbacks();
        }

        void register_default_sink_types() {
            sink_factory_.register_type<ConsoleSink>("console");
            sink_factory_.register_type<FileSink>("file");
            sink_factory_.register_type<MemorySink>("memory");
            sink_factory_.register_type<NullSink>("null");
            sink_factory_.register_type<MultiSink>("multi");
        }

        void setup_registry_callbacks() {
            // Add callback for when loggers are registered
            logger_registry_.add_registration_callback(
                    [this](object_ptr<Logger> logger) {
                        // Apply any pending configuration
                        configure_logger(logger);
                    }
            );
        }

        /**
         * @brief Check if a logger is a child of another
         */
        bool is_child_logger(const std::string& child, const std::string& parent) const {
            if (child.size() <= parent.size()) return false;
            if (child.substr(0, parent.size()) != parent) return false;
            return child.size() == parent.size() || child[parent.size()] == '.';
        }

        /**
         * @brief Configure a newly created logger
         */
        void configure_logger(LoggerPtr& logger) {
            const std::string& name = logger->name();

            // Set level based on most specific configuration
            LogLevel level = default_level_;
            std::string best_match;

            for (const auto& [config_name, config] : logger_configs_) {
                if (name == config_name || is_child_logger(name, config_name)) {
                    if (config_name.size() > best_match.size()) {
                        best_match = config_name;
                        level = config.level;
                    }
                }
            }

            logger->set_level(level);

            // Add global sinks
            for (const auto& sink : global_sinks_) {
                logger->add_sink(sink);
            }

            // Add pattern-based sinks
            for (const auto& [pattern, sinks] : pattern_sinks_) {
                std::regex re(pattern);
                if (std::regex_match(name, re)) {
                    for (const auto& sink : sinks) {
                        logger->add_sink(sink);
                    }
                }
            }

            // Apply any logger-specific configuration
            if (auto it = logger_configs_.find(name); it != logger_configs_.end()) {
                const auto& config = it->second;
                logger->set_enabled(config.enabled);

                // Add configured sinks
                for (const auto& sink_name : config.sink_names) {
                    if (auto sink_it = named_sinks_.find(sink_name);
                            sink_it != named_sinks_.end()) {
                        logger->add_sink(sink_it->second);
                    }
                }
            }
        }

        Registry<Logger, std::string> logger_registry_{"Loggers"};
        Factory<LogSink> sink_factory_;
        std::unordered_map<std::string, LoggerConfig> logger_configs_;
        std::unordered_map<std::string, std::vector<SinkPtr>> pattern_sinks_;
        std::unordered_map<std::string, SinkPtr> named_sinks_;
        std::vector<SinkPtr> global_sinks_;
        std::atomic<LogLevel> default_level_;
    };

// Convenience functions

/**
 * @brief Get a logger by name
 */
    inline object_ptr<Logger> get_logger(const std::string& name) {
        return LoggerManager::instance().get_logger(name);
    }

/**
 * @brief Get the root logger
 */
    inline object_ptr<Logger> get_root_logger() {
        return LoggerManager::instance().get_root_logger();
    }

/**
 * @brief Create a logger with async capability
 */
    inline object_ptr<Logger> get_async_logger(const std::string& name) {
        auto logger = get_logger(name);
        if (!logger->has_component<AsyncLoggingComponent>()) {
            logger->add_component<AsyncLoggingComponent>();
        }
        return logger;
    }

} // namespace fem::core::logging

#endif //LOGGING_LOGGERMANAGER_H
