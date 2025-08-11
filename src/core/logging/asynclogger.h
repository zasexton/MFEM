#pragma once

#ifndef LOGGING_ASYNCLOGGER_H
#define LOGGING_ASYNCLOGGER_H

#include <thread>
#include <condition_variable>
#include <queue>
#include <atomic>

#include "logger.h"
#include "logbuffer.h"

#include "../base/component.h"
#include "../base/policy.h"

namespace fem::core::logging {

/**
 * @brief Async logging component using Component pattern
 *
 * This component can be attached to any Logger to provide async capabilities
 */
    class AsyncLoggingComponent : public TypedComponent<AsyncLoggingComponent>,
                                  public NonCopyableNonMovable<AsyncLoggingComponent> {
    public:
        struct Config {
            size_t queue_size = 8192;
            size_t batch_size = 100;
            int flush_interval_ms = 100;
            bool block_when_full = false;
            size_t worker_thread_count = 1;
        };

        explicit AsyncLoggingComponent(const Config& config = {})
                : TypedComponent("AsyncLogging")
                , config_(config) {}

        ~AsyncLoggingComponent() override {
            stop_worker_threads();
        }

        void on_attach(Entity* entity) override {
            TypedComponent::on_attach(entity);
            logger_ = dynamic_cast<Logger*>(entity);
            if (logger_) {
                start_worker_threads();
            }
        }

        void on_detach() override {
            stop_worker_threads();
            logger_ = nullptr;
            TypedComponent::on_detach();
        }

        void queue_message(const LogMessage& message) {
            bool enqueued = false;
            {
                std::unique_lock lock(queue_mutex_);

                if (message_queue_.size() < config_.queue_size) {
                    message_queue_.push(message.clone());
                    enqueued = true;
                    stats_.messages_queued++;
                } else if (config_.block_when_full) {
                    queue_not_full_.wait(lock, [this] {
                        return message_queue_.size() < config_.queue_size || !running_;
                    });

                    if (running_) {
                        message_queue_.push(message.clone());
                        enqueued = true;
                        stats_.messages_queued++;
                    }
                } else {
                    stats_.messages_dropped++;
                }
            }

            if (enqueued) {
                cv_.notify_one();
            }
        }

        void flush() {
            flush_requested_ = true;
            cv_.notify_all();

            std::unique_lock lock(flush_mutex_);
            flush_cv_.wait(lock, [this] {
                return !flush_requested_ || !running_;
            });
        }

        void update(double) override {
            // Could update statistics or perform maintenance
        }

        void reset() override {
            std::lock_guard lock(queue_mutex_);
            while (!message_queue_.empty()) {
                message_queue_.pop();
            }
            stats_ = {};
        }

        std::vector<std::type_index> get_dependencies() const override {
            return {};  // No dependencies on other components
        }

        struct Statistics {
            std::atomic<uint64_t> messages_queued{0};
            std::atomic<uint64_t> messages_processed{0};
            std::atomic<uint64_t> messages_dropped{0};
            std::atomic<uint64_t> flush_count{0};

            [[nodiscard]] size_t current_queue_size() const {
                return messages_queued - messages_processed;
            }

            [[nodiscard]] double drop_rate() const {
                auto total = messages_queued.load() + messages_dropped.load();
                return total > 0 ? static_cast<double>(messages_dropped) / total : 0.0;
            }
        };

        [[nodiscard]] const Statistics& get_statistics() const { return stats_; }

        [[nodiscard]] bool is_queue_full() const {
            std::lock_guard lock(queue_mutex_);
            return message_queue_.size() >= config_.queue_size;
        }

        void wait_until_empty() {
            std::unique_lock lock(queue_mutex_);
            queue_empty_.wait(lock, [this] {
                return message_queue_.empty() || !running_;
            });
        }

    private:
        void start_worker_threads() {
            running_ = true;
            for (size_t i = 0; i < config_.worker_thread_count; ++i) {
                workers_.emplace_back(&AsyncLoggingComponent::worker_thread, this);
            }
        }

        void stop_worker_threads() {
            running_ = false;
            cv_.notify_all();

            for (auto& worker : workers_) {
                if (worker.joinable()) {
                    worker.join();
                }
            }

            process_all_messages();
        }

        void worker_thread() {
            set_thread_name("AsyncLogger");

            std::vector<LogMessage> batch;
            batch.reserve(config_.batch_size);

            while (running_) {
                batch.clear();

                {
                    std::unique_lock lock(queue_mutex_);

                    cv_.wait_for(lock, std::chrono::milliseconds(config_.flush_interval_ms),
                                 [this] {
                                     return !message_queue_.empty() ||
                                            flush_requested_ ||
                                            !running_;
                                 });

                    while (!message_queue_.empty() && batch.size() < config_.batch_size) {
                        batch.push_back(std::move(message_queue_.front()));
                        message_queue_.pop();
                    }

                    if (batch.size() > 0) {
                        queue_not_full_.notify_all();
                    }

                    if (message_queue_.empty()) {
                        queue_empty_.notify_all();
                    }
                }

                for (const auto& msg : batch) {
                    if (logger_) {
                        logger_->write_to_sinks(msg);
                    }
                    stats_.messages_processed++;
                }

                if (flush_requested_) {
                    process_all_messages();
                    if (logger_) {
                        logger_->flush();
                    }
                    flush_requested_ = false;
                    stats_.flush_count++;
                    flush_cv_.notify_all();
                }
            }

            process_all_messages();
        }

        void process_all_messages() {
            std::queue<LogMessage> temp_queue;

            {
                std::lock_guard lock(queue_mutex_);
                temp_queue.swap(message_queue_);
            }

            while (!temp_queue.empty() && logger_) {
                logger_->write_to_sinks(temp_queue.front());
                temp_queue.pop();
                stats_.messages_processed++;
            }
        }

        void set_thread_name(const std::string& name) {
#ifdef __linux__
            pthread_setname_np(pthread_self(), name.c_str());
#elif defined(_WIN32)
            // Windows-specific implementation
#endif
        }

        Config config_;
        Logger* logger_{nullptr};

        mutable std::mutex queue_mutex_;
        std::condition_variable cv_;
        std::condition_variable queue_not_full_;
        std::condition_variable queue_empty_;
        std::queue<LogMessage> message_queue_;

        std::atomic<bool> running_{false};
        std::atomic<bool> flush_requested_{false};
        std::mutex flush_mutex_;
        std::condition_variable flush_cv_;

        std::vector<std::thread> workers_;
        Statistics stats_;
    };

/**
 * @brief Filter component for loggers
 */
    class FilterComponent : public TypedComponent<FilterComponent> {
    public:
        FilterComponent() : TypedComponent("LogFilter") {}

        void add_filter(std::unique_ptr<LogFilter> filter) {
            std::lock_guard lock(filters_mutex_);
            filters_.push_back(std::move(filter));
        }

        bool should_log(const LogMessage& message) const {
            std::lock_guard lock(filters_mutex_);
            for (const auto& filter : filters_) {
                if (!filter->should_log(message)) {
                    return false;
                }
            }
            return true;
        }

        void clear_filters() {
            std::lock_guard lock(filters_mutex_);
            filters_.clear();
        }

        void update(double) override {}
        void reset() override { clear_filters(); }

    private:
        std::vector<std::unique_ptr<LogFilter>> filters_;
        mutable std::mutex filters_mutex_;
    };

/**
 * @brief Buffering component for batch writing
 */
    class BufferingComponent : public TypedComponent<BufferingComponent> {
    public:
        explicit BufferingComponent(std::unique_ptr<LogBuffer> buffer)
                : TypedComponent("LogBuffering")
                , buffer_(std::move(buffer)) {}

        void buffer_message(const LogMessage& message) {
            if (!buffer_->push(message.clone())) {
                // Buffer full, flush it
                flush();
                buffer_->push(message.clone());
            }
        }

        void flush() {
            auto messages = buffer_->drain();
            if (logger_) {
                for (const auto& msg : messages) {
                    logger_->write_to_sinks(msg);
                }
            }
        }

        void on_attach(Entity* entity) override {
            TypedComponent::on_attach(entity);
            logger_ = dynamic_cast<Logger*>(entity);
        }

        void on_detach() override {
            flush();
            logger_ = nullptr;
            TypedComponent::on_detach();
        }

        void update(double) override {
            // Could periodically flush based on time
        }

        void reset() override {
            buffer_->clear();
        }

    private:
        std::unique_ptr<LogBuffer> buffer_;
        Logger* logger_{nullptr};
    };

} // namespace fem::core::logging

#endif //LOGGING_ASYNCLOGGER_H
