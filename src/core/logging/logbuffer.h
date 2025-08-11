#pragma once

#ifndef LOGGING_LOGBUFFER_H
#define LOGGING_LOGBUFFER_H

#include <vector>
#include <deque>
#include <mutex>
#include <atomic>

#include "logmessage.h"

namespace fem::core::logging {

/**
 * @brief Abstract base class for log message buffering
 *
 * Buffers temporarily store log messages before they're written to sinks.
 * Different strategies optimize for different use cases.
 *
 * Usage context:
 * - Batch writing to improve I/O performance
 * - Circular buffers for recent message history
 * - Priority queues for important messages
 * - Compression buffers to reduce storage
 */
    class LogBuffer {
    public:
        virtual ~LogBuffer() = default;

        /**
         * @brief Add a message to the buffer
         * @return true if message was added, false if buffer is full
         */
        [[nodiscard]] virtual bool push(LogMessage message) = 0;

        /**
         * @brief Remove and return the next message
         * @return Message if available, nullopt if buffer is empty
         */
        [[nodiscard]] virtual std::optional<LogMessage> pop() = 0;

        /**
         * @brief Get all messages and clear buffer
         */
        [[nodiscard]] virtual std::vector<LogMessage> drain() = 0;

        /**
         * @brief Get number of messages in buffer
         */
        [[nodiscard]] virtual size_t size() const = 0;

        /**
         * @brief Check if buffer is empty
         */
        [[nodiscard]] virtual bool empty() const = 0;

        /**
         * @brief Check if buffer is full
         */
        [[nodiscard]] virtual bool full() const = 0;

        /**
         * @brief Clear all messages
         */
        virtual void clear() = 0;

        /**
         * @brief Get buffer capacity
         */
        [[nodiscard]] virtual size_t capacity() const = 0;
    };

/**
 * @brief Simple FIFO buffer with fixed capacity
 */
    class FifoBuffer : public LogBuffer {
    public:
        explicit FifoBuffer(size_t capacity = 1000)
                : capacity_(capacity) {
            messages_.reserve(capacity);
        }

        [[nodiscard]] bool push(LogMessage message) override {
            std::lock_guard lock(mutex_);

            if (messages_.size() >= capacity_) {
                return false;
            }

            messages_.push_back(std::move(message));
            return true;
        }

        [[nodiscard]] std::optional<LogMessage> pop() override {
            std::lock_guard lock(mutex_);

            if (messages_.empty()) {
                return std::nullopt;
            }

            LogMessage msg = std::move(messages_.front());
            messages_.pop_front();
            return msg;
        }

        [[nodiscard]] std::vector<LogMessage> drain() override {
            std::lock_guard lock(mutex_);

            std::vector<LogMessage> result;
            result.reserve(messages_.size());

            for (auto& msg : messages_) {
                result.push_back(std::move(msg));
            }

            messages_.clear();
            return result;
        }

        [[nodiscard]] size_t size() const override {
            std::lock_guard lock(mutex_);
            return messages_.size();
        }

        [[nodiscard]] bool empty() const override {
            std::lock_guard lock(mutex_);
            return messages_.empty();
        }

        [[nodiscard]] bool full() const override {
            std::lock_guard lock(mutex_);
            return messages_.size() >= capacity_;
        }

        void clear() override {
            std::lock_guard lock(mutex_);
            messages_.clear();
        }

        [[nodiscard]] size_t capacity() const override {
            return capacity_;
        }

    private:
        mutable std::mutex mutex_;
        std::deque<LogMessage> messages_;
        size_t capacity_;
    };

/**
 * @brief Circular buffer that overwrites oldest messages
 */
    class CircularBuffer : public LogBuffer {
    public:
        explicit CircularBuffer(size_t capacity = 1000)
                : capacity_(capacity), buffer_(capacity) {}

        [[nodiscard]] bool push(LogMessage message) override {
            std::lock_guard lock(mutex_);

            buffer_[write_pos_] = std::move(message);
            write_pos_ = (write_pos_ + 1) % capacity_;

            if (size_ < capacity_) {
                size_++;
            } else {
                // Overwriting oldest message
                read_pos_ = (read_pos_ + 1) % capacity_;
                overwrites_++;
            }

            return true; // Always succeeds
        }

        [[nodiscard]] std::optional<LogMessage> pop() override {
            std::lock_guard lock(mutex_);

            if (size_ == 0) {
                return std::nullopt;
            }

            LogMessage msg = std::move(buffer_[read_pos_]);
            read_pos_ = (read_pos_ + 1) % capacity_;
            size_--;

            return msg;
        }

        [[nodiscard]] std::vector<LogMessage> drain() override {
            std::lock_guard lock(mutex_);

            std::vector<LogMessage> result;
            result.reserve(size_);

            size_t pos = read_pos_;
            for (size_t i = 0; i < size_; ++i) {
                result.push_back(std::move(buffer_[pos]));
                pos = (pos + 1) % capacity_;
            }

            read_pos_ = 0;
            write_pos_ = 0;
            size_ = 0;

            return result;
        }

        [[nodiscard]] size_t size() const override {
            std::lock_guard lock(mutex_);
            return size_;
        }

        [[nodiscard]] bool empty() const override {
            std::lock_guard lock(mutex_);
            return size_ == 0;
        }

        [[nodiscard]] bool full() const override {
            std::lock_guard lock(mutex_);
            return size_ >= capacity_;
        }

        void clear() override {
            std::lock_guard lock(mutex_);
            read_pos_ = 0;
            write_pos_ = 0;
            size_ = 0;
        }

        [[nodiscard]] size_t capacity() const override {
            return capacity_;
        }

        [[nodiscard]] uint64_t get_overwrite_count() const {
            return overwrites_.load();
        }

    private:
        mutable std::mutex mutex_;
        std::vector<LogMessage> buffer_;
        size_t capacity_;
        size_t read_pos_{0};
        size_t write_pos_{0};
        size_t size_{0};
        std::atomic<uint64_t> overwrites_{0};
    };

/**
 * @brief Priority buffer that sorts messages by severity
 */
    class PriorityBuffer : public LogBuffer {
    public:
        explicit PriorityBuffer(size_t capacity = 1000)
                : capacity_(capacity) {}

        [[nodiscard]] bool push(LogMessage message) override {
            std::lock_guard lock(mutex_);

            if (messages_.size() >= capacity_) {
                // If full, only accept if higher priority than lowest
                if (message.get_level() <= min_level_) {
                    return false;
                }

                // Remove lowest priority message
                remove_lowest_priority();
            }

            // Insert in priority order
            auto level = message.get_level();
            auto it = std::lower_bound(messages_.begin(), messages_.end(), message,
                                       [](const LogMessage& a, const LogMessage& b) {
                                           return a.get_level() > b.get_level(); // Higher severity first
                                       });

            messages_.insert(it, std::move(message));
            update_min_level();

            return true;
        }

        [[nodiscard]] std::optional<LogMessage> pop() override {
            std::lock_guard lock(mutex_);

            if (messages_.empty()) {
                return std::nullopt;
            }

            // Return highest priority (front)
            LogMessage msg = std::move(messages_.front());
            messages_.erase(messages_.begin());
            update_min_level();

            return msg;
        }

        [[nodiscard]] std::vector<LogMessage> drain() override {
            std::lock_guard lock(mutex_);

            std::vector<LogMessage> result = std::move(messages_);
            messages_.clear();
            min_level_ = LogLevel::OFF;

            return result;
        }

        [[nodiscard]] size_t size() const override {
            std::lock_guard lock(mutex_);
            return messages_.size();
        }

        [[nodiscard]] bool empty() const override {
            std::lock_guard lock(mutex_);
            return messages_.empty();
        }

        [[nodiscard]] bool full() const override {
            std::lock_guard lock(mutex_);
            return messages_.size() >= capacity_;
        }

        void clear() override {
            std::lock_guard lock(mutex_);
            messages_.clear();
            min_level_ = LogLevel::OFF;
        }

        [[nodiscard]] size_t capacity() const override {
            return capacity_;
        }

    private:
        void remove_lowest_priority() {
            if (!messages_.empty()) {
                messages_.pop_back();
            }
        }

        void update_min_level() {
            if (messages_.empty()) {
                min_level_ = LogLevel::OFF;
            } else {
                min_level_ = messages_.back().get_level();
            }
        }

        mutable std::mutex mutex_;
        std::vector<LogMessage> messages_;
        size_t capacity_;
        LogLevel min_level_{LogLevel::OFF};
    };

/**
 * @brief Time-window buffer that keeps messages from last N seconds
 */
    class TimeWindowBuffer : public LogBuffer {
    public:
        explicit TimeWindowBuffer(std::chrono::seconds window = std::chrono::seconds(60),
                                  size_t max_capacity = 10000)
                : window_(window), max_capacity_(max_capacity) {}

        [[nodiscard]] bool push(LogMessage message) override {
            std::lock_guard lock(mutex_);

            // Remove old messages
            remove_expired_messages();

            if (messages_.size() >= max_capacity_) {
                return false;
            }

            messages_.push_back(std::move(message));
            return true;
        }

        [[nodiscard]] std::optional<LogMessage> pop() override {
            std::lock_guard lock(mutex_);

            remove_expired_messages();

            if (messages_.empty()) {
                return std::nullopt;
            }

            LogMessage msg = std::move(messages_.front());
            messages_.pop_front();
            return msg;
        }

        [[nodiscard]] std::vector<LogMessage> drain() override {
            std::lock_guard lock(mutex_);

            remove_expired_messages();

            std::vector<LogMessage> result;
            result.reserve(messages_.size());

            for (auto& msg : messages_) {
                result.push_back(std::move(msg));
            }

            messages_.clear();
            return result;
        }

        [[nodiscard]] size_t size() const override {
            std::lock_guard lock(mutex_);
            const_cast<TimeWindowBuffer*>(this)->remove_expired_messages();
            return messages_.size();
        }

        [[nodiscard]] bool empty() const override {
            return size() == 0;
        }

        [[nodiscard]] bool full() const override {
            std::lock_guard lock(mutex_);
            return messages_.size() >= max_capacity_;
        }

        void clear() override {
            std::lock_guard lock(mutex_);
            messages_.clear();
        }

        [[nodiscard]] size_t capacity() const override {
            return max_capacity_;
        }

        void set_window(std::chrono::seconds window) {
            std::lock_guard lock(mutex_);
            window_ = window;
        }

    private:
        void remove_expired_messages() {
            auto now = std::chrono::system_clock::now();
            auto cutoff = now - window_;

            messages_.erase(
                    std::remove_if(messages_.begin(), messages_.end(),
                                   [cutoff](const LogMessage& msg) {
                                       return msg.get_timestamp() < cutoff;
                                   }),
                    messages_.end()
            );
        }

        mutable std::mutex mutex_;
        std::deque<LogMessage> messages_;
        std::chrono::seconds window_;
        size_t max_capacity_;
    };

/**
 * @brief Compression buffer that groups similar messages
 */
    class CompressionBuffer : public LogBuffer {
    public:
        explicit CompressionBuffer(size_t capacity = 1000)
                : capacity_(capacity) {}

        [[nodiscard]] bool push(LogMessage message) override {
            std::lock_guard lock(mutex_);

            // Try to find similar message
            auto hash = compute_hash(message);
            auto it = compressed_messages_.find(hash);

            if (it != compressed_messages_.end()) {
                // Increment count for similar message
                it->second.count++;
                it->second.last_timestamp = message.get_timestamp();
                return true;
            }

            if (compressed_messages_.size() >= capacity_) {
                return false;
            }

            // Add new compressed entry
            CompressedMessage compressed{
                    .original = std::move(message),
                    .count = 1,
                    .last_timestamp = message.get_timestamp()
            };

            compressed_messages_[hash] = std::move(compressed);
            return true;
        }

        [[nodiscard]] std::optional<LogMessage> pop() override {
            // Not applicable for compression buffer
            return std::nullopt;
        }

        [[nodiscard]] std::vector<LogMessage> drain() override {
            std::lock_guard lock(mutex_);

            std::vector<LogMessage> result;
            result.reserve(compressed_messages_.size());

            for (auto& [hash, compressed] : compressed_messages_) {
                if (compressed.count > 1) {
                    // Modify message to show repetition count
                    auto msg = compressed.original.clone();
                    std::string new_message = std::format(
                            "{} [repeated {} times, last at {}]",
                            msg.get_message(),
                            compressed.count,
                            format_timestamp(compressed.last_timestamp)
                    );

                    LogMessage repeated_msg(
                            msg.get_level(),
                            msg.get_logger_name(),
                            new_message,
                            msg.get_location()
                    );

                    result.push_back(std::move(repeated_msg));
                } else {
                    result.push_back(std::move(compressed.original));
                }
            }

            compressed_messages_.clear();
            return result;
        }

        [[nodiscard]] size_t size() const override {
            std::lock_guard lock(mutex_);
            return compressed_messages_.size();
        }

        [[nodiscard]] bool empty() const override {
            std::lock_guard lock(mutex_);
            return compressed_messages_.empty();
        }

        [[nodiscard]] bool full() const override {
            std::lock_guard lock(mutex_);
            return compressed_messages_.size() >= capacity_;
        }

        void clear() override {
            std::lock_guard lock(mutex_);
            compressed_messages_.clear();
        }

        [[nodiscard]] size_t capacity() const override {
            return capacity_;
        }

    private:
        struct CompressedMessage {
            LogMessage original;
            uint32_t count;
            std::chrono::system_clock::time_point last_timestamp;
        };

        size_t compute_hash(const LogMessage& msg) const {
            size_t hash = 0;
            hash ^= std::hash<std::string>{}(msg.get_message());
            hash ^= std::hash<int>{}(static_cast<int>(msg.get_level()));
            hash ^= std::hash<std::string>{}(msg.get_logger_name());
            return hash;
        }

        std::string format_timestamp(std::chrono::system_clock::time_point tp) const {
            auto time_t = std::chrono::system_clock::to_time_t(tp);
            std::tm tm{};
#ifdef _WIN32
            localtime_s(&tm, &time_t);
#else
            localtime_r(&time_t, &tm);
#endif
            char buffer[32];
            std::strftime(buffer, sizeof(buffer), "%H:%M:%S", &tm);
            return buffer;
        }

        mutable std::mutex mutex_;
        std::unordered_map<size_t, CompressedMessage> compressed_messages_;
        size_t capacity_;
    };

} // namespace fem::core::logging

#endif //LOGGING_LOGBUFFER_H
