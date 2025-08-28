# Core I/O - AGENT.md

## Purpose
The `io/` layer provides comprehensive input/output abstractions for various data sources and sinks, supporting synchronous and asynchronous operations, multiple formats, buffering strategies, and protocol implementations. It creates a unified interface for file I/O, network communication, inter-process communication, and stream processing while maintaining high performance and type safety.

## Architecture Philosophy
- **Stream abstraction**: Universal stream interface for all I/O operations
- **Async-first design**: Non-blocking I/O with optional synchronous fallback
- **Zero-copy where possible**: Minimize data copying with view-based operations
- **Protocol agnostic**: Support various protocols through common interfaces
- **Composable filters**: Chain I/O operations (compression, encryption, encoding)

## Files Overview

### Core Abstractions
```cpp
stream.hpp           // Base stream interface
reader.hpp           // Universal reader interface
writer.hpp           // Universal writer interface
seekable.hpp         // Seekable stream interface
closeable.hpp        // Resource cleanup interface
io_error.hpp         // I/O error hierarchy
```

### Stream Implementations
```cpp
file_stream.hpp      // File-based I/O
memory_stream.hpp    // Memory buffer I/O
string_stream.hpp    // String-based I/O
null_stream.hpp      // Null device (/dev/null)
pipe_stream.hpp      // Pipe/FIFO streams
socket_stream.hpp    // Network socket I/O
```

### Buffering
```cpp
buffered_stream.hpp  // Buffered I/O wrapper
circular_buffer.hpp  // Ring buffer implementation
double_buffer.hpp    // Double buffering for smooth I/O
cache_stream.hpp     // LRU cache for repeated reads
prefetch_stream.hpp  // Read-ahead buffering
```

### Async I/O
```cpp
async_reader.hpp     // Asynchronous reading
async_writer.hpp     // Asynchronous writing
io_service.hpp       // I/O completion service
io_context.hpp       // Async I/O context
completion_handler.hpp // Completion callbacks
future_io.hpp        // Future-based async I/O
```

### Data Formats
```cpp
binary_reader.hpp    // Binary data reading
binary_writer.hpp    // Binary data writing
text_reader.hpp      // Text processing with encoding
text_writer.hpp      // Text output with formatting
line_reader.hpp      // Line-by-line reading
csv_reader.hpp       // CSV parsing
json_reader.hpp      // JSON streaming parser
xml_reader.hpp       // XML SAX parser
```

### Encoding & Transformation
```cpp
encoding.hpp         // Character encoding conversion
endian.hpp          // Endianness conversion
base64.hpp          // Base64 encoding/decoding
hex_encoding.hpp    // Hexadecimal encoding
compression.hpp     // Compression filters
encryption.hpp      // Encryption filters
```

### Network I/O
```cpp
tcp_stream.hpp      // TCP socket streams
udp_stream.hpp      // UDP datagram I/O
http_stream.hpp     // HTTP client/server streams
websocket_stream.hpp // WebSocket communication
serial_port.hpp     // Serial port I/O
```

### Utilities
```cpp
io_utils.hpp        // Common I/O utilities
stream_copier.hpp   // Efficient stream copying
stream_splitter.hpp // Tee streams to multiple outputs
stream_merger.hpp   // Merge multiple input streams
rate_limiter.hpp    // Bandwidth throttling
progress_reporter.hpp // I/O progress tracking
```

## Detailed Component Specifications

### `stream.hpp`
```cpp
class Stream {
public:
    enum class SeekOrigin {
        Begin,
        Current,
        End
    };
    
    enum class Mode {
        Read = 0x01,
        Write = 0x02,
        ReadWrite = Read | Write,
        Append = 0x04,
        Binary = 0x08,
        Text = 0x10
    };
    
    virtual ~Stream() = default;
    
    // Core I/O operations
    virtual std::size_t read(void* buffer, std::size_t size) = 0;
    virtual std::size_t write(const void* data, std::size_t size) = 0;
    virtual void flush() = 0;
    
    // Stream state
    virtual bool is_open() const = 0;
    virtual bool is_readable() const = 0;
    virtual bool is_writable() const = 0;
    virtual bool is_seekable() const = 0;
    virtual bool eof() const = 0;
    virtual bool good() const = 0;
    
    // Position management
    virtual std::size_t tell() const = 0;
    virtual void seek(std::ptrdiff_t offset, SeekOrigin origin = SeekOrigin::Begin) = 0;
    virtual std::size_t size() const = 0;
    
    // Convenience methods
    virtual std::vector<uint8_t> read_all() {
        std::vector<uint8_t> buffer;
        buffer.reserve(available());
        
        constexpr std::size_t chunk_size = 8192;
        uint8_t chunk[chunk_size];
        
        std::size_t bytes_read;
        while ((bytes_read = read(chunk, chunk_size)) > 0) {
            buffer.insert(buffer.end(), chunk, chunk + bytes_read);
        }
        
        return buffer;
    }
    
    virtual std::string read_string(std::size_t max_size = std::numeric_limits<std::size_t>::max()) {
        std::string result;
        result.reserve(std::min(available(), max_size));
        
        char ch;
        while (result.size() < max_size && read(&ch, 1) == 1 && ch != '\0') {
            result.push_back(ch);
        }
        
        return result;
    }
    
    virtual void write_string(const std::string& str) {
        write(str.data(), str.size());
    }
    
    // Typed I/O
    template<typename T>
    T read_value() {
        static_assert(std::is_trivially_copyable_v<T>, "Type must be trivially copyable");
        T value;
        if (read(&value, sizeof(T)) != sizeof(T)) {
            throw IOError("Failed to read value of type " + std::string(typeid(T).name()));
        }
        return value;
    }
    
    template<typename T>
    void write_value(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>, "Type must be trivially copyable");
        if (write(&value, sizeof(T)) != sizeof(T)) {
            throw IOError("Failed to write value of type " + std::string(typeid(T).name()));
        }
    }
    
    // Bulk operations
    template<typename Container>
    void write_container(const Container& container) {
        write_value<std::size_t>(container.size());
        for (const auto& item : container) {
            write_value(item);
        }
    }
    
    template<typename Container>
    Container read_container() {
        Container container;
        std::size_t size = read_value<std::size_t>();
        container.reserve(size);
        
        for (std::size_t i = 0; i < size; ++i) {
            container.push_back(read_value<typename Container::value_type>());
        }
        
        return container;
    }
    
protected:
    virtual std::size_t available() const {
        if (is_seekable()) {
            std::size_t current = tell();
            const_cast<Stream*>(this)->seek(0, SeekOrigin::End);
            std::size_t end = tell();
            const_cast<Stream*>(this)->seek(current, SeekOrigin::Begin);
            return end - current;
        }
        return 0;
    }
};

// Stream operators for convenience
template<typename T>
Stream& operator<<(Stream& stream, const T& value) {
    if constexpr (std::is_arithmetic_v<T>) {
        stream.write_value(value);
    } else {
        stream.write_string(std::to_string(value));
    }
    return stream;
}

template<typename T>
Stream& operator>>(Stream& stream, T& value) {
    if constexpr (std::is_arithmetic_v<T>) {
        value = stream.read_value<T>();
    } else {
        value = stream.read_string();
    }
    return stream;
}
```
**Why necessary**: Unified interface for all I/O operations, polymorphic stream handling.
**Usage**: Base for all stream implementations, generic I/O algorithms.

### `reader.hpp` & `writer.hpp`
```cpp
class Reader {
public:
    virtual ~Reader() = default;
    
    // Core reading
    virtual std::size_t read(void* buffer, std::size_t size) = 0;
    virtual bool eof() const = 0;
    
    // Convenience methods
    virtual std::vector<uint8_t> read_bytes(std::size_t count) {
        std::vector<uint8_t> buffer(count);
        std::size_t bytes_read = read(buffer.data(), count);
        buffer.resize(bytes_read);
        return buffer;
    }
    
    virtual std::string read_line(char delimiter = '\n') {
        std::string line;
        char ch;
        while (read(&ch, 1) == 1 && ch != delimiter) {
            line.push_back(ch);
        }
        return line;
    }
    
    virtual std::vector<std::string> read_all_lines() {
        std::vector<std::string> lines;
        while (!eof()) {
            lines.push_back(read_line());
        }
        return lines;
    }
    
    // Typed reading with endianness support
    template<typename T>
    T read_binary(Endianness endian = Endianness::Native) {
        T value;
        read(&value, sizeof(T));
        
        if (endian != Endianness::Native) {
            value = swap_endian(value);
        }
        
        return value;
    }
};

class Writer {
public:
    virtual ~Writer() = default;
    
    // Core writing
    virtual std::size_t write(const void* data, std::size_t size) = 0;
    virtual void flush() = 0;
    
    // Convenience methods
    virtual void write_bytes(const std::vector<uint8_t>& data) {
        write(data.data(), data.size());
    }
    
    virtual void write_line(const std::string& line) {
        write(line.data(), line.size());
        write("\n", 1);
    }
    
    virtual void write_lines(const std::vector<std::string>& lines) {
        for (const auto& line : lines) {
            write_line(line);
        }
    }
    
    // Typed writing with endianness support
    template<typename T>
    void write_binary(T value, Endianness endian = Endianness::Native) {
        if (endian != Endianness::Native) {
            value = swap_endian(value);
        }
        write(&value, sizeof(T));
    }
    
    // Printf-style formatting
    template<typename... Args>
    void write_formatted(const std::string& format, Args&&... args) {
        auto formatted = std::vformat(format, std::make_format_args(args...));
        write(formatted.data(), formatted.size());
    }
};

// Typed reader/writer for specific formats
template<typename T>
class TypedReader : public Reader {
public:
    virtual T read_object() = 0;
    virtual std::vector<T> read_objects(std::size_t count) {
        std::vector<T> objects;
        objects.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            objects.push_back(read_object());
        }
        return objects;
    }
};

template<typename T>
class TypedWriter : public Writer {
public:
    virtual void write_object(const T& object) = 0;
    virtual void write_objects(const std::vector<T>& objects) {
        for (const auto& obj : objects) {
            write_object(obj);
        }
    }
};
```
**Why necessary**: Simplified interfaces for read-only and write-only operations.
**Usage**: File readers, network receivers, data parsers.

### `buffered_stream.hpp`
```cpp
class BufferedStream : public Stream {
    std::unique_ptr<Stream> underlying_;
    
    // Read buffer
    std::vector<uint8_t> read_buffer_;
    std::size_t read_pos_ = 0;
    std::size_t read_size_ = 0;
    
    // Write buffer
    std::vector<uint8_t> write_buffer_;
    std::size_t write_pos_ = 0;
    
    std::size_t buffer_size_;
    
public:
    explicit BufferedStream(std::unique_ptr<Stream> stream, 
                           std::size_t buffer_size = 8192)
        : underlying_(std::move(stream))
        , read_buffer_(buffer_size)
        , write_buffer_(buffer_size)
        , buffer_size_(buffer_size) {
    }
    
    std::size_t read(void* buffer, std::size_t size) override {
        uint8_t* dest = static_cast<uint8_t*>(buffer);
        std::size_t total_read = 0;
        
        while (size > 0) {
            // Use buffered data first
            std::size_t buffered = read_size_ - read_pos_;
            if (buffered > 0) {
                std::size_t to_copy = std::min(size, buffered);
                std::memcpy(dest, read_buffer_.data() + read_pos_, to_copy);
                
                dest += to_copy;
                size -= to_copy;
                read_pos_ += to_copy;
                total_read += to_copy;
            }
            
            // Refill buffer if needed
            if (size > 0 && !eof()) {
                fill_read_buffer();
                if (read_size_ == 0) {
                    break;  // EOF reached
                }
            } else {
                break;
            }
        }
        
        return total_read;
    }
    
    std::size_t write(const void* data, std::size_t size) override {
        const uint8_t* src = static_cast<const uint8_t*>(data);
        std::size_t total_written = 0;
        
        while (size > 0) {
            std::size_t available = buffer_size_ - write_pos_;
            
            if (available == 0) {
                flush_write_buffer();
                available = buffer_size_;
            }
            
            std::size_t to_copy = std::min(size, available);
            std::memcpy(write_buffer_.data() + write_pos_, src, to_copy);
            
            src += to_copy;
            size -= to_copy;
            write_pos_ += to_copy;
            total_written += to_copy;
        }
        
        return total_written;
    }
    
    void flush() override {
        flush_write_buffer();
        underlying_->flush();
    }
    
    // Optimized line reading
    std::string read_line(char delimiter = '\n') {
        std::string line;
        
        while (true) {
            // Search in buffer
            auto* start = read_buffer_.data() + read_pos_;
            auto* end = read_buffer_.data() + read_size_;
            auto* found = std::find(start, end, delimiter);
            
            if (found != end) {
                // Found delimiter in buffer
                line.append(reinterpret_cast<char*>(start), found - start);
                read_pos_ = (found - read_buffer_.data()) + 1;
                return line;
            }
            
            // Add remaining buffer to line
            line.append(reinterpret_cast<char*>(start), end - start);
            read_pos_ = read_size_;
            
            // Try to refill buffer
            fill_read_buffer();
            if (read_size_ == 0) {
                break;  // EOF
            }
        }
        
        return line;
    }
    
    // Buffer management
    void set_buffer_size(std::size_t size) {
        flush();
        buffer_size_ = size;
        read_buffer_.resize(size);
        write_buffer_.resize(size);
    }
    
    std::size_t get_buffer_size() const {
        return buffer_size_;
    }
    
private:
    void fill_read_buffer() {
        read_pos_ = 0;
        read_size_ = underlying_->read(read_buffer_.data(), buffer_size_);
    }
    
    void flush_write_buffer() {
        if (write_pos_ > 0) {
            underlying_->write(write_buffer_.data(), write_pos_);
            write_pos_ = 0;
        }
    }
};

// Double buffering for smooth I/O
class DoubleBufferedStream : public Stream {
    std::unique_ptr<Stream> underlying_;
    std::array<std::vector<uint8_t>, 2> buffers_;
    std::atomic<int> active_buffer_{0};
    std::future<std::size_t> read_future_;
    std::size_t buffer_size_;
    
public:
    DoubleBufferedStream(std::unique_ptr<Stream> stream, std::size_t buffer_size)
        : underlying_(std::move(stream))
        , buffer_size_(buffer_size) {
        buffers_[0].resize(buffer_size);
        buffers_[1].resize(buffer_size);
        
        // Start first async read
        start_async_read(0);
    }
    
    std::size_t read(void* buffer, std::size_t size) override {
        int current = active_buffer_.load();
        
        // Wait for current buffer to be ready
        std::size_t available = read_future_.get();
        
        // Start reading next buffer asynchronously
        int next = 1 - current;
        start_async_read(next);
        
        // Copy from current buffer
        std::size_t to_copy = std::min(size, available);
        std::memcpy(buffer, buffers_[current].data(), to_copy);
        
        // Switch buffers
        active_buffer_.store(next);
        
        return to_copy;
    }
    
private:
    void start_async_read(int buffer_index) {
        read_future_ = std::async(std::launch::async, [this, buffer_index]() {
            return underlying_->read(buffers_[buffer_index].data(), buffer_size_);
        });
    }
};
```
**Why necessary**: Reduce system calls, improve I/O performance, efficient line reading.
**Usage**: File I/O, network streams, any high-throughput I/O.

### `async_reader.hpp`
```cpp
class AsyncReader {
public:
    using ReadCallback = std::function<void(const std::vector<uint8_t>& data, std::error_code ec)>;
    using CompletionToken = std::variant<ReadCallback, std::promise<std::vector<uint8_t>>*>;
    
    virtual ~AsyncReader() = default;
    
    // Async read operations
    virtual void async_read(void* buffer, std::size_t size, ReadCallback callback) = 0;
    
    virtual std::future<std::vector<uint8_t>> async_read(std::size_t size) {
        auto promise = std::make_shared<std::promise<std::vector<uint8_t>>>();
        auto future = promise->get_future();
        
        async_read_bytes(size, [promise](const std::vector<uint8_t>& data, std::error_code ec) {
            if (ec) {
                promise->set_exception(std::make_exception_ptr(IOError(ec.message())));
            } else {
                promise->set_value(data);
            }
        });
        
        return future;
    }
    
    // Async read with timeout
    virtual std::future<std::vector<uint8_t>> async_read_timeout(
        std::size_t size, 
        std::chrono::milliseconds timeout) {
        
        auto promise = std::make_shared<std::promise<std::vector<uint8_t>>>();
        auto future = promise->get_future();
        
        auto timer = std::make_shared<std::chrono::steady_clock::time_point>(
            std::chrono::steady_clock::now() + timeout
        );
        
        async_read_bytes(size, [promise, timer, timeout](
            const std::vector<uint8_t>& data, 
            std::error_code ec) {
            
            if (std::chrono::steady_clock::now() > *timer) {
                promise->set_exception(std::make_exception_ptr(
                    TimeoutError("Read timeout after " + std::to_string(timeout.count()) + "ms")
                ));
            } else if (ec) {
                promise->set_exception(std::make_exception_ptr(IOError(ec.message())));
            } else {
                promise->set_value(data);
            }
        });
        
        return future;
    }
    
    // Async read until delimiter
    virtual void async_read_until(char delimiter, ReadCallback callback) = 0;
    
    // Async read line
    virtual std::future<std::string> async_read_line() {
        auto promise = std::make_shared<std::promise<std::string>>();
        auto future = promise->get_future();
        
        async_read_until('\n', [promise](const std::vector<uint8_t>& data, std::error_code ec) {
            if (ec) {
                promise->set_exception(std::make_exception_ptr(IOError(ec.message())));
            } else {
                promise->set_value(std::string(data.begin(), data.end()));
            }
        });
        
        return future;
    }
    
protected:
    virtual void async_read_bytes(std::size_t size, ReadCallback callback) {
        std::vector<uint8_t> buffer(size);
        async_read(buffer.data(), size, std::move(callback));
    }
};

// Async I/O service for completion handling
class IOService {
    struct Operation {
        std::function<void()> handler;
        std::chrono::steady_clock::time_point deadline;
    };
    
    std::queue<Operation> completion_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> running_{true};
    
public:
    explicit IOService(std::size_t thread_count = std::thread::hardware_concurrency())
        : worker_threads_(thread_count) {
        
        for (auto& thread : worker_threads_) {
            thread = std::thread([this] { worker_loop(); });
        }
    }
    
    ~IOService() {
        stop();
        for (auto& thread : worker_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }
    
    void post(std::function<void()> handler) {
        {
            std::lock_guard lock(queue_mutex_);
            completion_queue_.push({std::move(handler), {}});
        }
        queue_cv_.notify_one();
    }
    
    void post_delayed(std::function<void()> handler, std::chrono::milliseconds delay) {
        auto deadline = std::chrono::steady_clock::now() + delay;
        {
            std::lock_guard lock(queue_mutex_);
            completion_queue_.push({std::move(handler), deadline});
        }
        queue_cv_.notify_one();
    }
    
    void run() {
        worker_loop();
    }
    
    void stop() {
        running_ = false;
        queue_cv_.notify_all();
    }
    
private:
    void worker_loop() {
        while (running_) {
            std::unique_lock lock(queue_mutex_);
            
            queue_cv_.wait(lock, [this] {
                return !completion_queue_.empty() || !running_;
            });
            
            if (!running_) break;
            
            auto op = std::move(completion_queue_.front());
            completion_queue_.pop();
            lock.unlock();
            
            // Check deadline
            if (op.deadline != std::chrono::steady_clock::time_point{}) {
                auto now = std::chrono::steady_clock::now();
                if (now < op.deadline) {
                    std::this_thread::sleep_until(op.deadline);
                }
            }
            
            op.handler();
        }
    }
};
```
**Why necessary**: Non-blocking I/O, concurrent operations, event-driven I/O.
**Usage**: Network servers, async file operations, responsive UIs.

### `binary_reader.hpp` & `binary_writer.hpp`
```cpp
class BinaryReader : public TypedReader<std::any> {
    std::unique_ptr<Stream> stream_;
    Endianness endianness_;
    
public:
    BinaryReader(std::unique_ptr<Stream> stream, 
                 Endianness endian = Endianness::Native)
        : stream_(std::move(stream)), endianness_(endian) {}
    
    // Primitive type reading
    template<typename T>
    T read() {
        static_assert(std::is_trivially_copyable_v<T>);
        T value;
        stream_->read(&value, sizeof(T));
        
        if (endianness_ != Endianness::Native) {
            value = swap_endian(value);
        }
        
        return value;
    }
    
    // String reading
    std::string read_string() {
        uint32_t length = read<uint32_t>();
        std::string str(length, '\0');
        stream_->read(str.data(), length);
        return str;
    }
    
    std::string read_fixed_string(std::size_t size) {
        std::string str(size, '\0');
        stream_->read(str.data(), size);
        
        // Null-terminate if needed
        auto null_pos = str.find('\0');
        if (null_pos != std::string::npos) {
            str.resize(null_pos);
        }
        
        return str;
    }
    
    // Container reading
    template<typename Container>
    Container read_container() {
        uint32_t size = read<uint32_t>();
        Container container;
        container.reserve(size);
        
        for (uint32_t i = 0; i < size; ++i) {
            if constexpr (std::is_same_v<typename Container::value_type, std::string>) {
                container.push_back(read_string());
            } else {
                container.push_back(read<typename Container::value_type>());
            }
        }
        
        return container;
    }
    
    // Array reading
    template<typename T, std::size_t N>
    std::array<T, N> read_array() {
        std::array<T, N> arr;
        stream_->read(arr.data(), sizeof(T) * N);
        
        if (endianness_ != Endianness::Native) {
            for (auto& elem : arr) {
                elem = swap_endian(elem);
            }
        }
        
        return arr;
    }
    
    // Skip bytes
    void skip(std::size_t bytes) {
        stream_->seek(bytes, Stream::SeekOrigin::Current);
    }
    
    // Read with validation
    template<typename T>
    T read_with_validation(std::function<bool(const T&)> validator) {
        T value = read<T>();
        if (!validator(value)) {
            throw ValidationError("Value failed validation: " + std::to_string(value));
        }
        return value;
    }
};

class BinaryWriter : public TypedWriter<std::any> {
    std::unique_ptr<Stream> stream_;
    Endianness endianness_;
    
public:
    BinaryWriter(std::unique_ptr<Stream> stream,
                 Endianness endian = Endianness::Native)
        : stream_(std::move(stream)), endianness_(endian) {}
    
    // Primitive type writing
    template<typename T>
    void write(T value) {
        static_assert(std::is_trivially_copyable_v<T>);
        
        if (endianness_ != Endianness::Native) {
            value = swap_endian(value);
        }
        
        stream_->write(&value, sizeof(T));
    }
    
    // String writing
    void write_string(const std::string& str) {
        write<uint32_t>(str.length());
        stream_->write(str.data(), str.length());
    }
    
    void write_fixed_string(const std::string& str, std::size_t size) {
        std::vector<char> buffer(size, '\0');
        std::memcpy(buffer.data(), str.data(), std::min(str.length(), size));
        stream_->write(buffer.data(), size);
    }
    
    // Container writing
    template<typename Container>
    void write_container(const Container& container) {
        write<uint32_t>(container.size());
        
        for (const auto& item : container) {
            if constexpr (std::is_same_v<std::decay_t<decltype(item)>, std::string>) {
                write_string(item);
            } else {
                write(item);
            }
        }
    }
    
    // Padding
    void write_padding(std::size_t bytes, uint8_t value = 0) {
        std::vector<uint8_t> padding(bytes, value);
        stream_->write(padding.data(), bytes);
    }
    
    // Alignment
    void align(std::size_t alignment) {
        std::size_t pos = stream_->tell();
        std::size_t padding = (alignment - (pos % alignment)) % alignment;
        if (padding > 0) {
            write_padding(padding);
        }
    }
};
```
**Why necessary**: Structured binary I/O, cross-platform data exchange, file formats.
**Usage**: Binary file formats, network protocols, serialization.

### `text_reader.hpp` & `text_writer.hpp`
```cpp
class TextReader : public Reader {
    std::unique_ptr<Stream> stream_;
    std::string encoding_;
    std::deque<char> buffer_;
    
public:
    TextReader(std::unique_ptr<Stream> stream, 
               const std::string& encoding = "UTF-8")
        : stream_(std::move(stream)), encoding_(encoding) {}
    
    // Line reading
    std::string read_line() {
        std::string line;
        char ch;
        
        while (read_char(ch)) {
            if (ch == '\n') {
                break;
            }
            if (ch != '\r') {  // Skip carriage returns
                line.push_back(ch);
            }
        }
        
        return line;
    }
    
    // Token reading
    std::string read_token(const std::string& delimiters = " \t\n\r") {
        std::string token;
        char ch;
        
        // Skip leading delimiters
        while (peek_char(ch) && delimiters.find(ch) != std::string::npos) {
            read_char(ch);
        }
        
        // Read until delimiter
        while (read_char(ch) && delimiters.find(ch) == std::string::npos) {
            token.push_back(ch);
        }
        
        return token;
    }
    
    // Parse numbers
    template<typename T>
    T read_number() {
        std::string token = read_token();
        if constexpr (std::is_integral_v<T>) {
            return std::stoll(token);
        } else {
            return std::stod(token);
        }
    }
    
    // Read all text
    std::string read_all() {
        std::stringstream ss;
        std::string line;
        
        while (!eof()) {
            line = read_line();
            ss << line;
            if (!eof()) {
                ss << '\n';
            }
        }
        
        return ss.str();
    }
    
    // CSV-style reading
    std::vector<std::string> read_csv_line(char delimiter = ',', char quote = '"') {
        std::vector<std::string> fields;
        std::string field;
        bool in_quotes = false;
        char ch;
        
        while (read_char(ch)) {
            if (ch == quote) {
                in_quotes = !in_quotes;
            } else if (ch == delimiter && !in_quotes) {
                fields.push_back(field);
                field.clear();
            } else if (ch == '\n' && !in_quotes) {
                break;
            } else {
                field.push_back(ch);
            }
        }
        
        if (!field.empty() || !fields.empty()) {
            fields.push_back(field);
        }
        
        return fields;
    }
    
private:
    bool read_char(char& ch) {
        if (!buffer_.empty()) {
            ch = buffer_.front();
            buffer_.pop_front();
            return true;
        }
        
        return stream_->read(&ch, 1) == 1;
    }
    
    bool peek_char(char& ch) {
        if (buffer_.empty()) {
            if (stream_->read(&ch, 1) == 1) {
                buffer_.push_back(ch);
            } else {
                return false;
            }
        }
        
        ch = buffer_.front();
        return true;
    }
};

class TextWriter : public Writer {
    std::unique_ptr<Stream> stream_;
    std::string encoding_;
    std::string line_ending_;
    int indent_level_ = 0;
    std::string indent_string_ = "    ";
    
public:
    TextWriter(std::unique_ptr<Stream> stream,
               const std::string& encoding = "UTF-8",
               const std::string& line_ending = "\n")
        : stream_(std::move(stream))
        , encoding_(encoding)
        , line_ending_(line_ending) {}
    
    // Basic writing
    void write_string(const std::string& str) {
        stream_->write(str.data(), str.size());
    }
    
    void write_line(const std::string& line = "") {
        write_string(line);
        write_string(line_ending_);
    }
    
    void write_lines(const std::vector<std::string>& lines) {
        for (const auto& line : lines) {
            write_line(line);
        }
    }
    
    // Formatted writing
    template<typename... Args>
    void write_formatted(const std::string& format, Args&&... args) {
        auto formatted = std::vformat(format, std::make_format_args(args...));
        write_string(formatted);
    }
    
    template<typename... Args>
    void write_line_formatted(const std::string& format, Args&&... args) {
        write_formatted(format, std::forward<Args>(args)...);
        write_string(line_ending_);
    }
    
    // Indentation support
    void indent() { ++indent_level_; }
    void unindent() { if (indent_level_ > 0) --indent_level_; }
    
    void write_indented(const std::string& str) {
        for (int i = 0; i < indent_level_; ++i) {
            write_string(indent_string_);
        }
        write_string(str);
    }
    
    void write_line_indented(const std::string& line) {
        write_indented(line);
        write_string(line_ending_);
    }
    
    // CSV-style writing
    void write_csv_line(const std::vector<std::string>& fields,
                       char delimiter = ',',
                       char quote = '"') {
        for (std::size_t i = 0; i < fields.size(); ++i) {
            if (i > 0) {
                stream_->write(&delimiter, 1);
            }
            
            const auto& field = fields[i];
            bool needs_quotes = field.find(delimiter) != std::string::npos ||
                              field.find(quote) != std::string::npos ||
                              field.find('\n') != std::string::npos;
            
            if (needs_quotes) {
                stream_->write(&quote, 1);
                for (char ch : field) {
                    if (ch == quote) {
                        stream_->write(&quote, 1);  // Escape quotes
                    }
                    stream_->write(&ch, 1);
                }
                stream_->write(&quote, 1);
            } else {
                write_string(field);
            }
        }
        
        write_string(line_ending_);
    }
    
    // Table formatting
    void write_table(const std::vector<std::vector<std::string>>& rows,
                    const std::vector<std::size_t>& column_widths = {}) {
        
        // Calculate column widths if not provided
        std::vector<std::size_t> widths = column_widths;
        if (widths.empty() && !rows.empty()) {
            widths.resize(rows[0].size(), 0);
            for (const auto& row : rows) {
                for (std::size_t i = 0; i < row.size(); ++i) {
                    widths[i] = std::max(widths[i], row[i].length());
                }
            }
        }
        
        // Write rows
        for (const auto& row : rows) {
            for (std::size_t i = 0; i < row.size(); ++i) {
                write_formatted("{:<{}}", row[i], widths[i]);
                if (i < row.size() - 1) {
                    write_string(" | ");
                }
            }
            write_line();
        }
    }
    
    void flush() override {
        stream_->flush();
    }
};
```
**Why necessary**: Human-readable I/O, text file processing, formatted output.
**Usage**: Configuration files, log files, reports, CSV data.

### `compression.hpp`
```cpp
class CompressionStream : public Stream {
public:
    enum class Algorithm {
        None,
        Gzip,
        Zlib,
        Bzip2,
        LZ4,
        Zstd
    };
    
private:
    std::unique_ptr<Stream> underlying_;
    Algorithm algorithm_;
    int compression_level_;
    
    // Compression state (algorithm-specific)
    void* compression_state_ = nullptr;
    
    // Buffers
    std::vector<uint8_t> input_buffer_;
    std::vector<uint8_t> output_buffer_;
    
public:
    CompressionStream(std::unique_ptr<Stream> stream,
                     Algorithm algo,
                     int level = -1)  // -1 = default level
        : underlying_(std::move(stream))
        , algorithm_(algo)
        , compression_level_(level)
        , input_buffer_(64 * 1024)
        , output_buffer_(64 * 1024) {
        
        initialize_compression();
    }
    
    ~CompressionStream() {
        flush();
        cleanup_compression();
    }
    
    std::size_t write(const void* data, std::size_t size) override {
        const uint8_t* src = static_cast<const uint8_t*>(data);
        std::size_t total_written = 0;
        
        while (size > 0) {
            std::size_t chunk = std::min(size, input_buffer_.size());
            std::memcpy(input_buffer_.data(), src, chunk);
            
            compress_chunk(input_buffer_.data(), chunk);
            
            src += chunk;
            size -= chunk;
            total_written += chunk;
        }
        
        return total_written;
    }
    
    std::size_t read(void* buffer, std::size_t size) override {
        uint8_t* dest = static_cast<uint8_t*>(buffer);
        std::size_t total_read = 0;
        
        while (size > 0 && !eof()) {
            std::size_t decompressed = decompress_chunk(dest, size);
            
            if (decompressed == 0) {
                break;  // No more data
            }
            
            dest += decompressed;
            size -= decompressed;
            total_read += decompressed;
        }
        
        return total_read;
    }
    
    void flush() override {
        finish_compression();
        underlying_->flush();
    }
    
    // Compression statistics
    struct Statistics {
        std::size_t uncompressed_bytes;
        std::size_t compressed_bytes;
        double compression_ratio;
        std::chrono::nanoseconds compression_time;
    };
    
    Statistics get_statistics() const {
        // Return compression statistics
    }
    
private:
    void initialize_compression();
    void cleanup_compression();
    void compress_chunk(const uint8_t* data, std::size_t size);
    std::size_t decompress_chunk(uint8_t* buffer, std::size_t size);
    void finish_compression();
};

// Factory for compression streams
class CompressionFactory {
public:
    static std::unique_ptr<Stream> create_compressor(
        std::unique_ptr<Stream> stream,
        CompressionStream::Algorithm algorithm,
        int level = -1) {
        
        return std::make_unique<CompressionStream>(
            std::move(stream), algorithm, level
        );
    }
    
    static std::unique_ptr<Stream> create_decompressor(
        std::unique_ptr<Stream> stream,
        CompressionStream::Algorithm algorithm) {
        
        return std::make_unique<CompressionStream>(
            std::move(stream), algorithm, 0
        );
    }
    
    // Auto-detect compression from magic bytes
    static CompressionStream::Algorithm detect_compression(Stream& stream) {
        uint8_t magic[4];
        std::size_t read = stream.read(magic, sizeof(magic));
        stream.seek(-static_cast<std::ptrdiff_t>(read), Stream::SeekOrigin::Current);
        
        if (read >= 2) {
            if (magic[0] == 0x1f && magic[1] == 0x8b) return CompressionStream::Algorithm::Gzip;
            if (magic[0] == 0x78 && (magic[1] == 0x01 || magic[1] == 0x9c || magic[1] == 0xda)) {
                return CompressionStream::Algorithm::Zlib;
            }
            if (magic[0] == 0x42 && magic[1] == 0x5a) return CompressionStream::Algorithm::Bzip2;
        }
        
        if (read >= 4) {
            if (memcmp(magic, "\x04\x22\x4d\x18", 4) == 0) {
                return CompressionStream::Algorithm::LZ4;
            }
            if (memcmp(magic, "\x28\xb5\x2f\xfd", 4) == 0) {
                return CompressionStream::Algorithm::Zstd;
            }
        }
        
        return CompressionStream::Algorithm::None;
    }
};
```
**Why necessary**: Transparent compression, bandwidth reduction, storage optimization.
**Usage**: Compressed file formats, network protocols, backup systems.

### `stream_utils.hpp`
```cpp
class StreamCopier {
public:
    static std::size_t copy(Stream& source, Stream& destination,
                           std::size_t buffer_size = 8192) {
        std::vector<uint8_t> buffer(buffer_size);
        std::size_t total_copied = 0;
        
        while (!source.eof()) {
            std::size_t read = source.read(buffer.data(), buffer_size);
            if (read == 0) break;
            
            destination.write(buffer.data(), read);
            total_copied += read;
        }
        
        return total_copied;
    }
    
    static std::size_t copy_n(Stream& source, Stream& destination,
                             std::size_t bytes,
                             std::size_t buffer_size = 8192) {
        std::vector<uint8_t> buffer(buffer_size);
        std::size_t total_copied = 0;
        
        while (total_copied < bytes && !source.eof()) {
            std::size_t to_read = std::min(buffer_size, bytes - total_copied);
            std::size_t read = source.read(buffer.data(), to_read);
            if (read == 0) break;
            
            destination.write(buffer.data(), read);
            total_copied += read;
        }
        
        return total_copied;
    }
    
    // Async copy
    static std::future<std::size_t> copy_async(Stream& source, Stream& destination) {
        return std::async(std::launch::async, [&source, &destination]() {
            return copy(source, destination);
        });
    }
    
    // Copy with progress callback
    static std::size_t copy_with_progress(
        Stream& source, 
        Stream& destination,
        std::function<void(std::size_t bytes_copied, std::size_t total_bytes)> progress_callback,
        std::size_t buffer_size = 8192) {
        
        std::size_t total_size = source.size();
        std::vector<uint8_t> buffer(buffer_size);
        std::size_t total_copied = 0;
        
        while (!source.eof()) {
            std::size_t read = source.read(buffer.data(), buffer_size);
            if (read == 0) break;
            
            destination.write(buffer.data(), read);
            total_copied += read;
            
            if (progress_callback) {
                progress_callback(total_copied, total_size);
            }
        }
        
        return total_copied;
    }
};

// Stream splitter (tee)
class StreamSplitter : public Stream {
    std::vector<std::shared_ptr<Stream>> outputs_;
    
public:
    void add_output(std::shared_ptr<Stream> stream) {
        outputs_.push_back(stream);
    }
    
    std::size_t write(const void* data, std::size_t size) override {
        for (auto& output : outputs_) {
            output->write(data, size);
        }
        return size;
    }
    
    void flush() override {
        for (auto& output : outputs_) {
            output->flush();
        }
    }
};

// Stream merger
class StreamMerger : public Stream {
    std::vector<std::shared_ptr<Stream>> inputs_;
    std::size_t current_input_ = 0;
    
public:
    void add_input(std::shared_ptr<Stream> stream) {
        inputs_.push_back(stream);
    }
    
    std::size_t read(void* buffer, std::size_t size) override {
        while (current_input_ < inputs_.size()) {
            std::size_t read = inputs_[current_input_]->read(buffer, size);
            
            if (read > 0) {
                return read;
            }
            
            // Move to next input
            ++current_input_;
        }
        
        return 0;  // All inputs exhausted
    }
    
    bool eof() const override {
        return current_input_ >= inputs_.size();
    }
};

// Rate limiter
class RateLimitedStream : public Stream {
    std::unique_ptr<Stream> underlying_;
    std::size_t bytes_per_second_;
    std::chrono::steady_clock::time_point last_operation_;
    std::size_t bytes_since_last_sleep_ = 0;
    
public:
    RateLimitedStream(std::unique_ptr<Stream> stream, std::size_t bytes_per_second)
        : underlying_(std::move(stream))
        , bytes_per_second_(bytes_per_second)
        , last_operation_(std::chrono::steady_clock::now()) {}
    
    std::size_t read(void* buffer, std::size_t size) override {
        throttle(size);
        return underlying_->read(buffer, size);
    }
    
    std::size_t write(const void* data, std::size_t size) override {
        throttle(size);
        return underlying_->write(data, size);
    }
    
private:
    void throttle(std::size_t bytes) {
        bytes_since_last_sleep_ += bytes;
        
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_operation_
        );
        
        auto expected_time = std::chrono::milliseconds(
            bytes_since_last_sleep_ * 1000 / bytes_per_second_
        );
        
        if (expected_time > elapsed) {
            std::this_thread::sleep_for(expected_time - elapsed);
            bytes_since_last_sleep_ = 0;
            last_operation_ = std::chrono::steady_clock::now();
        }
    }
};
```
**Why necessary**: Stream manipulation, I/O utilities, bandwidth control.
**Usage**: File copying, stream processing, network throttling.

## I/O Patterns

### Buffered File Reading
```cpp
void process_large_file(const std::string& filename) {
    auto file = std::make_unique<FileStream>(filename, Stream::Mode::Read);
    BufferedStream buffered(std::move(file), 64 * 1024);  // 64KB buffer
    
    TextReader reader(std::make_unique<BufferedStream>(std::move(buffered)));
    
    std::string line;
    while (!reader.eof()) {
        line = reader.read_line();
        process_line(line);
    }
}
```

### Async Network Communication
```cpp
class NetworkClient {
    IOService io_service_;
    std::unique_ptr<AsyncReader> reader_;
    std::unique_ptr<AsyncWriter> writer_;
    
public:
    void send_request(const Request& request) {
        auto data = serialize(request);
        
        writer_->async_write(data).then([this](auto result) {
            if (result.has_value()) {
                handle_write_complete();
            } else {
                handle_error(result.error());
            }
        });
    }
    
    void receive_response() {
        reader_->async_read_until('\n').then([this](auto data) {
            auto response = deserialize(data);
            handle_response(response);
            
            // Continue reading
            receive_response();
        });
    }
};
```

### Compressed File Writing
```cpp
void write_compressed_data(const std::vector<Data>& data) {
    auto file = std::make_unique<FileStream>("output.gz", Stream::Mode::Write);
    auto compressed = CompressionFactory::create_compressor(
        std::move(file), 
        CompressionStream::Algorithm::Gzip,
        9  // Maximum compression
    );
    
    BinaryWriter writer(std::move(compressed));
    
    for (const auto& item : data) {
        writer.write(item);
    }
}
```

### Stream Pipeline
```cpp
void process_pipeline() {
    // Input -> Decompress -> Transform -> Compress -> Output
    auto input = std::make_unique<FileStream>("input.gz", Stream::Mode::Read);
    auto decompressed = CompressionFactory::create_decompressor(
        std::move(input), CompressionStream::Algorithm::Gzip
    );
    
    auto transformed = std::make_unique<TransformStream>(
        std::move(decompressed),
        [](const std::vector<uint8_t>& data) {
            return transform_data(data);
        }
    );
    
    auto output = std::make_unique<FileStream>("output.bz2", Stream::Mode::Write);
    auto compressed = CompressionFactory::create_compressor(
        std::move(output), CompressionStream::Algorithm::Bzip2
    );
    
    StreamCopier::copy(*transformed, *compressed);
}
```

## Performance Considerations

- **Buffer sizes**: Typically 4KB-64KB for optimal performance
- **Async operations**: Reduce blocking, improve throughput
- **Zero-copy**: Use views and move semantics
- **Memory-mapped I/O**: For large files
- **Vectored I/O**: Reduce system calls with scatter-gather

## Testing Strategy

- **Stream implementations**: Test all stream types
- **Buffer correctness**: Edge cases in buffering
- **Async operations**: Concurrent I/O testing
- **Error handling**: I/O failure scenarios
- **Performance**: Throughput and latency benchmarks

## Usage Guidelines

1. **Choose appropriate buffering**: Match buffer size to use case
2. **Prefer async for network**: Non-blocking for responsiveness
3. **Use compression wisely**: Balance CPU vs I/O
4. **Handle errors gracefully**: Always check I/O results
5. **Clean up resources**: Use RAII for streams

## Anti-patterns to Avoid

- Unbuffered small reads/writes
- Blocking I/O in UI threads
- Ignoring error codes
- Manual resource management
- Mixing text and binary modes

## Dependencies
- `base/` - For base patterns
- `error/` - For error handling
- `concurrency/` - For async operations
- Standard library (C++20)
- Platform APIs (POSIX, Windows)

## Future Enhancements
- io_uring support (Linux)
- Memory-mapped I/O abstractions
- Direct I/O support
- RDMA support
- WebRTC data channels