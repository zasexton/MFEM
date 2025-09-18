# Core Filesystem - AGENT.md

## Purpose
The `filesystem/` layer provides platform-independent file system operations, path manipulation, resource management, and integration points that sit on top of the core `io/` stream abstractions. It extends the standard filesystem library with utilities for monitoring, caching, and virtual file systems while delegating byte-level stream implementation to `io/`.

## Architecture Philosophy
- **Platform independence**: Abstract OS differences
- **RAII resource management**: Automatic cleanup
- **Virtual file systems**: Support for archives, memory FS
- **Async I/O support**: Non-blocking file operations
- **Resource abstraction**: Uniform interface for files/archives/network

## Files Overview

### Path Management
```cpp
path.hpp              // Enhanced path operations
path_utils.hpp        // Path manipulation utilities
glob.hpp              // File pattern matching
path_resolver.hpp     // Path resolution and canonicalization
path_watcher.hpp      // Path change monitoring
```

### File Operations
```cpp
file.hpp              // File abstraction
file_utils.hpp        // File manipulation utilities
file_reader.hpp       // Efficient file reading
file_writer.hpp       // Buffered file writing
temp_file.hpp         // Temporary file management
file_lock.hpp         // File locking mechanisms
```

### Directory Operations
```cpp
directory.hpp         // Directory abstraction
directory_iterator.hpp // Directory traversal
directory_watcher.hpp // Directory monitoring
directory_utils.hpp   // Directory utilities
```

### Virtual File Systems
```cpp
vfs.hpp              // Virtual filesystem interface
memory_fs.hpp        // In-memory filesystem
archive_fs.hpp       // Archive file systems (zip, tar)
overlay_fs.hpp       // Layered filesystems
mount_point.hpp      // Filesystem mounting
```

### Resource Management
```cpp
resource_manager.hpp  // Resource loading and caching
resource_loader.hpp   // Resource loader interface
resource_cache.hpp    // Resource caching
resource_handle.hpp   // Resource references
resource_pack.hpp     // Resource bundling
```

### Utilities
```cpp
file_monitor.hpp     // File change detection
file_hash.hpp        // File hashing utilities
file_metadata.hpp    // Extended file metadata
file_permissions.hpp // Permission management
disk_usage.hpp       // Disk space monitoring
```

> **Stream Usage**: File and archive loading rely on the shared `fem::core::io` stream hierarchy; this module focuses on discovering paths and orchestrating resources, not defining new stream types.

## Detailed Component Specifications

### `path.hpp`
```cpp
class Path {
    std::filesystem::path path_;
    
public:
    Path() = default;
    Path(const std::string& str);
    Path(const char* str);
    Path(const std::filesystem::path& p);
    
    // Path operations
    Path parent() const;
    Path filename() const;
    Path stem() const;
    Path extension() const;
    
    // Path manipulation
    Path& append(const Path& p);
    Path& concat(const std::string& str);
    Path& replace_extension(const std::string& ext);
    Path& make_absolute();
    Path& normalize();
    
    // Queries
    bool is_absolute() const;
    bool is_relative() const;
    bool exists() const;
    bool is_file() const;
    bool is_directory() const;
    bool is_symlink() const;
    
    // Conversion
    std::string string() const;
    std::string native_string() const;
    const std::filesystem::path& std_path() const { return path_; }
    
    // Operators
    Path operator/(const Path& rhs) const;
    Path operator/(const std::string& rhs) const;
    bool operator==(const Path& rhs) const;
    bool operator<(const Path& rhs) const;
    
    // Static utilities
    static Path current_directory();
    static Path home_directory();
    static Path temp_directory();
    static Path executable_path();
    
    // Platform-specific
    static char separator();
    static std::vector<Path> search_paths();
};

// Path utilities
namespace path_utils {
    Path relative(const Path& from, const Path& to);
    Path common_parent(const Path& p1, const Path& p2);
    bool is_parent_of(const Path& parent, const Path& child);
    std::vector<Path> split(const Path& p);
    Path join(const std::vector<Path>& parts);
}
```
**Why necessary**: Cross-platform path handling, convenient path operations.
**Usage**: File operations, resource loading, configuration paths.

### `file.hpp`
```cpp
class File {
    std::fstream stream_;
    Path path_;
    size_t size_;
    
public:
    enum class Mode {
        Read,
        Write,
        ReadWrite,
        Append
    };
    
    enum class Type {
        Text,
        Binary
    };
    
    File() = default;
    File(const Path& path, Mode mode = Mode::Read, Type type = Type::Binary);
    ~File();
    
    // File operations
    bool open(const Path& path, Mode mode, Type type);
    void close();
    bool is_open() const;
    
    // Reading
    std::vector<uint8_t> read_all();
    std::string read_text();
    std::vector<std::string> read_lines();
    size_t read(void* buffer, size_t size);
    
    template<typename T>
    T read() {
        T value;
        read(&value, sizeof(T));
        return value;
    }
    
    // Writing
    void write(const void* data, size_t size);
    void write(const std::vector<uint8_t>& data);
    void write(const std::string& text);
    void write_lines(const std::vector<std::string>& lines);
    
    template<typename T>
    void write(const T& value) {
        write(&value, sizeof(T));
    }
    
    // Seeking
    void seek(size_t position);
    void seek_relative(int64_t offset);
    void seek_end();
    size_t tell() const;
    
    // Properties
    size_t size() const;
    Path path() const { return path_; }
    bool eof() const;
    
    // Utilities
    void flush();
    void truncate(size_t new_size);
    
    // Static utilities
    static bool exists(const Path& path);
    static bool remove(const Path& path);
    static bool copy(const Path& from, const Path& to);
    static bool move(const Path& from, const Path& to);
    static size_t size(const Path& path);
    static std::string read_text_file(const Path& path);
    static void write_text_file(const Path& path, const std::string& content);
};

// RAII file handle
class FileHandle {
    File* file_;
public:
    explicit FileHandle(File* f) : file_(f) {}
    ~FileHandle() { if (file_) file_->close(); }
    File* operator->() { return file_; }
    File& operator*() { return *file_; }
};
```
**Why necessary**: Unified file operations, RAII management, convenience methods.
**Usage**: File I/O, data persistence, configuration loading.

### `resource_manager.hpp`
```cpp
template<typename Resource>
class ResourceManager : public Singleton<ResourceManager<Resource>> {
    struct ResourceEntry {
        std::shared_ptr<Resource> resource;
        Path path;
        std::chrono::time_point<std::chrono::steady_clock> last_accessed;
        size_t access_count;
        bool persistent;
    };
    
    std::unordered_map<std::string, ResourceEntry> resources_;
    std::unordered_map<std::string, std::function<std::shared_ptr<Resource>(const Path&)>> loaders_;
    mutable std::shared_mutex mutex_;
    
    struct Config {
        size_t max_cached = 100;
        std::chrono::seconds cache_duration{300};
        bool auto_reload = false;
        std::vector<Path> search_paths;
    } config_;
    
public:
    // Resource loading
    std::shared_ptr<Resource> load(const std::string& name, const Path& path) {
        std::unique_lock lock(mutex_);
        
        auto it = resources_.find(name);
        if (it != resources_.end()) {
            it->second.last_accessed = std::chrono::steady_clock::now();
            it->second.access_count++;
            return it->second.resource;
        }
        
        auto resource = load_resource(path);
        resources_[name] = {resource, path, std::chrono::steady_clock::now(), 1, false};
        
        cleanup_cache();
        return resource;
    }
    
    std::shared_ptr<Resource> get(const std::string& name) {
        std::shared_lock lock(mutex_);
        auto it = resources_.find(name);
        return it != resources_.end() ? it->second.resource : nullptr;
    }
    
    // Async loading
    std::future<std::shared_ptr<Resource>> load_async(const std::string& name, const Path& path) {
        return std::async(std::launch::async, [this, name, path] {
            return load(name, path);
        });
    }
    
    // Resource management
    void reload(const std::string& name);
    void unload(const std::string& name);
    void unload_all();
    void mark_persistent(const std::string& name);
    
    // Loader registration
    void register_loader(const std::string& extension, 
                        std::function<std::shared_ptr<Resource>(const Path&)> loader) {
        loaders_[extension] = loader;
    }
    
    // Configuration
    void set_search_paths(const std::vector<Path>& paths) {
        config_.search_paths = paths;
    }
    
    void set_cache_size(size_t max_cached) {
        config_.max_cached = max_cached;
    }
    
    // Statistics
    size_t cached_count() const;
    size_t memory_usage() const;
    std::vector<std::string> cached_resources() const;
    
private:
    std::shared_ptr<Resource> load_resource(const Path& path);
    void cleanup_cache();
    Path resolve_path(const std::string& name);
};

// Specialized resource managers
using TextureManager = ResourceManager<Texture>;
using ShaderManager = ResourceManager<Shader>;
using ConfigManager = ResourceManager<Config>;
```
**Why necessary**: Centralized resource loading, caching, memory management.
**Usage**: Asset management, configuration loading, data files.

### `vfs.hpp`
```cpp
class VirtualFileSystem {
public:
    virtual ~VirtualFileSystem() = default;
    
    // File operations
    virtual bool exists(const Path& path) const = 0;
    virtual bool is_file(const Path& path) const = 0;
    virtual bool is_directory(const Path& path) const = 0;
    
    virtual std::unique_ptr<Stream> open(const Path& path, 
                                         File::Mode mode) = 0;
    virtual std::vector<uint8_t> read_file(const Path& path) = 0;
    virtual void write_file(const Path& path, 
                           const std::vector<uint8_t>& data) = 0;
    
    // Directory operations
    virtual std::vector<Path> list_directory(const Path& path) const = 0;
    virtual bool create_directory(const Path& path) = 0;
    virtual bool remove_directory(const Path& path) = 0;
    
    // File management
    virtual bool remove_file(const Path& path) = 0;
    virtual bool rename(const Path& from, const Path& to) = 0;
    virtual bool copy(const Path& from, const Path& to) = 0;
    
    // Metadata
    virtual size_t file_size(const Path& path) const = 0;
    virtual std::chrono::system_clock::time_point 
        last_modified(const Path& path) const = 0;
};

// Layered filesystem
class OverlayFileSystem : public VirtualFileSystem {
    std::vector<std::unique_ptr<VirtualFileSystem>> layers_;
    
public:
    void add_layer(std::unique_ptr<VirtualFileSystem> layer);
    void remove_layer(size_t index);
    
    // Searches layers in order
    bool exists(const Path& path) const override;
    std::unique_ptr<Stream> open(const Path& path, File::Mode mode) override;
    // ... implements all VFS methods
};

// Memory filesystem
class MemoryFileSystem : public VirtualFileSystem {
    struct FileNode {
        std::vector<uint8_t> data;
        std::chrono::system_clock::time_point modified;
    };
    
    struct DirectoryNode {
        std::unordered_map<std::string, std::variant<FileNode, DirectoryNode>> children;
    };
    
    DirectoryNode root_;
    
public:
    bool exists(const Path& path) const override;
    std::unique_ptr<Stream> open(const Path& path, File::Mode mode) override;
    // ... implements all VFS methods
};
```
**Why necessary**: Abstract file sources, support archives, testing.
**Usage**: Resource packs, mod systems, unit testing.

### `file_watcher.hpp`
```cpp
class FileWatcher {
public:
    enum class Event {
        Created,
        Modified,
        Deleted,
        Renamed,
        AttributeChanged
    };
    
    using Callback = std::function<void(const Path&, Event)>;
    
private:
    std::thread watcher_thread_;
    std::atomic<bool> running_{true};
    std::unordered_map<Path, Callback> watches_;
    
    // Platform-specific handle
#ifdef _WIN32
    HANDLE dir_handle_;
#else
    int inotify_fd_;
#endif
    
public:
    FileWatcher();
    ~FileWatcher();
    
    void watch(const Path& path, Callback callback);
    void unwatch(const Path& path);
    void clear();
    
    // Polling fallback for platforms without native support
    void poll();
    
private:
    void watch_thread();
    void process_events();
};

// Convenience class for single file
class FileMonitor {
    Path path_;
    std::function<void()> on_change_;
    std::chrono::system_clock::time_point last_modified_;
    
public:
    FileMonitor(const Path& path, std::function<void()> callback);
    
    void check();  // Manual check
    void start_auto_check(std::chrono::milliseconds interval);
    void stop_auto_check();
};
```
**Why necessary**: File change detection, hot reloading, cache invalidation.
**Usage**: Configuration reloading, asset hot-reload, file synchronization.

### `stream.hpp`
```cpp
class Stream {
public:
    virtual ~Stream() = default;
    
    // Reading
    virtual size_t read(void* buffer, size_t size) = 0;
    virtual std::vector<uint8_t> read_all() = 0;
    virtual std::string read_line() = 0;
    
    // Writing
    virtual size_t write(const void* data, size_t size) = 0;
    virtual void write_line(const std::string& line) = 0;
    virtual void flush() = 0;
    
    // Positioning
    virtual void seek(size_t position) = 0;
    virtual size_t tell() const = 0;
    virtual size_t size() const = 0;
    
    // Status
    virtual bool eof() const = 0;
    virtual bool good() const = 0;
    virtual bool is_readable() const = 0;
    virtual bool is_writable() const = 0;
    
    // Typed I/O
    template<typename T>
    T read() {
        T value;
        read(&value, sizeof(T));
        return value;
    }
    
    template<typename T>
    void write(const T& value) {
        write(&value, sizeof(T));
    }
};

// Buffered stream wrapper
class BufferedStream : public Stream {
    std::unique_ptr<Stream> underlying_;
    std::vector<uint8_t> read_buffer_;
    std::vector<uint8_t> write_buffer_;
    size_t read_pos_ = 0;
    size_t write_pos_ = 0;
    
public:
    BufferedStream(std::unique_ptr<Stream> stream, 
                   size_t buffer_size = 8192);
    
    // Implements Stream interface with buffering
    size_t read(void* buffer, size_t size) override;
    size_t write(const void* data, size_t size) override;
    void flush() override;
};
```
**Why necessary**: Uniform I/O interface, stream composition, buffering.
**Usage**: File I/O, network I/O, compression, encryption.

## Filesystem Patterns

### Resource Loading Pattern
```cpp
auto& manager = ResourceManager<Config>::instance();
manager.register_loader(".json", [](const Path& p) {
    return std::make_shared<Config>(Config::from_json(File::read_text_file(p)));
});

auto config = manager.load("app_config", "config/app.json");
```

### Virtual Filesystem Usage
```cpp
auto vfs = std::make_unique<OverlayFileSystem>();
vfs->add_layer(std::make_unique<MemoryFileSystem>());  // RAM cache
vfs->add_layer(std::make_unique<ArchiveFileSystem>("assets.zip"));  // Archive
vfs->add_layer(std::make_unique<NativeFileSystem>());  // Disk

// Read through layers
auto data = vfs->read_file("textures/player.png");
```

### File Watching
```cpp
FileWatcher watcher;
watcher.watch("config/", [](const Path& p, FileWatcher::Event e) {
    if (e == FileWatcher::Event::Modified) {
        reload_config(p);
    }
});
```

### Temporary Files
```cpp
{
    TempFile temp("data_XXXXXX");
    temp.write(process_data());
    
    // Use temp file
    external_program(temp.path());
    
}  // File automatically deleted
```

## Performance Considerations

- **Memory mapping**: For large files
- **Buffered I/O**: Reduce system calls
- **Async I/O**: Non-blocking operations
- **Path caching**: Avoid repeated resolution
- **Resource pooling**: Reuse file handles

## Testing Strategy

- **Cross-platform testing**: Windows, Linux, macOS
- **Permission testing**: Read-only, restricted access
- **Large file testing**: >4GB files
- **Concurrent access**: Multiple readers/writers
- **Error handling**: Disk full, network paths

## Usage Guidelines

1. **Use RAII**: Automatic resource cleanup
2. **Check errors**: Always verify operations succeeded
3. **Buffer I/O**: Use buffered streams for small reads/writes
4. **Cache resources**: Avoid reloading unchanged files
5. **Watch for changes**: Use file monitoring for hot-reload

## Anti-patterns to Avoid

- Opening files without closing
- Hardcoded paths
- Synchronous I/O in UI thread
- Not handling missing files
- Excessive file system queries

## Dependencies
- `base/` - For Object patterns
- `error/` - For error handling
- `memory/` - For memory management
- Standard library (C++20)
- Platform APIs (Windows API, POSIX)

## Future Enhancements
- Cloud storage support
- Distributed file systems
- File versioning
- Compression/encryption streams
- Database-backed VFS
