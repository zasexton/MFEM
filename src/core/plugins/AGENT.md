# Core Plugins - AGENT.md

## Purpose
The `plugins/` layer provides a comprehensive plugin system enabling runtime extensibility through dynamically loaded libraries. It supports plugin discovery, dependency management, versioning, sandboxing, and hot-reloading while maintaining type safety and ABI stability across plugin boundaries.

## Architecture Philosophy
- **ABI stability**: Stable binary interface across versions
- **Dependency management**: Automatic resolution and loading
- **Hot-reloading**: Update plugins without restart
- **Sandboxing**: Isolated plugin execution environments
- **Type safety**: Strong typing across plugin boundaries

## Files Overview

### Core Plugin System
```cpp
plugin.hpp           // Plugin interface and base class
plugin_manager.hpp   // Plugin lifecycle management
plugin_loader.hpp    // Dynamic library loading
plugin_registry.hpp  // Plugin registration and discovery
plugin_info.hpp      // Plugin metadata and versioning
```

### Dependency Management
```cpp
dependency.hpp       // Dependency specification
dependency_resolver.hpp // Dependency graph resolution
version.hpp          // Semantic versioning
compatibility.hpp    // ABI compatibility checking
plugin_repository.hpp // Plugin repository management
```

### Plugin Communication
```cpp
plugin_interface.hpp // Stable API interfaces
service_locator.hpp  // Service discovery
message_bus.hpp      // Inter-plugin messaging
shared_context.hpp   // Shared data between plugins
plugin_events.hpp    // Plugin event system
```

### Loading & Lifecycle
```cpp
loader_factory.hpp   // Platform-specific loaders
hot_reload.hpp       // Hot-reloading support
plugin_sandbox.hpp   // Sandboxed execution
plugin_cache.hpp     // Plugin caching
lazy_loader.hpp      // Lazy plugin loading
```

### Security & Validation
```cpp
plugin_validator.hpp // Plugin validation
signature.hpp        // Digital signatures
permissions.hpp      // Plugin permissions
capability.hpp       // Capability-based security
audit.hpp           // Plugin audit logging
```

### Utilities
```cpp
plugin_macros.hpp    // Helper macros
plugin_discovery.hpp // Plugin discovery mechanisms
plugin_manifest.hpp  // Manifest parsing
plugin_bundle.hpp    // Plugin packaging
plugin_tools.hpp     // Development tools
```

## Detailed Component Specifications

### `plugin.hpp`
```cpp
// Stable plugin interface - never change this after v1.0!
class IPlugin {
public:
    virtual ~IPlugin() = default;
    
    // Plugin identification
    virtual const char* name() const = 0;
    virtual const char* version() const = 0;
    virtual const char* author() const = 0;
    virtual const char* description() const = 0;
    virtual const char* license() const = 0;
    
    // Plugin lifecycle
    virtual bool initialize(IPluginContext* context) = 0;
    virtual bool start() = 0;
    virtual void stop() = 0;
    virtual void shutdown() = 0;
    
    // Plugin capabilities
    virtual uint32_t get_api_version() const = 0;
    virtual const char** get_required_plugins() const = 0;
    virtual const char** get_optional_plugins() const = 0;
    
    // Service registration
    virtual void register_services(IServiceRegistry* registry) = 0;
    virtual void* get_service(const char* service_name) = 0;
    
    // Configuration
    virtual bool configure(const char* config_json) = 0;
    virtual const char* get_default_config() const = 0;
    
    // Status and health
    enum class Status {
        Unloaded,
        Loaded,
        Initialized,
        Started,
        Stopped,
        Failed,
        Disabled
    };
    
    virtual Status get_status() const = 0;
    virtual const char* get_status_message() const = 0;
    virtual bool is_healthy() const = 0;
};

// Plugin context for accessing host services
class IPluginContext {
public:
    virtual ~IPluginContext() = default;
    
    // Host information
    virtual const char* get_host_version() const = 0;
    virtual const char* get_host_name() const = 0;
    
    // Service access
    virtual void* get_host_service(const char* service_name) = 0;
    virtual void* get_plugin_service(const char* plugin_name, const char* service_name) = 0;
    
    // Resource access
    virtual const char* get_plugin_directory() const = 0;
    virtual const char* get_data_directory() const = 0;
    virtual const char* get_config_directory() const = 0;
    
    // Logging
    virtual void log(LogLevel level, const char* message) = 0;
    
    // Events
    virtual void emit_event(const char* event_name, const void* data) = 0;
    virtual void subscribe_event(const char* event_name, EventHandler handler) = 0;
    
    // Permissions
    virtual bool has_permission(const char* permission) const = 0;
    virtual bool request_permission(const char* permission) = 0;
};

// Base implementation with common functionality
class PluginBase : public IPlugin {
protected:
    struct Metadata {
        std::string name;
        std::string version;
        std::string author;
        std::string description;
        std::string license;
        std::vector<std::string> required_plugins;
        std::vector<std::string> optional_plugins;
    } metadata_;
    
    IPluginContext* context_ = nullptr;
    Status status_ = Status::Unloaded;
    std::string status_message_;
    
public:
    // IPlugin implementation
    const char* name() const override { return metadata_.name.c_str(); }
    const char* version() const override { return metadata_.version.c_str(); }
    const char* author() const override { return metadata_.author.c_str(); }
    const char* description() const override { return metadata_.description.c_str(); }
    const char* license() const override { return metadata_.license.c_str(); }
    
    Status get_status() const override { return status_; }
    const char* get_status_message() const override { return status_message_.c_str(); }
    
    uint32_t get_api_version() const override {
        return PLUGIN_API_VERSION;
    }
    
    bool initialize(IPluginContext* context) override {
        context_ = context;
        status_ = Status::Initialized;
        return true;
    }
    
    bool start() override {
        if (status_ != Status::Initialized && status_ != Status::Stopped) {
            return false;
        }
        status_ = Status::Started;
        return on_start();
    }
    
    void stop() override {
        if (status_ == Status::Started) {
            on_stop();
            status_ = Status::Stopped;
        }
    }
    
    void shutdown() override {
        stop();
        on_shutdown();
        status_ = Status::Unloaded;
    }
    
protected:
    // Override these in derived classes
    virtual bool on_start() { return true; }
    virtual void on_stop() {}
    virtual void on_shutdown() {}
    
    // Helper methods
    void log_info(const std::string& message) {
        if (context_) {
            context_->log(LogLevel::Info, message.c_str());
        }
    }
    
    void log_error(const std::string& message) {
        if (context_) {
            context_->log(LogLevel::Error, message.c_str());
        }
        status_message_ = message;
    }
};

// Plugin API version for ABI compatibility
#define PLUGIN_API_VERSION 1

// Plugin export macros
#ifdef _WIN32
    #define PLUGIN_EXPORT extern "C" __declspec(dllexport)
#else
    #define PLUGIN_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// Required plugin entry points
#define PLUGIN_ENTRY_POINTS(PluginClass) \
    PLUGIN_EXPORT IPlugin* create_plugin() { \
        return new PluginClass(); \
    } \
    PLUGIN_EXPORT void destroy_plugin(IPlugin* plugin) { \
        delete plugin; \
    } \
    PLUGIN_EXPORT uint32_t get_plugin_api_version() { \
        return PLUGIN_API_VERSION; \
    }
```
**Why necessary**: Stable plugin interface, lifecycle management, cross-platform compatibility.
**Usage**: All plugins implement this interface, ensures ABI stability.

### `plugin_manager.hpp`
```cpp
class PluginManager : public Singleton<PluginManager> {
public:
    using PluginPtr = std::shared_ptr<IPlugin>;
    using LoaderPtr = std::unique_ptr<PluginLoader>;
    
    struct PluginEntry {
        PluginPtr plugin;
        LoaderPtr loader;
        std::filesystem::path path;
        PluginInfo info;
        std::vector<std::string> provided_services;
        std::chrono::steady_clock::time_point load_time;
        bool is_hot_reloadable = false;
        std::filesystem::file_time_type last_modified;
    };
    
private:
    std::unordered_map<std::string, PluginEntry> plugins_;
    std::unique_ptr<DependencyResolver> dependency_resolver_;
    std::unique_ptr<ServiceLocator> service_locator_;
    std::unique_ptr<PluginContext> global_context_;
    std::unique_ptr<PluginValidator> validator_;
    
    std::vector<std::filesystem::path> plugin_paths_;
    std::thread hot_reload_thread_;
    std::atomic<bool> hot_reload_enabled_{false};
    
    // Plugin filters and policies
    std::function<bool(const PluginInfo&)> load_filter_;
    std::function<void(const std::string&, const std::exception&)> error_handler_;
    
    // Events
    Signal<const std::string&> plugin_loaded_;
    Signal<const std::string&> plugin_unloaded_;
    Signal<const std::string&, const std::string&> plugin_reloaded_;
    
public:
    // Plugin discovery and loading
    void add_plugin_directory(const std::filesystem::path& dir) {
        plugin_paths_.push_back(dir);
    }
    
    void discover_plugins() {
        for (const auto& dir : plugin_paths_) {
            if (!std::filesystem::exists(dir)) continue;
            
            for (const auto& entry : std::filesystem::recursive_directory_iterator(dir)) {
                if (is_plugin_file(entry.path())) {
                    try {
                        auto info = read_plugin_info(entry.path());
                        if (!load_filter_ || load_filter_(info)) {
                            register_plugin(entry.path(), info);
                        }
                    } catch (const std::exception& e) {
                        handle_error(entry.path().string(), e);
                    }
                }
            }
        }
    }
    
    bool load_plugin(const std::string& name) {
        auto it = plugins_.find(name);
        if (it == plugins_.end()) {
            return false;
        }
        
        auto& entry = it->second;
        if (entry.plugin && entry.plugin->get_status() != IPlugin::Status::Unloaded) {
            return true;  // Already loaded
        }
        
        // Check dependencies
        auto deps = dependency_resolver_->resolve(name);
        for (const auto& dep : deps) {
            if (!load_plugin(dep)) {
                log_error("Failed to load dependency: " + dep);
                return false;
            }
        }
        
        // Load the plugin
        try {
            entry.loader = create_loader(entry.path);
            entry.plugin = entry.loader->load();
            
            // Validate plugin
            if (validator_ && !validator_->validate(entry.plugin.get(), entry.info)) {
                throw std::runtime_error("Plugin validation failed");
            }
            
            // Initialize plugin
            if (!entry.plugin->initialize(global_context_.get())) {
                throw std::runtime_error("Plugin initialization failed");
            }
            
            // Register services
            entry.plugin->register_services(service_locator_.get());
            
            // Start plugin
            if (!entry.plugin->start()) {
                throw std::runtime_error("Plugin start failed");
            }
            
            entry.load_time = std::chrono::steady_clock::now();
            plugin_loaded_.emit(name);
            
            log_info("Plugin loaded: " + name);
            return true;
            
        } catch (const std::exception& e) {
            handle_error(name, e);
            return false;
        }
    }
    
    bool unload_plugin(const std::string& name) {
        auto it = plugins_.find(name);
        if (it == plugins_.end()) {
            return false;
        }
        
        auto& entry = it->second;
        if (!entry.plugin) {
            return true;  // Already unloaded
        }
        
        // Check if other plugins depend on this one
        auto dependents = dependency_resolver_->get_dependents(name);
        if (!dependents.empty()) {
            log_error("Cannot unload plugin with active dependents: " + name);
            return false;
        }
        
        // Stop and shutdown plugin
        entry.plugin->stop();
        entry.plugin->shutdown();
        
        // Unregister services
        for (const auto& service : entry.provided_services) {
            service_locator_->unregister_service(service);
        }
        
        // Unload
        entry.plugin.reset();
        entry.loader.reset();
        
        plugin_unloaded_.emit(name);
        log_info("Plugin unloaded: " + name);
        
        return true;
    }
    
    bool reload_plugin(const std::string& name) {
        auto it = plugins_.find(name);
        if (it == plugins_.end() || !it->second.is_hot_reloadable) {
            return false;
        }
        
        log_info("Hot-reloading plugin: " + name);
        
        // Save plugin state
        std::string state;
        if (auto* stateful = dynamic_cast<IStatefulPlugin*>(it->second.plugin.get())) {
            state = stateful->save_state();
        }
        
        // Unload
        if (!unload_plugin(name)) {
            return false;
        }
        
        // Reload
        if (!load_plugin(name)) {
            return false;
        }
        
        // Restore state
        if (!state.empty()) {
            if (auto* stateful = dynamic_cast<IStatefulPlugin*>(it->second.plugin.get())) {
                stateful->restore_state(state);
            }
        }
        
        plugin_reloaded_.emit(name, it->second.info.version);
        return true;
    }
    
    // Plugin queries
    PluginPtr get_plugin(const std::string& name) const {
        auto it = plugins_.find(name);
        return it != plugins_.end() ? it->second.plugin : nullptr;
    }
    
    template<typename T>
    std::shared_ptr<T> get_plugin_as(const std::string& name) const {
        return std::dynamic_pointer_cast<T>(get_plugin(name));
    }
    
    std::vector<std::string> get_loaded_plugins() const {
        std::vector<std::string> result;
        for (const auto& [name, entry] : plugins_) {
            if (entry.plugin && entry.plugin->get_status() == IPlugin::Status::Started) {
                result.push_back(name);
            }
        }
        return result;
    }
    
    std::vector<std::string> get_available_plugins() const {
        std::vector<std::string> result;
        for (const auto& [name, _] : plugins_) {
            result.push_back(name);
        }
        return result;
    }
    
    // Service access
    template<typename T>
    T* get_service(const std::string& service_name) const {
        return static_cast<T*>(service_locator_->get_service(service_name));
    }
    
    // Hot reload support
    void enable_hot_reload(std::chrono::milliseconds check_interval = std::chrono::seconds(1)) {
        hot_reload_enabled_ = true;
        hot_reload_thread_ = std::thread([this, check_interval]() {
            hot_reload_loop(check_interval);
        });
    }
    
    void disable_hot_reload() {
        hot_reload_enabled_ = false;
        if (hot_reload_thread_.joinable()) {
            hot_reload_thread_.join();
        }
    }
    
    // Configuration
    void set_load_filter(std::function<bool(const PluginInfo&)> filter) {
        load_filter_ = filter;
    }
    
    void set_error_handler(std::function<void(const std::string&, const std::exception&)> handler) {
        error_handler_ = handler;
    }
    
    // Events
    auto& on_plugin_loaded() { return plugin_loaded_; }
    auto& on_plugin_unloaded() { return plugin_unloaded_; }
    auto& on_plugin_reloaded() { return plugin_reloaded_; }
    
private:
    bool is_plugin_file(const std::filesystem::path& path) const {
        auto ext = path.extension();
        #ifdef _WIN32
            return ext == ".dll";
        #elif __APPLE__
            return ext == ".dylib" || ext == ".so";
        #else
            return ext == ".so";
        #endif
    }
    
    PluginInfo read_plugin_info(const std::filesystem::path& path) {
        // Try to read manifest file first
        auto manifest_path = path;
        manifest_path.replace_extension(".manifest");
        
        if (std::filesystem::exists(manifest_path)) {
            return PluginManifest::parse(manifest_path);
        }
        
        // Otherwise, load plugin temporarily to get info
        auto loader = create_loader(path);
        auto plugin = loader->load();
        
        PluginInfo info;
        info.name = plugin->name();
        info.version = plugin->version();
        info.author = plugin->author();
        info.description = plugin->description();
        
        // Immediately unload
        plugin.reset();
        loader.reset();
        
        return info;
    }
    
    std::unique_ptr<PluginLoader> create_loader(const std::filesystem::path& path) {
        return std::make_unique<DynamicLibraryLoader>(path);
    }
    
    void hot_reload_loop(std::chrono::milliseconds check_interval) {
        while (hot_reload_enabled_) {
            for (auto& [name, entry] : plugins_) {
                if (entry.is_hot_reloadable && std::filesystem::exists(entry.path)) {
                    auto current_modified = std::filesystem::last_write_time(entry.path);
                    
                    if (current_modified > entry.last_modified) {
                        reload_plugin(name);
                        entry.last_modified = current_modified;
                    }
                }
            }
            
            std::this_thread::sleep_for(check_interval);
        }
    }
    
    void handle_error(const std::string& plugin, const std::exception& e) {
        if (error_handler_) {
            error_handler_(plugin, e);
        } else {
            log_error("Plugin error (" + plugin + "): " + e.what());
        }
    }
    
    void log_info(const std::string& message) {
        // Log implementation
    }
    
    void log_error(const std::string& message) {
        // Log implementation
    }
};
```
**Why necessary**: Central plugin management, dependency resolution, hot-reloading support.
**Usage**: Application-wide plugin system management, service discovery.

### `plugin_loader.hpp`
```cpp
class PluginLoader {
public:
    virtual ~PluginLoader() = default;
    
    virtual std::shared_ptr<IPlugin> load() = 0;
    virtual void unload() = 0;
    virtual bool is_loaded() const = 0;
    virtual std::filesystem::path get_path() const = 0;
};

// Platform-specific dynamic library loader
class DynamicLibraryLoader : public PluginLoader {
public:
    using CreateFunc = IPlugin*(*)();
    using DestroyFunc = void(*)(IPlugin*);
    using VersionFunc = uint32_t(*)();
    
private:
    std::filesystem::path path_;
    void* handle_ = nullptr;
    CreateFunc create_func_ = nullptr;
    DestroyFunc destroy_func_ = nullptr;
    std::shared_ptr<IPlugin> plugin_;
    
public:
    explicit DynamicLibraryLoader(const std::filesystem::path& path)
        : path_(path) {}
    
    ~DynamicLibraryLoader() {
        unload();
    }
    
    std::shared_ptr<IPlugin> load() override {
        if (plugin_) {
            return plugin_;
        }
        
        // Load library
        #ifdef _WIN32
            handle_ = LoadLibraryW(path_.wstring().c_str());
            if (!handle_) {
                throw std::runtime_error("Failed to load library: " + 
                    std::to_string(GetLastError()));
            }
        #else
            handle_ = dlopen(path_.c_str(), RTLD_LAZY | RTLD_LOCAL);
            if (!handle_) {
                throw std::runtime_error("Failed to load library: " + 
                    std::string(dlerror()));
            }
        #endif
        
        // Get function pointers
        create_func_ = get_symbol<CreateFunc>("create_plugin");
        destroy_func_ = get_symbol<DestroyFunc>("destroy_plugin");
        auto version_func = get_symbol<VersionFunc>("get_plugin_api_version");
        
        // Check API version
        uint32_t plugin_version = version_func();
        if (plugin_version != PLUGIN_API_VERSION) {
            throw std::runtime_error("API version mismatch. Expected: " +
                std::to_string(PLUGIN_API_VERSION) + ", Got: " +
                std::to_string(plugin_version));
        }
        
        // Create plugin instance
        IPlugin* raw_plugin = create_func_();
        if (!raw_plugin) {
            throw std::runtime_error("Failed to create plugin instance");
        }
        
        // Wrap in shared_ptr with custom deleter
        plugin_ = std::shared_ptr<IPlugin>(raw_plugin, 
            [this](IPlugin* p) {
                if (destroy_func_) {
                    destroy_func_(p);
                }
            });
        
        return plugin_;
    }
    
    void unload() override {
        plugin_.reset();
        
        if (handle_) {
            #ifdef _WIN32
                FreeLibrary(static_cast<HMODULE>(handle_));
            #else
                dlclose(handle_);
            #endif
            handle_ = nullptr;
        }
        
        create_func_ = nullptr;
        destroy_func_ = nullptr;
    }
    
    bool is_loaded() const override {
        return handle_ != nullptr && plugin_ != nullptr;
    }
    
    std::filesystem::path get_path() const override {
        return path_;
    }
    
private:
    template<typename T>
    T get_symbol(const char* name) {
        if (!handle_) {
            throw std::runtime_error("Library not loaded");
        }
        
        #ifdef _WIN32
            void* symbol = GetProcAddress(static_cast<HMODULE>(handle_), name);
        #else
            void* symbol = dlsym(handle_, name);
        #endif
        
        if (!symbol) {
            throw std::runtime_error("Symbol not found: " + std::string(name));
        }
        
        return reinterpret_cast<T>(symbol);
    }
};

// Sandboxed plugin loader
class SandboxedLoader : public PluginLoader {
    std::unique_ptr<PluginSandbox> sandbox_;
    std::unique_ptr<PluginLoader> inner_loader_;
    
public:
    SandboxedLoader(std::unique_ptr<PluginLoader> loader, 
                   const SandboxConfig& config)
        : inner_loader_(std::move(loader))
        , sandbox_(std::make_unique<PluginSandbox>(config)) {}
    
    std::shared_ptr<IPlugin> load() override {
        return sandbox_->load_plugin(inner_loader_.get());
    }
    
    void unload() override {
        sandbox_->unload_plugin();
        inner_loader_->unload();
    }
};
```
**Why necessary**: Platform-specific library loading, symbol resolution, version checking.
**Usage**: Loading plugins from disk, managing library lifecycle.

### `dependency_resolver.hpp`
```cpp
class DependencyResolver {
public:
    struct Dependency {
        std::string name;
        VersionRequirement version_requirement;
        bool is_optional = false;
    };
    
    using DependencyGraph = std::unordered_map<std::string, std::vector<Dependency>>;
    
private:
    DependencyGraph dependencies_;
    std::unordered_map<std::string, Version> available_versions_;
    
public:
    // Build dependency graph
    void add_plugin(const std::string& name, const Version& version,
                   const std::vector<Dependency>& deps) {
        dependencies_[name] = deps;
        available_versions_[name] = version;
    }
    
    // Resolve dependencies for a plugin
    std::vector<std::string> resolve(const std::string& plugin) {
        std::vector<std::string> result;
        std::unordered_set<std::string> visited;
        std::unordered_set<std::string> in_progress;
        
        resolve_recursive(plugin, result, visited, in_progress);
        
        return result;
    }
    
    // Get plugins that depend on given plugin
    std::vector<std::string> get_dependents(const std::string& plugin) {
        std::vector<std::string> result;
        
        for (const auto& [name, deps] : dependencies_) {
            for (const auto& dep : deps) {
                if (dep.name == plugin && !dep.is_optional) {
                    result.push_back(name);
                    break;
                }
            }
        }
        
        return result;
    }
    
    // Check if all dependencies are satisfied
    bool are_dependencies_satisfied(const std::string& plugin) {
        auto it = dependencies_.find(plugin);
        if (it == dependencies_.end()) {
            return true;
        }
        
        for (const auto& dep : it->second) {
            if (dep.is_optional) continue;
            
            auto version_it = available_versions_.find(dep.name);
            if (version_it == available_versions_.end()) {
                return false;
            }
            
            if (!dep.version_requirement.is_satisfied_by(version_it->second)) {
                return false;
            }
        }
        
        return true;
    }
    
    // Find conflicts
    std::vector<std::string> find_conflicts() {
        std::vector<std::string> conflicts;
        
        for (const auto& [plugin, deps] : dependencies_) {
            for (const auto& dep : deps) {
                if (!dep.is_optional) {
                    auto it = available_versions_.find(dep.name);
                    if (it != available_versions_.end() &&
                        !dep.version_requirement.is_satisfied_by(it->second)) {
                        conflicts.push_back(plugin + " requires " + dep.name + 
                                          " " + dep.version_requirement.to_string() +
                                          " but " + it->second.to_string() + " is available");
                    }
                }
            }
        }
        
        return conflicts;
    }
    
private:
    void resolve_recursive(const std::string& plugin,
                          std::vector<std::string>& result,
                          std::unordered_set<std::string>& visited,
                          std::unordered_set<std::string>& in_progress) {
        if (visited.count(plugin)) {
            return;
        }
        
        if (in_progress.count(plugin)) {
            throw std::runtime_error("Circular dependency detected: " + plugin);
        }
        
        in_progress.insert(plugin);
        
        // Resolve dependencies first (depth-first)
        auto it = dependencies_.find(plugin);
        if (it != dependencies_.end()) {
            for (const auto& dep : it->second) {
                if (!dep.is_optional) {
                    resolve_recursive(dep.name, result, visited, in_progress);
                }
            }
        }
        
        // Add this plugin after its dependencies
        result.push_back(plugin);
        visited.insert(plugin);
        in_progress.erase(plugin);
    }
};

// Version handling
class Version {
    uint32_t major_;
    uint32_t minor_;
    uint32_t patch_;
    std::string prerelease_;
    
public:
    Version(uint32_t major, uint32_t minor, uint32_t patch)
        : major_(major), minor_(minor), patch_(patch) {}
    
    static Version parse(const std::string& str) {
        // Parse semantic version string "1.2.3-beta"
        std::regex pattern(R"((\d+)\.(\d+)\.(\d+)(?:-(.+))?)");
        std::smatch matches;
        
        if (std::regex_match(str, matches, pattern)) {
            return Version{
                std::stoul(matches[1]),
                std::stoul(matches[2]),
                std::stoul(matches[3])
            };
        }
        
        throw std::invalid_argument("Invalid version string: " + str);
    }
    
    bool operator==(const Version& other) const {
        return major_ == other.major_ && 
               minor_ == other.minor_ && 
               patch_ == other.patch_;
    }
    
    bool operator<(const Version& other) const {
        if (major_ != other.major_) return major_ < other.major_;
        if (minor_ != other.minor_) return minor_ < other.minor_;
        return patch_ < other.patch_;
    }
    
    bool is_compatible_with(const Version& other) const {
        // Same major version = compatible
        return major_ == other.major_;
    }
    
    std::string to_string() const {
        return std::to_string(major_) + "." + 
               std::to_string(minor_) + "." + 
               std::to_string(patch_) +
               (prerelease_.empty() ? "" : "-" + prerelease_);
    }
};

class VersionRequirement {
    enum class Op { Equal, Greater, GreaterEqual, Less, LessEqual, Compatible };
    Op op_;
    Version version_;
    
public:
    static VersionRequirement parse(const std::string& str) {
        // Parse requirements like ">=1.2.0", "~1.2.0", "^1.0.0"
        if (str.empty()) return any();
        
        if (str[0] == '^') {  // Compatible with (same major)
            return compatible(Version::parse(str.substr(1)));
        }
        if (str[0] == '~') {  // Approximately (same major.minor)
            return approximately(Version::parse(str.substr(1)));
        }
        if (str.substr(0, 2) == ">=") {
            return at_least(Version::parse(str.substr(2)));
        }
        // ... other operators
        
        return exactly(Version::parse(str));
    }
    
    bool is_satisfied_by(const Version& v) const {
        switch (op_) {
            case Op::Equal: return v == version_;
            case Op::Greater: return v > version_;
            case Op::GreaterEqual: return v >= version_;
            case Op::Less: return v < version_;
            case Op::LessEqual: return v <= version_;
            case Op::Compatible: return v.is_compatible_with(version_);
        }
        return false;
    }
    
    static VersionRequirement any() { 
        return VersionRequirement{Op::GreaterEqual, Version{0, 0, 0}};
    }
    static VersionRequirement exactly(const Version& v) { 
        return VersionRequirement{Op::Equal, v};
    }
    static VersionRequirement at_least(const Version& v) { 
        return VersionRequirement{Op::GreaterEqual, v};
    }
    static VersionRequirement compatible(const Version& v) { 
        return VersionRequirement{Op::Compatible, v};
    }
};
```
**Why necessary**: Dependency management, version compatibility, conflict detection.
**Usage**: Automatic plugin loading order, dependency validation.

### `service_locator.hpp`
```cpp
class IServiceRegistry {
public:
    virtual ~IServiceRegistry() = default;
    
    virtual void register_service(const char* name, void* service) = 0;
    virtual void unregister_service(const char* name) = 0;
    virtual void* get_service(const char* name) = 0;
    
    template<typename T>
    void register_service(const std::string& name, T* service) {
        register_service(name.c_str(), static_cast<void*>(service));
    }
    
    template<typename T>
    T* get_service(const std::string& name) {
        return static_cast<T*>(get_service(name.c_str()));
    }
};

class ServiceLocator : public IServiceRegistry {
    struct ServiceEntry {
        void* service;
        std::string provider_plugin;
        std::type_info* type_info;
        std::chrono::steady_clock::time_point registration_time;
    };
    
    std::unordered_map<std::string, ServiceEntry> services_;
    mutable std::shared_mutex mutex_;
    
public:
    void register_service(const char* name, void* service) override {
        std::unique_lock lock(mutex_);
        
        ServiceEntry entry;
        entry.service = service;
        entry.registration_time = std::chrono::steady_clock::now();
        
        services_[name] = entry;
    }
    
    void unregister_service(const char* name) override {
        std::unique_lock lock(mutex_);
        services_.erase(name);
    }
    
    void* get_service(const char* name) override {
        std::shared_lock lock(mutex_);
        
        auto it = services_.find(name);
        return it != services_.end() ? it->second.service : nullptr;
    }
    
    // Extended interface
    std::vector<std::string> get_available_services() const {
        std::shared_lock lock(mutex_);
        
        std::vector<std::string> result;
        for (const auto& [name, _] : services_) {
            result.push_back(name);
        }
        
        return result;
    }
    
    bool has_service(const std::string& name) const {
        std::shared_lock lock(mutex_);
        return services_.count(name) > 0;
    }
};

// Type-safe service wrapper
template<typename Interface>
class TypedService {
    Interface* service_;
    std::string name_;
    ServiceLocator* locator_;
    
public:
    TypedService(const std::string& name, ServiceLocator* locator)
        : name_(name), locator_(locator) {
        service_ = locator->get_service<Interface>(name);
    }
    
    Interface* operator->() {
        if (!service_) {
            throw std::runtime_error("Service not available: " + name_);
        }
        return service_;
    }
    
    bool is_available() const {
        return service_ != nullptr;
    }
    
    void refresh() {
        service_ = locator_->get_service<Interface>(name_);
    }
};
```
**Why necessary**: Service discovery, plugin communication, dependency injection.
**Usage**: Plugins expose and consume services, decoupled communication.

### `plugin_sandbox.hpp`
```cpp
class PluginSandbox {
public:
    struct SandboxConfig {
        // Resource limits
        std::size_t max_memory = 100 * 1024 * 1024;  // 100MB
        std::size_t max_cpu_time = 1000;  // ms
        std::size_t max_threads = 4;
        
        // Permissions
        bool allow_file_access = false;
        bool allow_network_access = false;
        bool allow_process_spawn = false;
        std::vector<std::filesystem::path> allowed_paths;
        
        // Security
        bool enable_aslr = true;
        bool enable_dep = true;
        bool enable_cfi = true;
    };
    
private:
    SandboxConfig config_;
    std::unique_ptr<IPlugin> plugin_;
    
    // Resource tracking
    std::atomic<std::size_t> memory_usage_{0};
    std::atomic<std::size_t> cpu_time_{0};
    std::atomic<std::size_t> thread_count_{0};
    
    // Security context
    void* security_context_ = nullptr;
    
public:
    explicit PluginSandbox(const SandboxConfig& config)
        : config_(config) {
        initialize_sandbox();
    }
    
    ~PluginSandbox() {
        cleanup_sandbox();
    }
    
    std::shared_ptr<IPlugin> load_plugin(PluginLoader* loader) {
        // Create restricted environment
        enter_sandbox();
        
        try {
            auto plugin = loader->load();
            
            // Wrap plugin in monitoring proxy
            return std::make_shared<SandboxedPlugin>(
                plugin, 
                [this](const std::string& op) { check_permission(op); }
            );
            
        } catch (...) {
            exit_sandbox();
            throw;
        }
    }
    
    void unload_plugin() {
        plugin_.reset();
        exit_sandbox();
    }
    
    // Resource monitoring
    std::size_t get_memory_usage() const {
        return memory_usage_.load();
    }
    
    std::size_t get_cpu_time() const {
        return cpu_time_.load();
    }
    
    bool check_limits() const {
        return memory_usage_ <= config_.max_memory &&
               cpu_time_ <= config_.max_cpu_time &&
               thread_count_ <= config_.max_threads;
    }
    
private:
    void initialize_sandbox() {
        #ifdef __linux__
            // Use seccomp for system call filtering
            setup_seccomp();
            
            // Use cgroups for resource limits
            setup_cgroups();
            
            // Use namespaces for isolation
            setup_namespaces();
        #elif _WIN32
            // Use Windows Job Objects for resource limits
            setup_job_object();
            
            // Use Windows AppContainer for isolation
            setup_app_container();
        #endif
    }
    
    void enter_sandbox() {
        // Apply sandbox restrictions
        #ifdef __linux__
            // Drop capabilities
            drop_capabilities();
            
            // Apply seccomp filters
            apply_seccomp_filters();
        #elif _WIN32
            // Reduce token privileges
            reduce_token_privileges();
        #endif
    }
    
    void exit_sandbox() {
        // Restore normal environment
    }
    
    void check_permission(const std::string& operation) {
        if (operation == "file_read" && !config_.allow_file_access) {
            throw SecurityException("File access not allowed");
        }
        if (operation == "network" && !config_.allow_network_access) {
            throw SecurityException("Network access not allowed");
        }
        // ... other checks
    }
    
    void cleanup_sandbox() {
        // Clean up sandbox resources
    }
};

// Sandboxed plugin wrapper
class SandboxedPlugin : public IPlugin {
    std::shared_ptr<IPlugin> inner_;
    std::function<void(const std::string&)> permission_checker_;
    
public:
    SandboxedPlugin(std::shared_ptr<IPlugin> plugin,
                   std::function<void(const std::string&)> checker)
        : inner_(plugin), permission_checker_(checker) {}
    
    // Delegate all calls with permission checks
    bool initialize(IPluginContext* context) override {
        // Wrap context in security proxy
        auto secure_context = std::make_unique<SecureContext>(context, permission_checker_);
        return inner_->initialize(secure_context.get());
    }
    
    // ... delegate other methods
};
```
**Why necessary**: Security isolation, resource limits, permission control.
**Usage**: Running untrusted plugins, resource management, security enforcement.

### `plugin_macros.hpp`
```cpp
// Plugin declaration macros
#define DECLARE_PLUGIN(ClassName, Name, Version, Author, Description) \
    class ClassName : public PluginBase { \
    public: \
        ClassName() { \
            metadata_.name = Name; \
            metadata_.version = Version; \
            metadata_.author = Author; \
            metadata_.description = Description; \
        } \
        PLUGIN_ENTRY_POINTS(ClassName) \
    private:

#define END_PLUGIN() };

// Service registration macros
#define REGISTER_SERVICE(ServiceInterface, ServiceImpl) \
    registry->register_service(#ServiceInterface, \
        static_cast<ServiceInterface*>(new ServiceImpl()))

#define GET_SERVICE(ServiceInterface, ServiceName) \
    static_cast<ServiceInterface*>(context_->get_host_service(ServiceName))

// Dependency declaration
#define REQUIRE_PLUGIN(PluginName, Version) \
    metadata_.required_plugins.push_back({PluginName, Version})

#define OPTIONAL_PLUGIN(PluginName, Version) \
    metadata_.optional_plugins.push_back({PluginName, Version})

// Plugin lifecycle logging
#define PLUGIN_LOG_INFO(message) \
    context_->log(LogLevel::Info, message)

#define PLUGIN_LOG_ERROR(message) \
    context_->log(LogLevel::Error, message)

#define PLUGIN_LOG_DEBUG(message) \
    if (context_->has_permission("debug")) \
        context_->log(LogLevel::Debug, message)

// Event handling
#define PLUGIN_EMIT_EVENT(EventName, Data) \
    context_->emit_event(EventName, Data)

#define PLUGIN_SUBSCRIBE_EVENT(EventName, Handler) \
    context_->subscribe_event(EventName, Handler)

// Configuration
#define PLUGIN_CONFIG_PARAM(Type, Name, Default) \
    Type Name = get_config_value<Type>(#Name, Default)

// Export validation
#define VALIDATE_PLUGIN_EXPORTS() \
    static_assert(std::is_base_of_v<IPlugin, PluginClass>, \
        "Plugin must inherit from IPlugin"); \
    static_assert(sizeof(IPlugin) == sizeof(void*) * 16, \
        "IPlugin size has changed - ABI break!");
```
**Why necessary**: Simplify plugin development, ensure consistency, reduce boilerplate.
**Usage**: Plugin implementation, service registration, configuration.

## Plugin System Patterns

### Basic Plugin Implementation
```cpp
// myplugin.cpp
DECLARE_PLUGIN(MyPlugin, "MyPlugin", "1.0.0", "Author", "Description")
    
    REQUIRE_PLUGIN("CorePlugin", ">=1.0.0")
    OPTIONAL_PLUGIN("ExtensionPlugin", "~2.0.0")
    
protected:
    bool on_start() override {
        PLUGIN_LOG_INFO("MyPlugin starting");
        
        // Get required service
        auto core_service = GET_SERVICE(ICoreService, "CoreService");
        if (!core_service) {
            PLUGIN_LOG_ERROR("CoreService not available");
            return false;
        }
        
        // Initialize plugin
        initialize_components();
        
        // Subscribe to events
        PLUGIN_SUBSCRIBE_EVENT("SystemReady", [this](const void* data) {
            on_system_ready();
        });
        
        return true;
    }
    
    void on_stop() override {
        PLUGIN_LOG_INFO("MyPlugin stopping");
        cleanup_components();
    }
    
    void register_services(IServiceRegistry* registry) override {
        REGISTER_SERVICE(IMyService, MyServiceImpl);
    }
    
private:
    void initialize_components() {
        // Plugin initialization
    }
    
    void cleanup_components() {
        // Plugin cleanup
    }
    
END_PLUGIN()
```

### Plugin Manager Usage
```cpp
// Application setup
auto& manager = PluginManager::instance();

// Configure plugin system
manager.add_plugin_directory("plugins");
manager.set_load_filter([](const PluginInfo& info) {
    return info.version >= Version::parse("1.0.0");
});

// Discover and load plugins
manager.discover_plugins();
manager.load_plugin("CorePlugin");
manager.load_plugin("MyPlugin");

// Enable hot reloading for development
#ifdef DEBUG
manager.enable_hot_reload(std::chrono::seconds(1));
#endif

// Access plugin services
if (auto service = manager.get_service<IMyService>("MyService")) {
    service->do_something();
}

// Handle plugin events
manager.on_plugin_loaded().connect([](const std::string& name) {
    std::cout << "Plugin loaded: " << name << std::endl;
});
```

### Inter-Plugin Communication
```cpp
// Plugin A exposes a service
class PluginA : public PluginBase {
    void register_services(IServiceRegistry* registry) override {
        registry->register_service("CalculatorService", 
            new CalculatorService());
    }
};

// Plugin B uses the service
class PluginB : public PluginBase {
    bool on_start() override {
        auto calc = context_->get_plugin_service("PluginA", "CalculatorService");
        if (calc) {
            auto result = static_cast<ICalculator*>(calc)->calculate(42);
        }
        return true;
    }
};
```

### Sandboxed Plugin Execution
```cpp
// Load untrusted plugin with restrictions
PluginSandbox::SandboxConfig sandbox_config;
sandbox_config.max_memory = 50 * 1024 * 1024;  // 50MB limit
sandbox_config.allow_file_access = false;
sandbox_config.allow_network_access = false;

auto loader = std::make_unique<DynamicLibraryLoader>("untrusted_plugin.dll");
auto sandboxed_loader = std::make_unique<SandboxedLoader>(
    std::move(loader), sandbox_config
);

manager.load_plugin_with_loader("UntrustedPlugin", std::move(sandboxed_loader));
```

## Performance Considerations

- **Lazy loading**: ~1-10ms per plugin load
- **Service lookup**: O(1) hash table lookup
- **Hot reload check**: ~1ms per plugin file
- **Sandbox overhead**: ~5-10% performance penalty
- **Inter-plugin calls**: ~50-100ns overhead vs direct calls

## Testing Strategy

- **ABI compatibility**: Test across compiler versions
- **Dependency resolution**: Complex dependency graphs
- **Hot reloading**: File modification detection
- **Sandbox security**: Permission enforcement
- **Service discovery**: Dynamic service registration

## 📝 Usage Guidelines

1. **Version everything**: Use semantic versioning
2. **Minimize dependencies**: Keep plugins loosely coupled
3. **Use services**: Communicate through service interfaces
4. **Handle failures**: Graceful degradation
5. **Document interfaces**: Clear API documentation

## Anti-patterns to Avoid

- Direct plugin-to-plugin references
- Modifying plugin interface after release
- Circular dependencies
- Blocking operations in plugin init
- Assuming plugin load order

## Dependencies
- `base/` - For base patterns
- `filesystem/` - For plugin discovery
- `reflection/` - For type information
- Platform APIs (dlopen/LoadLibrary)

## Future Enhancements
- Remote plugin loading
- Plugin marketplace
- Automatic dependency resolution
- Plugin signing and verification
- Cross-process plugins