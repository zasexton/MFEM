# Core Serialization - AGENT.md

## 🎯 Purpose
The `serialization/` layer provides comprehensive object serialization and deserialization capabilities supporting multiple formats, versioning, and cross-platform compatibility. It enables persistence, network communication, and data exchange with automatic type registration and schema evolution.

## 🏗️ Architecture Philosophy
- **Format agnostic**: Support multiple serialization formats
- **Type safety**: Compile-time type checking where possible
- **Version compatibility**: Handle schema evolution gracefully
- **Zero-copy where possible**: Minimize memory allocations
- **Extensibility**: Easy to add new types and formats

## 📁 Files Overview

### Core Components
```cpp
serializer.hpp       // Base serializer interface
deserializer.hpp     // Base deserializer interface
archive.hpp          // Archive abstraction
serializable.hpp     // Serializable object interface
type_registry.hpp    // Runtime type registration
```

### Format Implementations
```cpp
binary_archive.hpp   // Binary serialization
json_archive.hpp     // JSON serialization
xml_archive.hpp      // XML serialization
yaml_archive.hpp     // YAML serialization
msgpack_archive.hpp  // MessagePack format
protobuf_archive.hpp // Protocol Buffers support
```

### Advanced Features
```cpp
versioning.hpp       // Version management
schema.hpp           // Schema definition
migration.hpp        // Schema migration
compression.hpp      // Compressed serialization
encryption.hpp       // Encrypted serialization
validation.hpp       // Data validation
```

### Type Support
```cpp
primitive_types.hpp  // Built-in type serialization
container_types.hpp  // STL container support
custom_types.hpp     // Custom type registration
polymorphic.hpp      // Polymorphic serialization
shared_ptr_support.hpp // Smart pointer support
optional_support.hpp // Optional/variant support
```

### Utilities
```cpp
buffer.hpp           // Serialization buffers
endianness.hpp       // Endianness conversion
base64.hpp           // Base64 encoding
hex.hpp              // Hex encoding
checksum.hpp         // Data integrity checks
```

## 🔧 Detailed Component Specifications

### `serializable.hpp`
```cpp
class Serializable {
public:
    virtual ~Serializable() = default;
    
    // Version info
    virtual uint32_t serialization_version() const { return 1; }
    virtual std::string serialization_type() const = 0;
    
    // Serialization interface
    virtual void serialize(Archive& archive) const = 0;
    virtual void deserialize(Archive& archive) = 0;
    
    // Convenience methods
    std::vector<uint8_t> to_bytes() const;
    void from_bytes(const std::vector<uint8_t>& data);
    
    std::string to_json() const;
    void from_json(const std::string& json);
    
    std::string to_xml() const;
    void from_xml(const std::string& xml);
};

// CRTP base for automatic registration
template<typename Derived>
class AutoSerializable : public Serializable {
    static bool registered_;
    
public:
    AutoSerializable() {
        if (!registered_) {
            TypeRegistry::register_type<Derived>();
            registered_ = true;
        }
    }
    
    std::string serialization_type() const override {
        return typeid(Derived).name();
    }
};

// Macro for easy serialization
#define SERIALIZABLE(Type) \
    friend class Archive; \
    template<typename Archive> \
    void serialize(Archive& ar) { \
        serialize_impl(ar); \
    } \
    template<typename Archive> \
    void serialize_impl(Archive& ar)
```
**Why necessary**: Common interface for serializable objects, automatic registration.
**Usage**: Data persistence, network protocols, save/load functionality.

### `archive.hpp`
```cpp
class Archive {
public:
    enum class Mode {
        Save,
        Load
    };
    
    virtual ~Archive() = default;
    
    virtual Mode mode() const = 0;
    virtual bool is_saving() const { return mode() == Mode::Save; }
    virtual bool is_loading() const { return mode() == Mode::Load; }
    
    // Primitive types
    virtual Archive& operator<<(bool value) = 0;
    virtual Archive& operator<<(int8_t value) = 0;
    virtual Archive& operator<<(int16_t value) = 0;
    virtual Archive& operator<<(int32_t value) = 0;
    virtual Archive& operator<<(int64_t value) = 0;
    virtual Archive& operator<<(float value) = 0;
    virtual Archive& operator<<(double value) = 0;
    virtual Archive& operator<<(const std::string& value) = 0;
    
    virtual Archive& operator>>(bool& value) = 0;
    virtual Archive& operator>>(int8_t& value) = 0;
    virtual Archive& operator>>(int16_t& value) = 0;
    virtual Archive& operator>>(int32_t& value) = 0;
    virtual Archive& operator>>(int64_t& value) = 0;
    virtual Archive& operator>>(float& value) = 0;
    virtual Archive& operator>>(double& value) = 0;
    virtual Archive& operator>>(std::string& value) = 0;
    
    // Named values (for readable formats)
    template<typename T>
    Archive& operator&(const NamedValue<T>& nv) {
        if (is_saving()) {
            begin_field(nv.name);
            *this << nv.value;
            end_field();
        } else {
            begin_field(nv.name);
            *this >> nv.value;
            end_field();
        }
        return *this;
    }
    
    // Object serialization
    template<typename T>
    Archive& operator&(T& object) {
        if (is_saving()) {
            save_object(object);
        } else {
            load_object(object);
        }
        return *this;
    }
    
    // Container support
    template<typename T>
    Archive& operator&(std::vector<T>& vec) {
        size_t size = vec.size();
        *this & size;
        if (is_loading()) {
            vec.resize(size);
        }
        for (auto& elem : vec) {
            *this & elem;
        }
        return *this;
    }
    
    // Versioning
    virtual void set_version(uint32_t version) = 0;
    virtual uint32_t get_version() const = 0;
    
protected:
    virtual void begin_field(const std::string& name) {}
    virtual void end_field() {}
    
    template<typename T>
    void save_object(const T& obj) {
        const_cast<T&>(obj).serialize(*this);
    }
    
    template<typename T>
    void load_object(T& obj) {
        obj.serialize(*this);
    }
};

// Helper for named values
template<typename T>
struct NamedValue {
    const char* name;
    T& value;
    
    NamedValue(const char* n, T& v) : name(n), value(v) {}
};

template<typename T>
NamedValue<T> make_named(const char* name, T& value) {
    return NamedValue<T>(name, value);
}

#define SERIALIZE_NAMED(ar, value) ar & make_named(#value, value)
```
**Why necessary**: Unified serialization interface, format independence.
**Usage**: Save/load, network protocols, data exchange.

### `binary_archive.hpp`
```cpp
class BinaryOutputArchive : public Archive {
    std::ostream& stream_;
    std::stack<size_t> size_positions_;
    Endianness endianness_;
    
public:
    explicit BinaryOutputArchive(std::ostream& stream, 
                                Endianness endian = Endianness::Native);
    
    Mode mode() const override { return Mode::Save; }
    
    // Efficient primitive serialization
    Archive& operator<<(bool value) override {
        write_raw(value);
        return *this;
    }
    
    Archive& operator<<(int32_t value) override {
        if (endianness_ != Endianness::Native) {
            value = swap_endian(value);
        }
        write_raw(value);
        return *this;
    }
    
    Archive& operator<<(const std::string& value) override {
        uint32_t size = value.size();
        *this << size;
        stream_.write(value.data(), size);
        return *this;
    }
    
    // Raw data writing
    void write_raw(const void* data, size_t size) {
        stream_.write(static_cast<const char*>(data), size);
    }
    
    template<typename T>
    void write_raw(const T& value) {
        write_raw(&value, sizeof(T));
    }
    
    // Size prefixing for safety
    void begin_size_prefix() {
        size_positions_.push(stream_.tellp());
        uint32_t placeholder = 0;
        write_raw(placeholder);
    }
    
    void end_size_prefix() {
        auto end_pos = stream_.tellp();
        auto size_pos = size_positions_.top();
        size_positions_.pop();
        
        uint32_t size = end_pos - size_pos - sizeof(uint32_t);
        stream_.seekp(size_pos);
        write_raw(size);
        stream_.seekp(end_pos);
    }
};

class BinaryInputArchive : public Archive {
    std::istream& stream_;
    Endianness endianness_;
    std::vector<uint8_t> buffer_;
    
public:
    explicit BinaryInputArchive(std::istream& stream,
                               Endianness endian = Endianness::Native);
    
    Mode mode() const override { return Mode::Load; }
    
    Archive& operator>>(int32_t& value) override {
        read_raw(value);
        if (endianness_ != Endianness::Native) {
            value = swap_endian(value);
        }
        return *this;
    }
    
    Archive& operator>>(std::string& value) override {
        uint32_t size;
        *this >> size;
        value.resize(size);
        stream_.read(value.data(), size);
        return *this;
    }
    
    // Zero-copy string view reading
    std::string_view read_string_view(size_t size) {
        buffer_.resize(size);
        stream_.read(reinterpret_cast<char*>(buffer_.data()), size);
        return std::string_view(reinterpret_cast<char*>(buffer_.data()), size);
    }
    
    template<typename T>
    void read_raw(T& value) {
        stream_.read(reinterpret_cast<char*>(&value), sizeof(T));
    }
};
```
**Why necessary**: Efficient binary serialization, compact storage.
**Usage**: File formats, network protocols, performance-critical serialization.

### `json_archive.hpp`
```cpp
class JsonOutputArchive : public Archive {
    nlohmann::json& json_;
    std::stack<nlohmann::json*> context_;
    
public:
    explicit JsonOutputArchive(nlohmann::json& j) : json_(j) {
        context_.push(&json_);
    }
    
    Mode mode() const override { return Mode::Save; }
    
    Archive& operator<<(int32_t value) override {
        current() = value;
        return *this;
    }
    
    Archive& operator<<(const std::string& value) override {
        current() = value;
        return *this;
    }
    
    void begin_object() {
        current() = nlohmann::json::object();
        context_.push(&current());
    }
    
    void end_object() {
        context_.pop();
    }
    
    void begin_array() {
        current() = nlohmann::json::array();
        context_.push(&current());
    }
    
    void end_array() {
        context_.pop();
    }
    
protected:
    void begin_field(const std::string& name) override {
        auto& obj = *context_.top();
        context_.push(&obj[name]);
    }
    
    void end_field() override {
        context_.pop();
    }
    
private:
    nlohmann::json& current() {
        return *context_.top();
    }
};

// Pretty printing options
class JsonSerializer {
    int indent_;
    bool sort_keys_;
    bool escape_unicode_;
    
public:
    JsonSerializer() : indent_(2), sort_keys_(false), escape_unicode_(false) {}
    
    JsonSerializer& set_indent(int indent) {
        indent_ = indent;
        return *this;
    }
    
    JsonSerializer& set_sort_keys(bool sort) {
        sort_keys_ = sort;
        return *this;
    }
    
    template<typename T>
    std::string serialize(const T& object) {
        nlohmann::json j;
        JsonOutputArchive archive(j);
        archive & const_cast<T&>(object);
        return j.dump(indent_, ' ', escape_unicode_);
    }
    
    template<typename T>
    T deserialize(const std::string& json) {
        auto j = nlohmann::json::parse(json);
        JsonInputArchive archive(j);
        T object;
        archive & object;
        return object;
    }
};
```
**Why necessary**: Human-readable format, web APIs, configuration.
**Usage**: REST APIs, configuration files, debugging.

### `versioning.hpp`
```cpp
class VersionedArchive : public Archive {
    std::unique_ptr<Archive> underlying_;
    uint32_t version_;
    std::map<std::string, uint32_t> type_versions_;
    
public:
    explicit VersionedArchive(std::unique_ptr<Archive> archive);
    
    void set_version(uint32_t version) override {
        version_ = version;
    }
    
    uint32_t get_version() const override {
        return version_;
    }
    
    void set_type_version(const std::string& type, uint32_t version) {
        type_versions_[type] = version;
    }
    
    uint32_t get_type_version(const std::string& type) const {
        auto it = type_versions_.find(type);
        return it != type_versions_.end() ? it->second : 1;
    }
    
    // Version-aware serialization
    template<typename T>
    void serialize_versioned(T& object) {
        uint32_t obj_version = object.serialization_version();
        *this << obj_version;
        
        if (is_saving()) {
            object.serialize(*this);
        } else {
            if (obj_version > object.serialization_version()) {
                throw VersionError("Cannot deserialize newer version");
            }
            
            // Let object handle older versions
            set_version(obj_version);
            object.serialize(*this);
        }
    }
};

// Migration support
template<typename T>
class Migrator {
    using MigrationFunc = std::function<void(T&, uint32_t from, uint32_t to)>;
    std::map<std::pair<uint32_t, uint32_t>, MigrationFunc> migrations_;
    
public:
    void register_migration(uint32_t from, uint32_t to, MigrationFunc func) {
        migrations_[{from, to}] = func;
    }
    
    void migrate(T& object, uint32_t from_version, uint32_t to_version) {
        while (from_version < to_version) {
            uint32_t next = from_version + 1;
            auto it = migrations_.find({from_version, next});
            if (it == migrations_.end()) {
                throw MigrationError("No migration path");
            }
            it->second(object, from_version, next);
            from_version = next;
        }
    }
};

// Versioned serialization macro
#define SERIALIZE_VERSIONED(Type, Version) \
    uint32_t serialization_version() const override { return Version; } \
    void serialize(Archive& ar) override { \
        if (ar.get_version() == 1) { \
            serialize_v1(ar); \
        } else if (ar.get_version() == 2) { \
            serialize_v2(ar); \
        } \
    }
```
**Why necessary**: Schema evolution, backward compatibility, data migration.
**Usage**: Long-lived data, database schemas, protocol evolution.

### `polymorphic.hpp`
```cpp
class PolymorphicArchive : public Archive {
    Archive& underlying_;
    
public:
    explicit PolymorphicArchive(Archive& ar) : underlying_(ar) {}
    
    // Polymorphic object serialization
    template<typename Base>
    void save_polymorphic(const std::unique_ptr<Base>& ptr) {
        if (!ptr) {
            underlying_ << std::string("null");
            return;
        }
        
        std::string type_name = typeid(*ptr).name();
        underlying_ << type_name;
        
        auto serializer = TypeRegistry::get_serializer(type_name);
        serializer(underlying_, ptr.get());
    }
    
    template<typename Base>
    void load_polymorphic(std::unique_ptr<Base>& ptr) {
        std::string type_name;
        underlying_ >> type_name;
        
        if (type_name == "null") {
            ptr.reset();
            return;
        }
        
        auto factory = TypeRegistry::get_factory<Base>(type_name);
        ptr = factory();
        
        auto deserializer = TypeRegistry::get_deserializer(type_name);
        deserializer(underlying_, ptr.get());
    }
    
    // Shared pointer support
    template<typename T>
    void save_shared(const std::shared_ptr<T>& ptr) {
        static std::map<void*, uint32_t> ptr_map;
        
        if (!ptr) {
            underlying_ << uint32_t(0);
            return;
        }
        
        void* raw_ptr = ptr.get();
        auto it = ptr_map.find(raw_ptr);
        
        if (it != ptr_map.end()) {
            // Already serialized, save reference
            underlying_ << it->second;
        } else {
            // First time, assign ID and serialize
            uint32_t id = ptr_map.size() + 1;
            ptr_map[raw_ptr] = id;
            underlying_ << id;
            save_polymorphic(ptr);
        }
    }
};

// Type registration
class TypeRegistry {
    struct TypeInfo {
        std::function<std::unique_ptr<void>()> factory;
        std::function<void(Archive&, void*)> serializer;
        std::function<void(Archive&, void*)> deserializer;
    };
    
    static std::map<std::string, TypeInfo> registry_;
    
public:
    template<typename T>
    static void register_type() {
        TypeInfo info;
        info.factory = []() { return std::make_unique<T>(); };
        info.serializer = [](Archive& ar, void* ptr) {
            static_cast<T*>(ptr)->serialize(ar);
        };
        info.deserializer = [](Archive& ar, void* ptr) {
            static_cast<T*>(ptr)->serialize(ar);
        };
        
        registry_[typeid(T).name()] = info;
    }
    
    template<typename Base>
    static std::unique_ptr<Base> create(const std::string& type_name) {
        auto it = registry_.find(type_name);
        if (it == registry_.end()) {
            throw std::runtime_error("Type not registered: " + type_name);
        }
        
        void* raw_ptr = it->second.factory().release();
        return std::unique_ptr<Base>(static_cast<Base*>(raw_ptr));
    }
};

// Auto-registration macro
#define REGISTER_SERIALIZABLE(Type) \
    namespace { \
        struct Type##_Registrar { \
            Type##_Registrar() { \
                TypeRegistry::register_type<Type>(); \
            } \
        } Type##_registrar; \
    }
```
**Why necessary**: Polymorphic serialization, plugin systems, dynamic types.
**Usage**: Abstract interfaces, plugin architectures, game objects.

## 🔄 Serialization Patterns

### Simple Object Serialization
```cpp
struct Person {
    std::string name;
    int age;
    std::vector<std::string> hobbies;
    
    template<typename Archive>
    void serialize(Archive& ar) {
        ar & make_named("name", name);
        ar & make_named("age", age);
        ar & make_named("hobbies", hobbies);
    }
};

// Usage
Person person{"Alice", 30, {"reading", "hiking"}};

// To JSON
JsonSerializer json_ser;
std::string json = json_ser.serialize(person);

// To binary
std::ofstream file("person.dat", std::ios::binary);
BinaryOutputArchive ar(file);
ar & person;
```

### Versioned Serialization
```cpp
class Document {
    std::string title_;
    std::string content_;
    std::string author_;  // Added in v2
    
public:
    uint32_t serialization_version() const { return 2; }
    
    void serialize(Archive& ar) {
        ar & make_named("title", title_);
        ar & make_named("content", content_);
        
        if (ar.get_version() >= 2) {
            ar & make_named("author", author_);
        }
    }
};
```

### Polymorphic Serialization
```cpp
class Shape : public Serializable {
public:
    virtual double area() const = 0;
};

class Circle : public Shape {
    double radius_;
public:
    void serialize(Archive& ar) override {
        ar & make_named("radius", radius_);
    }
};

class Rectangle : public Shape {
    double width_, height_;
public:
    void serialize(Archive& ar) override {
        ar & make_named("width", width_);
        ar & make_named("height", height_);
    }
};

REGISTER_SERIALIZABLE(Circle)
REGISTER_SERIALIZABLE(Rectangle)

// Usage
std::vector<std::unique_ptr<Shape>> shapes;
shapes.push_back(std::make_unique<Circle>());
shapes.push_back(std::make_unique<Rectangle>());

PolymorphicArchive ar(output);
ar & shapes;
```

## Performance Considerations

- **Binary format**: Most efficient for size and speed
- **Buffer reuse**: Minimize allocations
- **Schema caching**: Cache reflection data
- **Lazy deserialization**: Load on demand
- **Compression**: Use for large data sets

## Testing Strategy

- **Round-trip tests**: Serialize and deserialize
- **Version compatibility**: Test migrations
- **Format compatibility**: Cross-format testing
- **Performance benchmarks**: Speed and size
- **Edge cases**: Null, empty, circular references

## Usage Guidelines

1. **Choose appropriate format**: Binary for speed, JSON for debugging
2. **Version from start**: Always include version info
3. **Handle nulls**: Explicit null handling
4. **Validate data**: Check after deserialization
5. **Document schemas**: Keep schema documentation

## Anti-patterns to Avoid

- Serializing pointers directly
- Ignoring versioning
- Not handling endianness
- Circular references without detection
- Large monolithic serialization

## Dependencies
- `base/` - For Object patterns
- `error/` - For error handling
- `io/` - For I/O operations
- Standard library (C++20)
- Optional: nlohmann/json, protobuf

## Future Enhancements
- Reflection-based automatic serialization
- Schema generation
- Binary compatibility checking
- Differential serialization
- Distributed serialization