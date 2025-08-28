# Core Reflection - AGENT.md

## Purpose
The `reflection/` layer provides comprehensive runtime type information and introspection capabilities, enabling dynamic property access, method invocation, type discovery, and serialization without compile-time knowledge of types. It implements a lightweight, efficient reflection system that bridges the gap between C++'s static typing and dynamic runtime needs.

## Architecture Philosophy
- **Zero-overhead when unused**: Pay only for what you reflect
- **Type safety**: Compile-time verification where possible
- **Non-intrusive**: Reflect existing types without modification
- **Extensible**: Easy to add custom type handlers
- **Cache-friendly**: Optimized memory layout for reflection data

## Files Overview

### Core Components
```cpp
type_info.hpp        // Runtime type information
type_registry.hpp    // Global type registration
type_id.hpp          // Type identification system
type_traits.hpp      // Compile-time type traits
reflected_type.hpp   // Base for reflectable types
```

### Property System
```cpp
property.hpp         // Property abstraction
property_accessor.hpp // Property getter/setter
property_traits.hpp  // Property type traits
property_list.hpp    // Property collection
dynamic_property.hpp // Runtime property definition
```

### Method System
```cpp
method.hpp           // Method abstraction
method_invoker.hpp   // Method invocation
method_signature.hpp // Method signatures
method_list.hpp      // Method collection
overload_resolver.hpp // Overload resolution
```

### Field Access
```cpp
field.hpp            // Direct field access
field_pointer.hpp    // Member pointers
field_offset.hpp     // Offset-based access
field_proxy.hpp      // Field access proxy
```

### Constructor/Destructor
```cpp
constructor.hpp      // Constructor information
destructor.hpp       // Destructor information
factory.hpp          // Object factories
object_builder.hpp   // Dynamic object construction
```

### Attributes & Metadata
```cpp
attribute.hpp        // Custom attributes
metadata.hpp         // Type metadata
annotation.hpp       // Code annotations
tag.hpp              // Type tagging system
documentation.hpp    // Embedded documentation
```

### Advanced Features
```cpp
enum_reflection.hpp  // Enum introspection
template_reflection.hpp // Template introspection
inheritance.hpp      // Base class information
visitor.hpp          // Reflection visitor pattern
converter.hpp        // Type conversion
validator.hpp        // Value validation
```

### Utilities
```cpp
reflection_macros.hpp // Helper macros
type_name.hpp        // Human-readable type names
any.hpp              // Type-erased container
variant_reflection.hpp // Variant introspection
function_traits.hpp  // Function type decomposition
```

## Detailed Component Specifications

### `type_info.hpp`
```cpp
class TypeInfo {
public:
    using TypeId = std::size_t;
    using ConstructorFunc = std::function<void*(void)>;
    using DestructorFunc = std::function<void(void*)>;
    using CopyFunc = std::function<void*(const void*)>;
    using MoveFunc = std::function<void*(void*)>;
    
private:
    TypeId id_;
    std::string name_;
    std::string qualified_name_;
    std::size_t size_;
    std::size_t alignment_;
    bool is_pod_;
    bool is_polymorphic_;
    bool is_abstract_;
    
    ConstructorFunc default_constructor_;
    DestructorFunc destructor_;
    CopyFunc copy_constructor_;
    MoveFunc move_constructor_;
    
    std::vector<const TypeInfo*> base_types_;
    std::vector<Property> properties_;
    std::vector<Method> methods_;
    std::vector<Field> fields_;
    std::unordered_map<std::string, Attribute> attributes_;
    
public:
    // Type identification
    TypeId id() const { return id_; }
    const std::string& name() const { return name_; }
    const std::string& qualified_name() const { return qualified_name_; }
    
    // Type characteristics
    std::size_t size() const { return size_; }
    std::size_t alignment() const { return alignment_; }
    bool is_pod() const { return is_pod_; }
    bool is_polymorphic() const { return is_polymorphic_; }
    bool is_abstract() const { return is_abstract_; }
    
    // Object lifecycle
    void* create() const {
        if (!default_constructor_) {
            throw std::runtime_error("No default constructor for type " + name_);
        }
        return default_constructor_();
    }
    
    void* create_copy(const void* source) const {
        if (!copy_constructor_) {
            throw std::runtime_error("No copy constructor for type " + name_);
        }
        return copy_constructor_(source);
    }
    
    void destroy(void* instance) const {
        if (destructor_) {
            destructor_(instance);
        }
    }
    
    // Inheritance
    bool is_base_of(const TypeInfo* derived) const {
        if (this == derived) return true;
        
        for (const auto* base : derived->base_types_) {
            if (is_base_of(base)) return true;
        }
        return false;
    }
    
    bool is_derived_from(const TypeInfo* base) const {
        return base->is_base_of(this);
    }
    
    const std::vector<const TypeInfo*>& base_types() const { 
        return base_types_; 
    }
    
    // Member access
    const Property* find_property(const std::string& name) const {
        auto it = std::find_if(properties_.begin(), properties_.end(),
            [&name](const Property& p) { return p.name() == name; });
        return it != properties_.end() ? &(*it) : nullptr;
    }
    
    const Method* find_method(const std::string& name) const {
        auto it = std::find_if(methods_.begin(), methods_.end(),
            [&name](const Method& m) { return m.name() == name; });
        return it != methods_.end() ? &(*it) : nullptr;
    }
    
    const Field* find_field(const std::string& name) const {
        auto it = std::find_if(fields_.begin(), fields_.end(),
            [&name](const Field& f) { return f.name() == name; });
        return it != fields_.end() ? &(*it) : nullptr;
    }
    
    // Collections
    const std::vector<Property>& properties() const { return properties_; }
    const std::vector<Method>& methods() const { return methods_; }
    const std::vector<Field>& fields() const { return fields_; }
    
    // Attributes
    bool has_attribute(const std::string& name) const {
        return attributes_.find(name) != attributes_.end();
    }
    
    const Attribute* get_attribute(const std::string& name) const {
        auto it = attributes_.find(name);
        return it != attributes_.end() ? &it->second : nullptr;
    }
    
    // Type checking
    template<typename T>
    bool is() const {
        return id_ == TypeId::get<T>();
    }
    
    // Safe casting
    template<typename T>
    T* cast(void* instance) const {
        if (!is<T>()) {
            throw std::bad_cast();
        }
        return static_cast<T*>(instance);
    }
    
    // Static type info retrieval
    template<typename T>
    static const TypeInfo* get() {
        return TypeRegistry::instance().get<T>();
    }
    
    // Builder for registration
    class Builder {
        TypeInfo info_;
        
    public:
        Builder& name(const std::string& n) {
            info_.name_ = n;
            return *this;
        }
        
        Builder& size(std::size_t s) {
            info_.size_ = s;
            return *this;
        }
        
        Builder& alignment(std::size_t a) {
            info_.alignment_ = a;
            return *this;
        }
        
        template<typename T>
        Builder& constructor() {
            info_.default_constructor_ = []() -> void* {
                return new T();
            };
            return *this;
        }
        
        template<typename T>
        Builder& destructor() {
            info_.destructor_ = [](void* instance) {
                delete static_cast<T*>(instance);
            };
            return *this;
        }
        
        Builder& add_property(Property prop) {
            info_.properties_.push_back(std::move(prop));
            return *this;
        }
        
        Builder& add_method(Method method) {
            info_.methods_.push_back(std::move(method));
            return *this;
        }
        
        Builder& add_base(const TypeInfo* base) {
            info_.base_types_.push_back(base);
            return *this;
        }
        
        TypeInfo build() {
            return std::move(info_);
        }
    };
};
```
**Why necessary**: Core type information storage, object lifecycle management, member introspection.
**Usage**: Type discovery, dynamic object creation, member access.

### `property.hpp`
```cpp
class Property {
public:
    using Getter = std::function<Any(const void*)>;
    using Setter = std::function<void(void*, const Any&)>;
    
private:
    std::string name_;
    const TypeInfo* type_;
    Getter getter_;
    Setter setter_;
    bool is_readonly_;
    bool is_static_;
    std::unordered_map<std::string, Attribute> attributes_;
    
public:
    Property(const std::string& name, const TypeInfo* type)
        : name_(name), type_(type), is_readonly_(false), is_static_(false) {}
    
    // Property access
    const std::string& name() const { return name_; }
    const TypeInfo* type() const { return type_; }
    bool is_readonly() const { return is_readonly_; }
    bool is_static() const { return is_static_; }
    
    // Value access
    Any get(const void* instance) const {
        if (!getter_) {
            throw std::runtime_error("No getter for property " + name_);
        }
        return getter_(instance);
    }
    
    void set(void* instance, const Any& value) const {
        if (!setter_) {
            throw std::runtime_error("No setter for property " + name_);
        }
        if (is_readonly_) {
            throw std::runtime_error("Property " + name_ + " is readonly");
        }
        setter_(instance, value);
    }
    
    // Typed access
    template<typename T>
    T get_value(const void* instance) const {
        return get(instance).cast<T>();
    }
    
    template<typename T>
    void set_value(void* instance, const T& value) const {
        set(instance, Any(value));
    }
    
    // Builder
    class Builder {
        Property prop_;
        
    public:
        Builder(const std::string& name, const TypeInfo* type)
            : prop_(name, type) {}
        
        template<typename Class, typename Type>
        Builder& getter(Type (Class::*getter)() const) {
            prop_.getter_ = [getter](const void* instance) -> Any {
                const Class* obj = static_cast<const Class*>(instance);
                return Any((obj->*getter)());
            };
            return *this;
        }
        
        template<typename Class, typename Type>
        Builder& setter(void (Class::*setter)(Type)) {
            prop_.setter_ = [setter](void* instance, const Any& value) {
                Class* obj = static_cast<Class*>(instance);
                (obj->*setter)(value.cast<Type>());
            };
            return *this;
        }
        
        template<typename Class, typename Type>
        Builder& member(Type Class::*member) {
            prop_.getter_ = [member](const void* instance) -> Any {
                const Class* obj = static_cast<const Class*>(instance);
                return Any(obj->*member);
            };
            
            prop_.setter_ = [member](void* instance, const Any& value) {
                Class* obj = static_cast<Class*>(instance);
                obj->*member = value.cast<Type>();
            };
            return *this;
        }
        
        Builder& readonly(bool readonly = true) {
            prop_.is_readonly_ = readonly;
            return *this;
        }
        
        Builder& static_property(bool is_static = true) {
            prop_.is_static_ = is_static;
            return *this;
        }
        
        Builder& attribute(const std::string& name, const Attribute& attr) {
            prop_.attributes_[name] = attr;
            return *this;
        }
        
        Property build() {
            return std::move(prop_);
        }
    };
};

// Property with validation
class ValidatedProperty : public Property {
    std::function<bool(const Any&)> validator_;
    
public:
    void set(void* instance, const Any& value) const override {
        if (validator_ && !validator_(value)) {
            throw std::invalid_argument("Validation failed for property " + name());
        }
        Property::set(instance, value);
    }
    
    void set_validator(std::function<bool(const Any&)> validator) {
        validator_ = validator;
    }
};

// Computed property
class ComputedProperty : public Property {
    std::function<Any(const void*)> compute_;
    
public:
    ComputedProperty(const std::string& name, 
                    const TypeInfo* type,
                    std::function<Any(const void*)> compute)
        : Property(name, type), compute_(compute) {
        // Override getter to use compute function
    }
    
    Any get(const void* instance) const override {
        return compute_(instance);
    }
};
```
**Why necessary**: Dynamic property access, getter/setter abstraction, validation support.
**Usage**: Object serialization, data binding, property editors.

### `method.hpp`
```cpp
class Method {
public:
    using Invoker = std::function<Any(void*, const std::vector<Any>&)>;
    
private:
    std::string name_;
    const TypeInfo* return_type_;
    std::vector<const TypeInfo*> parameter_types_;
    std::vector<std::string> parameter_names_;
    Invoker invoker_;
    bool is_const_;
    bool is_static_;
    bool is_virtual_;
    std::unordered_map<std::string, Attribute> attributes_;
    
public:
    Method(const std::string& name) : name_(name) {}
    
    // Method information
    const std::string& name() const { return name_; }
    const TypeInfo* return_type() const { return return_type_; }
    const std::vector<const TypeInfo*>& parameter_types() const { 
        return parameter_types_; 
    }
    
    std::size_t parameter_count() const { return parameter_types_.size(); }
    bool is_const() const { return is_const_; }
    bool is_static() const { return is_static_; }
    bool is_virtual() const { return is_virtual_; }
    
    // Method invocation
    Any invoke(void* instance, const std::vector<Any>& args) const {
        if (!invoker_) {
            throw std::runtime_error("No invoker for method " + name_);
        }
        
        // Validate argument count
        if (args.size() != parameter_types_.size()) {
            throw std::invalid_argument("Argument count mismatch for method " + name_);
        }
        
        // Validate argument types
        for (std::size_t i = 0; i < args.size(); ++i) {
            if (!args[i].is_convertible_to(parameter_types_[i])) {
                throw std::invalid_argument("Type mismatch for argument " + 
                                           std::to_string(i) + " of method " + name_);
            }
        }
        
        return invoker_(instance, args);
    }
    
    // Variadic invocation
    template<typename... Args>
    Any invoke(void* instance, Args&&... args) const {
        std::vector<Any> arg_vector{Any(std::forward<Args>(args))...};
        return invoke(instance, arg_vector);
    }
    
    // Builder
    class Builder {
        Method method_;
        
    public:
        Builder(const std::string& name) : method_(name) {}
        
        template<typename Return, typename Class, typename... Args>
        Builder& function(Return (Class::*func)(Args...)) {
            method_.return_type_ = TypeInfo::get<Return>();
            method_.parameter_types_ = {TypeInfo::get<Args>()...};
            
            method_.invoker_ = [func](void* instance, const std::vector<Any>& args) -> Any {
                return invoke_impl<Return, Class, Args...>(
                    func, instance, args, std::index_sequence_for<Args...>{}
                );
            };
            
            return *this;
        }
        
        template<typename Return, typename Class, typename... Args>
        Builder& const_function(Return (Class::*func)(Args...) const) {
            method_.is_const_ = true;
            // Similar implementation for const methods
            return *this;
        }
        
        Builder& parameter_name(std::size_t index, const std::string& name) {
            if (index >= method_.parameter_names_.size()) {
                method_.parameter_names_.resize(index + 1);
            }
            method_.parameter_names_[index] = name;
            return *this;
        }
        
        Method build() {
            return std::move(method_);
        }
        
    private:
        template<typename Return, typename Class, typename... Args, std::size_t... Is>
        static Any invoke_impl(Return (Class::*func)(Args...),
                              void* instance,
                              const std::vector<Any>& args,
                              std::index_sequence<Is...>) {
            Class* obj = static_cast<Class*>(instance);
            if constexpr (std::is_void_v<Return>) {
                (obj->*func)(args[Is].cast<Args>()...);
                return Any();
            } else {
                return Any((obj->*func)(args[Is].cast<Args>()...));
            }
        }
    };
};

// Method overload set
class MethodOverloadSet {
    std::string name_;
    std::vector<Method> overloads_;
    
public:
    MethodOverloadSet(const std::string& name) : name_(name) {}
    
    void add_overload(Method method) {
        overloads_.push_back(std::move(method));
    }
    
    const Method* resolve(const std::vector<const TypeInfo*>& arg_types) const {
        // Find best matching overload
        for (const auto& method : overloads_) {
            if (matches(method, arg_types)) {
                return &method;
            }
        }
        return nullptr;
    }
    
    Any invoke(void* instance, const std::vector<Any>& args) const {
        std::vector<const TypeInfo*> arg_types;
        for (const auto& arg : args) {
            arg_types.push_back(arg.type());
        }
        
        const Method* method = resolve(arg_types);
        if (!method) {
            throw std::runtime_error("No matching overload for method " + name_);
        }
        
        return method->invoke(instance, args);
    }
    
private:
    bool matches(const Method& method, const std::vector<const TypeInfo*>& arg_types) const {
        if (method.parameter_count() != arg_types.size()) {
            return false;
        }
        
        for (std::size_t i = 0; i < arg_types.size(); ++i) {
            if (!arg_types[i]->is_convertible_to(method.parameter_types()[i])) {
                return false;
            }
        }
        
        return true;
    }
};
```
**Why necessary**: Dynamic method invocation, overload resolution, parameter validation.
**Usage**: RPC systems, scripting, command patterns.

### `type_registry.hpp`
```cpp
class TypeRegistry : public Singleton<TypeRegistry> {
    std::unordered_map<TypeInfo::TypeId, std::unique_ptr<TypeInfo>> types_by_id_;
    std::unordered_map<std::string, TypeInfo*> types_by_name_;
    mutable std::shared_mutex mutex_;
    
public:
    // Type registration
    template<typename T>
    void register_type() {
        auto type_info = build_type_info<T>();
        register_type(std::move(type_info));
    }
    
    void register_type(std::unique_ptr<TypeInfo> type_info) {
        std::unique_lock lock(mutex_);
        
        auto id = type_info->id();
        auto name = type_info->name();
        
        types_by_id_[id] = std::move(type_info);
        types_by_name_[name] = types_by_id_[id].get();
    }
    
    // Type retrieval
    template<typename T>
    const TypeInfo* get() const {
        return get(TypeId::get<T>());
    }
    
    const TypeInfo* get(TypeInfo::TypeId id) const {
        std::shared_lock lock(mutex_);
        auto it = types_by_id_.find(id);
        return it != types_by_id_.end() ? it->second.get() : nullptr;
    }
    
    const TypeInfo* get(const std::string& name) const {
        std::shared_lock lock(mutex_);
        auto it = types_by_name_.find(name);
        return it != types_by_name_.end() ? it->second : nullptr;
    }
    
    // Type discovery
    std::vector<const TypeInfo*> get_all_types() const {
        std::shared_lock lock(mutex_);
        std::vector<const TypeInfo*> types;
        for (const auto& [_, type] : types_by_id_) {
            types.push_back(type.get());
        }
        return types;
    }
    
    std::vector<const TypeInfo*> get_derived_types(const TypeInfo* base) const {
        std::shared_lock lock(mutex_);
        std::vector<const TypeInfo*> derived;
        
        for (const auto& [_, type] : types_by_id_) {
            if (type->is_derived_from(base)) {
                derived.push_back(type.get());
            }
        }
        
        return derived;
    }
    
    // Type checking
    template<typename T>
    bool is_registered() const {
        return get<T>() != nullptr;
    }
    
    // Automatic registration helper
    template<typename T>
    struct AutoRegister {
        AutoRegister() {
            TypeRegistry::instance().register_type<T>();
        }
    };
    
private:
    template<typename T>
    std::unique_ptr<TypeInfo> build_type_info() {
        auto builder = TypeInfo::Builder()
            .name(type_name<T>())
            .size(sizeof(T))
            .alignment(alignof(T));
        
        // Add constructor if available
        if constexpr (std::is_default_constructible_v<T>) {
            builder.constructor<T>();
        }
        
        // Add destructor
        builder.destructor<T>();
        
        // Register members (requires specialization)
        register_members<T>(builder);
        
        return std::make_unique<TypeInfo>(builder.build());
    }
    
    template<typename T>
    void register_members(TypeInfo::Builder& builder) {
        // Default: no members
        // Specialize for each type to add members
    }
};

// Automatic registration macro
#define REGISTER_TYPE(Type) \
    static TypeRegistry::AutoRegister<Type> _register_##Type;
```
**Why necessary**: Global type management, type discovery, automatic registration.
**Usage**: Plugin systems, serialization, factory patterns.

### `enum_reflection.hpp`
```cpp
template<typename Enum>
class EnumInfo {
    static_assert(std::is_enum_v<Enum>, "Type must be an enum");
    
public:
    struct Enumerator {
        std::string name;
        Enum value;
        std::unordered_map<std::string, Attribute> attributes;
    };
    
private:
    std::string name_;
    const TypeInfo* underlying_type_;
    std::vector<Enumerator> enumerators_;
    std::unordered_map<Enum, std::size_t> value_to_index_;
    std::unordered_map<std::string, std::size_t> name_to_index_;
    bool is_flags_;
    
public:
    EnumInfo(const std::string& name) 
        : name_(name)
        , underlying_type_(TypeInfo::get<std::underlying_type_t<Enum>>())
        , is_flags_(false) {}
    
    // Enum information
    const std::string& name() const { return name_; }
    const TypeInfo* underlying_type() const { return underlying_type_; }
    bool is_flags() const { return is_flags_; }
    
    // Enumerator access
    const std::vector<Enumerator>& enumerators() const { return enumerators_; }
    
    const Enumerator* find_by_value(Enum value) const {
        auto it = value_to_index_.find(value);
        if (it != value_to_index_.end()) {
            return &enumerators_[it->second];
        }
        return nullptr;
    }
    
    const Enumerator* find_by_name(const std::string& name) const {
        auto it = name_to_index_.find(name);
        if (it != name_to_index_.end()) {
            return &enumerators_[it->second];
        }
        return nullptr;
    }
    
    // Conversion
    std::string to_string(Enum value) const {
        if (is_flags_) {
            return flags_to_string(value);
        }
        
        const Enumerator* e = find_by_value(value);
        return e ? e->name : std::to_string(static_cast<std::underlying_type_t<Enum>>(value));
    }
    
    std::optional<Enum> from_string(const std::string& name) const {
        if (is_flags_) {
            return flags_from_string(name);
        }
        
        const Enumerator* e = find_by_name(name);
        return e ? std::optional(e->value) : std::nullopt;
    }
    
    // Builder
    class Builder {
        EnumInfo info_;
        
    public:
        Builder(const std::string& name) : info_(name) {}
        
        Builder& value(const std::string& name, Enum value) {
            std::size_t index = info_.enumerators_.size();
            info_.enumerators_.push_back({name, value});
            info_.value_to_index_[value] = index;
            info_.name_to_index_[name] = index;
            return *this;
        }
        
        Builder& flags(bool is_flags = true) {
            info_.is_flags_ = is_flags;
            return *this;
        }
        
        EnumInfo build() {
            return std::move(info_);
        }
    };
    
    // Static registration
    static void register_enum(EnumInfo info) {
        get_registry()[typeid(Enum)] = std::make_unique<EnumInfo>(std::move(info));
    }
    
    static const EnumInfo* get() {
        auto& registry = get_registry();
        auto it = registry.find(typeid(Enum));
        return it != registry.end() ? it->second.get() : nullptr;
    }
    
private:
    std::string flags_to_string(Enum value) const {
        std::vector<std::string> flags;
        auto int_value = static_cast<std::underlying_type_t<Enum>>(value);
        
        for (const auto& e : enumerators_) {
            auto flag_value = static_cast<std::underlying_type_t<Enum>>(e.value);
            if (flag_value != 0 && (int_value & flag_value) == flag_value) {
                flags.push_back(e.name);
            }
        }
        
        return flags.empty() ? "None" : join(flags, " | ");
    }
    
    std::optional<Enum> flags_from_string(const std::string& str) const {
        auto parts = split(str, '|');
        std::underlying_type_t<Enum> result = 0;
        
        for (const auto& part : parts) {
            auto trimmed = trim(part);
            const Enumerator* e = find_by_name(trimmed);
            if (!e) {
                return std::nullopt;
            }
            result |= static_cast<std::underlying_type_t<Enum>>(e->value);
        }
        
        return static_cast<Enum>(result);
    }
    
    static std::unordered_map<std::type_index, std::unique_ptr<EnumInfo>>& get_registry() {
        static std::unordered_map<std::type_index, std::unique_ptr<EnumInfo>> registry;
        return registry;
    }
};

// Enum registration macro
#define REFLECT_ENUM(EnumType, ...) \
    namespace { \
        struct EnumType##_Registrar { \
            EnumType##_Registrar() { \
                EnumInfo<EnumType>::register_enum( \
                    EnumInfo<EnumType>::Builder(#EnumType) \
                        __VA_ARGS__ \
                        .build() \
                ); \
            } \
        } EnumType##_registrar; \
    }

#define ENUM_VALUE(name, value) .value(name, value)
```
**Why necessary**: Enum string conversion, flag handling, enum iteration.
**Usage**: Configuration files, UI dropdowns, serialization.

### `reflection_macros.hpp`
```cpp
// Class reflection macros
#define REFLECT_CLASS_BEGIN(ClassName) \
    template<> \
    void TypeRegistry::register_members<ClassName>(TypeInfo::Builder& builder) { \
        using Class = ClassName;

#define REFLECT_CLASS_END() \
    }

#define REFLECT_BASE(BaseClass) \
    builder.add_base(TypeInfo::get<BaseClass>());

#define REFLECT_PROPERTY(name) \
    builder.add_property( \
        Property::Builder(#name, TypeInfo::get<decltype(Class::name)>()) \
            .member(&Class::name) \
            .build() \
    );

#define REFLECT_READONLY_PROPERTY(name) \
    builder.add_property( \
        Property::Builder(#name, TypeInfo::get<decltype(Class::name)>()) \
            .member(&Class::name) \
            .readonly() \
            .build() \
    );

#define REFLECT_GETTER_SETTER(name, getter, setter) \
    builder.add_property( \
        Property::Builder(#name, TypeInfo::get<decltype(std::declval<Class>().getter())>()) \
            .getter(&Class::getter) \
            .setter(&Class::setter) \
            .build() \
    );

#define REFLECT_METHOD(name) \
    builder.add_method( \
        Method::Builder(#name) \
            .function(&Class::name) \
            .build() \
    );

#define REFLECT_STATIC_METHOD(name) \
    builder.add_method( \
        Method::Builder(#name) \
            .static_function(&Class::name) \
            .build() \
    );

#define REFLECT_ATTRIBUTE(name, value) \
    builder.add_attribute(name, Attribute(value));

// Complete class reflection
#define REFLECT_CLASS(ClassName, ...) \
    REFLECT_CLASS_BEGIN(ClassName) \
        __VA_ARGS__ \
    REFLECT_CLASS_END() \
    REGISTER_TYPE(ClassName)

// Usage example:
/*
class Person {
    std::string name_;
    int age_;
    
public:
    const std::string& name() const { return name_; }
    void set_name(const std::string& n) { name_ = n; }
    int age() const { return age_; }
    void set_age(int a) { age_ = a; }
    
    void say_hello() {
        std::cout << "Hello, I'm " << name_ << std::endl;
    }
};

REFLECT_CLASS(Person,
    REFLECT_GETTER_SETTER(name, name, set_name)
    REFLECT_GETTER_SETTER(age, age, set_age)
    REFLECT_METHOD(say_hello)
    REFLECT_ATTRIBUTE("description", "A person class")
)
*/

// Automatic reflection using templates
template<typename T>
struct Reflector {
    static void reflect() {
        // Default: no reflection
    }
};

// Specialization macro
#define REFLECT_TYPE_AUTO(Type) \
    template<> \
    struct Reflector<Type> { \
        static void reflect() { \
            auto builder = TypeInfo::Builder() \
                .name(#Type) \
                .size(sizeof(Type)) \
                .alignment(alignof(Type)); \
            reflect_members(builder); \
            TypeRegistry::instance().register_type( \
                std::make_unique<TypeInfo>(builder.build()) \
            ); \
        } \
        static void reflect_members(TypeInfo::Builder& builder); \
    }; \
    inline void Reflector<Type>::reflect_members(TypeInfo::Builder& builder)
```
**Why necessary**: Simplified reflection registration, reduced boilerplate.
**Usage**: Declarative reflection, compile-time registration.

### `any.hpp`
```cpp
class Any {
    struct Base {
        virtual ~Base() = default;
        virtual const std::type_info& type() const = 0;
        virtual std::unique_ptr<Base> clone() const = 0;
        virtual const TypeInfo* type_info() const = 0;
    };
    
    template<typename T>
    struct Derived : Base {
        T value;
        
        explicit Derived(T v) : value(std::move(v)) {}
        
        const std::type_info& type() const override {
            return typeid(T);
        }
        
        std::unique_ptr<Base> clone() const override {
            return std::make_unique<Derived>(value);
        }
        
        const TypeInfo* type_info() const override {
            return TypeInfo::get<T>();
        }
    };
    
    std::unique_ptr<Base> content_;
    
public:
    Any() = default;
    
    template<typename T>
    Any(T value) : content_(std::make_unique<Derived<T>>(std::move(value))) {}
    
    Any(const Any& other) 
        : content_(other.content_ ? other.content_->clone() : nullptr) {}
    
    Any(Any&&) = default;
    
    Any& operator=(const Any& other) {
        content_ = other.content_ ? other.content_->clone() : nullptr;
        return *this;
    }
    
    Any& operator=(Any&&) = default;
    
    bool has_value() const { return content_ != nullptr; }
    
    const std::type_info& type() const {
        return content_ ? content_->type() : typeid(void);
    }
    
    const TypeInfo* type_info() const {
        return content_ ? content_->type_info() : nullptr;
    }
    
    template<typename T>
    T cast() const {
        if (!content_) {
            throw std::bad_cast();
        }
        
        if (typeid(T) != content_->type()) {
            throw std::bad_cast();
        }
        
        return static_cast<Derived<T>*>(content_.get())->value;
    }
    
    template<typename T>
    T* try_cast() {
        if (!content_ || typeid(T) != content_->type()) {
            return nullptr;
        }
        
        return &static_cast<Derived<T>*>(content_.get())->value;
    }
    
    template<typename T>
    bool is() const {
        return content_ && typeid(T) == content_->type();
    }
    
    bool is_convertible_to(const TypeInfo* target_type) const {
        if (!content_ || !target_type) return false;
        
        // Check for exact match
        if (type_info() == target_type) return true;
        
        // Check for base class conversion
        return type_info()->is_derived_from(target_type);
    }
};
```
**Why necessary**: Type-erased storage, dynamic typing, variant operations.
**Usage**: Property storage, method arguments, dynamic dispatch.

## Reflection Patterns

### Basic Class Reflection
```cpp
class GameObject {
    std::string name_;
    Vector3 position_;
    float health_;
    bool active_;
    
public:
    const std::string& name() const { return name_; }
    void set_name(const std::string& n) { name_ = n; }
    
    const Vector3& position() const { return position_; }
    void set_position(const Vector3& p) { position_ = p; }
    
    float health() const { return health_; }
    void set_health(float h) { health_ = std::clamp(h, 0.0f, 100.0f); }
    
    void take_damage(float amount) {
        set_health(health_ - amount);
    }
    
    void heal(float amount) {
        set_health(health_ + amount);
    }
};

REFLECT_CLASS(GameObject,
    REFLECT_GETTER_SETTER(name, name, set_name)
    REFLECT_GETTER_SETTER(position, position, set_position)
    REFLECT_GETTER_SETTER(health, health, set_health)
    REFLECT_PROPERTY(active)
    REFLECT_METHOD(take_damage)
    REFLECT_METHOD(heal)
    REFLECT_ATTRIBUTE("category", "game_object")
    REFLECT_ATTRIBUTE("serializable", true)
)
```

### Dynamic Property Access
```cpp
void inspect_object(void* obj, const TypeInfo* type) {
    std::cout << "Type: " << type->name() << std::endl;
    
    for (const auto& prop : type->properties()) {
        std::cout << "  " << prop.name() << " = ";
        
        Any value = prop.get(obj);
        if (value.is<int>()) {
            std::cout << value.cast<int>();
        } else if (value.is<float>()) {
            std::cout << value.cast<float>();
        } else if (value.is<std::string>()) {
            std::cout << value.cast<std::string>();
        } else {
            std::cout << "<complex type>";
        }
        
        std::cout << std::endl;
    }
}
```

### Serialization Using Reflection
```cpp
class JsonSerializer {
public:
    nlohmann::json serialize(const void* obj, const TypeInfo* type) {
        nlohmann::json json;
        
        // Serialize type information
        json["_type"] = type->name();
        
        // Serialize properties
        for (const auto& prop : type->properties()) {
            if (prop.has_attribute("transient")) continue;
            
            Any value = prop.get(obj);
            json[prop.name()] = serialize_value(value);
        }
        
        return json;
    }
    
    void deserialize(void* obj, const TypeInfo* type, const nlohmann::json& json) {
        // Check type
        if (json["_type"] != type->name()) {
            throw std::runtime_error("Type mismatch");
        }
        
        // Deserialize properties
        for (const auto& prop : type->properties()) {
            if (prop.is_readonly()) continue;
            
            auto it = json.find(prop.name());
            if (it != json.end()) {
                Any value = deserialize_value(*it, prop.type());
                prop.set(obj, value);
            }
        }
    }
    
private:
    nlohmann::json serialize_value(const Any& value) {
        if (value.is<int>()) return value.cast<int>();
        if (value.is<float>()) return value.cast<float>();
        if (value.is<std::string>()) return value.cast<std::string>();
        
        // Recursive serialization for complex types
        if (auto type = value.type_info()) {
            return serialize(value.try_cast<void>(), type);
        }
        
        return nullptr;
    }
    
    Any deserialize_value(const nlohmann::json& json, const TypeInfo* type) {
        // Handle primitive types
        if (type->is<int>()) return Any(json.get<int>());
        if (type->is<float>()) return Any(json.get<float>());
        if (type->is<std::string>()) return Any(json.get<std::string>());
        
        // Handle complex types
        void* obj = type->create();
        deserialize(obj, type, json);
        return Any(obj);
    }
};
```

### Factory Pattern with Reflection
```cpp
class ObjectFactory {
    std::unordered_map<std::string, std::function<void*()>> creators_;
    
public:
    template<typename T>
    void register_type(const std::string& name) {
        creators_[name] = []() { return new T(); };
    }
    
    void* create(const std::string& type_name) {
        auto type = TypeRegistry::instance().get(type_name);
        if (!type) {
            throw std::runtime_error("Unknown type: " + type_name);
        }
        
        return type->create();
    }
    
    template<typename T>
    std::unique_ptr<T> create_typed(const std::string& type_name) {
        void* obj = create(type_name);
        auto type = TypeRegistry::instance().get(type_name);
        
        if (!type->is_derived_from(TypeInfo::get<T>())) {
            type->destroy(obj);
            throw std::bad_cast();
        }
        
        return std::unique_ptr<T>(static_cast<T*>(obj));
    }
};
```

### Property Editor UI
```cpp
class PropertyEditor {
    void edit_object(void* obj, const TypeInfo* type) {
        ImGui::Text("Editing: %s", type->name().c_str());
        
        for (const auto& prop : type->properties()) {
            if (prop.is_readonly()) {
                ImGui::BeginDisabled();
            }
            
            Any value = prop.get(obj);
            bool changed = false;
            
            if (value.is<int>()) {
                int val = value.cast<int>();
                changed = ImGui::InputInt(prop.name().c_str(), &val);
                if (changed) prop.set(obj, Any(val));
                
            } else if (value.is<float>()) {
                float val = value.cast<float>();
                changed = ImGui::InputFloat(prop.name().c_str(), &val);
                if (changed) prop.set(obj, Any(val));
                
            } else if (value.is<std::string>()) {
                std::string val = value.cast<std::string>();
                char buffer[256];
                std::strncpy(buffer, val.c_str(), sizeof(buffer));
                
                changed = ImGui::InputText(prop.name().c_str(), buffer, sizeof(buffer));
                if (changed) prop.set(obj, Any(std::string(buffer)));
                
            } else if (value.is<bool>()) {
                bool val = value.cast<bool>();
                changed = ImGui::Checkbox(prop.name().c_str(), &val);
                if (changed) prop.set(obj, Any(val));
            }
            
            if (prop.is_readonly()) {
                ImGui::EndDisabled();
            }
        }
    }
};
```

## Performance Considerations

- **Registration overhead**: One-time cost at startup
- **Property access**: ~50-100ns through reflection vs direct access
- **Method invocation**: ~100-200ns overhead for dynamic dispatch
- **Type lookup**: O(1) hash table lookup
- **Memory usage**: ~200-500 bytes per reflected type

## Testing Strategy

- **Type registration**: Verify all types registered correctly
- **Property access**: Test getters/setters through reflection
- **Method invocation**: Test various parameter combinations
- **Inheritance**: Verify base class traversal
- **Thread safety**: Concurrent type registration and access

## Usage Guidelines

1. **Register at startup**: Use static initialization for registration
2. **Minimize reflection use**: Direct access when type is known
3. **Cache type info**: Store TypeInfo* instead of repeated lookups
4. **Validate carefully**: Check types before casting
5. **Consider performance**: Profile reflection-heavy code

## Anti-patterns to Avoid

- Reflecting everything (only reflect what needs introspection)
- Runtime registration in hot paths
- Deep inheritance hierarchies
- Storing Any when concrete type is known
- Ignoring const-correctness

## Dependencies
- `base/` - For Singleton, Registry patterns
- `error/` - For error handling
- Standard library (C++20)

## Future Enhancements
- Code generation for reflection
- Compile-time reflection (C++23)
- Attribute validation
- Performance optimizations
- Reflection-based scripting