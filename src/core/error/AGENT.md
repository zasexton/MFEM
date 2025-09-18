# Core Error Handling - AGENT.md

## 🎯 Purpose
The `error/` layer provides comprehensive error handling infrastructure including result types, error codes, exception hierarchies, and error propagation mechanisms. It enables robust error handling without forcing exceptions while supporting multiple error handling styles.

## 🏗️ Architecture Philosophy
- **Multiple error styles**: Support both exceptions and error codes
- **Zero-cost success path**: No overhead when operations succeed
- **Rich error context**: Detailed error information with stack traces
- **Type safety**: Compile-time error category checking
- **Composability**: Chain and transform error handling

## 📁 Files Overview

### Core Error Types
```cpp
error_code.h       // Portable error codes with categories
error_category.h   // Error category system
result.h          // Result<T, E> type (success or error)
expected.h        // Expected<T> with default error type
outcome.h         // Outcome<T> with exception support
status.h          // Lightweight status codes
```

### Exception Hierarchy
```cpp
exception_base.h   // Base exception with rich context
system_error.h    // System/OS errors
logic_error.h     // Programming errors
runtime_error.h   // Runtime failures
nested_exception.h // Exception chaining
```

### Error Context
```cpp
error_context.h   // Contextual error information
stack_trace.h     // Call stack capture
source_location.h // File/line/function info
error_message.h   // Formatted error messages
```

### Error Handling
```cpp
error_handler.h   // Global error handler registration
error_guard.h     // RAII error handling
try_catch.h      // Exception-to-result conversion
error_chain.h    // Error aggregation and chaining
panic.h          // Unrecoverable error handling
```

### Utilities
```cpp
// NOTE: Assert functionality is provided by logging/assert.h
// This subfolder focuses on contract programming and validation
precondition.h   // Precondition checking macros
postcondition.h  // Postcondition validation
invariant.h      // Class invariant checking
contract.h       // Design by contract support
validation.h     // Input validation utilities
```

## 🔧 Detailed Component Specifications

### `result.h`
```cpp
template<typename T, typename E = ErrorCode>
class Result {
    std::variant<T, E> storage_;
    
public:
    // Construction
    static Result success(T value);
    static Result failure(E error);
    
    // Checking
    bool is_success() const;
    bool is_failure() const;
    explicit operator bool() const { return is_success(); }
    
    // Access
    T& value() &;
    const T& value() const&;
    T&& value() &&;
    E& error() &;
    const E& error() const&;
    
    // Monadic operations
    template<typename F>
    auto map(F&& f) -> Result<decltype(f(std::declval<T>())), E>;
    
    template<typename F>
    auto map_error(F&& f) -> Result<T, decltype(f(std::declval<E>()))>;
    
    template<typename F>
    auto and_then(F&& f) -> decltype(f(std::declval<T>()));
    
    template<typename F>
    auto or_else(F&& f) -> Result<T, E>;
    
    // Value extraction
    T value_or(T default_value) const;
    T expect(const char* msg) const;  // Panics with message if error
};

// Convenience factory functions
template<typename T>
Result<T> ok(T value);

template<typename E>
Result<void, E> err(E error);
```
**Why necessary**: Explicit error handling without exceptions, composable error propagation, functional programming style.
**Usage**: Return values from fallible operations, chain operations with error handling.

### `error_code.h`
```cpp
class ErrorCategory {
public:
    virtual const char* name() const noexcept = 0;
    virtual std::string message(int code) const = 0;
    virtual bool equivalent(int code, const ErrorCode& other) const noexcept;
};

class ErrorCode {
    int code_;
    const ErrorCategory* category_;
    
public:
    ErrorCode() noexcept : code_(0), category_(&system_category()) {}
    ErrorCode(int code, const ErrorCategory& cat) noexcept;
    
    int value() const noexcept { return code_; }
    const ErrorCategory& category() const noexcept { return *category_; }
    std::string message() const { return category_->message(code_); }
    
    explicit operator bool() const noexcept { return code_ != 0; }
    
    friend bool operator==(const ErrorCode& a, const ErrorCode& b) noexcept;
};

// Common error categories
const ErrorCategory& system_category() noexcept;
const ErrorCategory& generic_category() noexcept;
const ErrorCategory& app_category() noexcept;

// Enum-based error codes
template<typename Enum>
class ErrorCodeEnum : public ErrorCategory {
    // Auto-generates category from enum
};
```
**Why necessary**: Portable error representation, categorized errors, interop with system errors.
**Usage**: System calls, library boundaries, error categorization.

### `expected.h`
```cpp
template<typename T>
class Expected : public Result<T, std::exception_ptr> {
public:
    // Try-catch wrapper
    template<typename F>
    static Expected try_invoke(F&& f) noexcept {
        try {
            return Expected::success(f());
        } catch (...) {
            return Expected::failure(std::current_exception());
        }
    }
    
    // Rethrow if error
    T& get() {
        if (is_failure()) {
            std::rethrow_exception(error());
        }
        return value();
    }
};
```
**Why necessary**: Bridge between exception and error-code worlds, exception safety.
**Usage**: Wrapping exception-throwing code, API boundaries.

### `exception_base.h`
```cpp
class Exception : public std::exception {
    std::string message_;
    ErrorCode code_;
    SourceLocation location_;
    std::vector<std::string> context_;
    std::optional<StackTrace> stack_trace_;
    
public:
    Exception(std::string message, 
             ErrorCode code = ErrorCode(),
             SourceLocation loc = SourceLocation::current());
    
    const char* what() const noexcept override { return message_.c_str(); }
    const ErrorCode& code() const noexcept { return code_; }
    const SourceLocation& where() const noexcept { return location_; }
    
    // Context building
    Exception& with_context(std::string ctx) {
        context_.push_back(std::move(ctx));
        return *this;
    }
    
    Exception& with_stack_trace() {
        stack_trace_ = StackTrace::current();
        return *this;
    }
    
    // Formatted output
    std::string full_message() const;
    void print_diagnostic(std::ostream& os) const;
};

// Derived exception types
class LogicError : public Exception { };
class RuntimeError : public Exception { };
class SystemError : public Exception { };
```
**Why necessary**: Rich error information, debugging support, error categorization.
**Usage**: Exceptional conditions, detailed error reporting, debugging.

### `error_handler.h`
```cpp
class ErrorHandler {
public:
    using Handler = std::function<void(const Exception&)>;
    
    static void set_handler(Handler handler);
    static void set_panic_handler(std::function<void(const char*)> handler);
    
    static void handle_error(const Exception& e);
    static void panic(const char* message) [[noreturn]];
    
    // Scoped handler override
    class ScopedHandler {
        Handler previous_;
    public:
        explicit ScopedHandler(Handler h);
        ~ScopedHandler();
    };
};

// Global handler setup
void set_terminate_handler();
void set_unexpected_handler();
void install_signal_handlers();
```
**Why necessary**: Centralized error handling, crash reporting, graceful shutdowns.
**Usage**: Application-wide error policy, logging integration, crash dumps.

### Assertion Integration
Assertion functionality is provided by `logging/assert.h` which includes:
- `FEM_ASSERT` - Debug-only assertions
- `FEM_ASSERT_ALWAYS` - Release assertions
- `FEM_VERIFY` - Non-fatal verification
- `FEM_ASSERT_FINITE`, `FEM_ASSERT_POSITIVE`, etc. - Numerical assertions
- `FEM_PRECONDITION`, `FEM_POSTCONDITION`, `FEM_INVARIANT` - Contract assertions
- `FEM_UNREACHABLE`, `FEM_NOT_IMPLEMENTED` - Special assertions

The error/ subfolder complements this with contract programming support in `contract.h`
and validation utilities in `validation.h`.

### `try_catch.h`
```cpp
// Convert exception-throwing code to Result
template<typename F>
auto try_catch(F&& f) -> Result<decltype(f()), Exception> {
    try {
        return Result<decltype(f()), Exception>::success(f());
    } catch (const Exception& e) {
        return Result<decltype(f()), Exception>::failure(e);
    } catch (const std::exception& e) {
        return Result<decltype(f()), Exception>::failure(
            Exception(e.what()).with_stack_trace()
        );
    } catch (...) {
        return Result<decltype(f()), Exception>::failure(
            Exception("Unknown exception").with_stack_trace()
        );
    }
}

// Execute with guaranteed cleanup
template<typename F, typename C>
auto try_finally(F&& f, C&& cleanup) {
    struct Cleanup {
        C cleanup;
        ~Cleanup() { cleanup(); }
    } c{std::forward<C>(cleanup)};
    
    return f();
}
```
**Why necessary**: Exception safety, resource cleanup, error conversion.
**Usage**: Library boundaries, resource management, error translation.

### `error_chain.h`
```cpp
class ErrorChain {
    std::vector<std::variant<ErrorCode, Exception>> errors_;
    
public:
    void add_error(ErrorCode e);
    void add_error(Exception e);
    
    bool has_errors() const { return !errors_.empty(); }
    size_t error_count() const { return errors_.size(); }
    
    // Combine multiple operations
    template<typename... Ops>
    Result<std::tuple<typename Ops::value_type...>, ErrorChain> 
    collect(Ops... ops) {
        ErrorChain chain;
        auto results = std::make_tuple(ops()...);
        
        std::apply([&chain](auto&&... r) {
            ((r.is_failure() ? chain.add_error(r.error()) : void()), ...);
        }, results);
        
        if (chain.has_errors()) {
            return Result<>::failure(chain);
        }
        
        return Result<>::success(/* extract values */);
    }
    
    std::string format() const;
};
```
**Why necessary**: Multiple error aggregation, batch operations, validation results.
**Usage**: Form validation, batch processing, multi-step operations.

## 🔄 Error Flow Patterns

### Result-Based Error Handling
```cpp
Result<Data> load_data(const Path& path) {
    auto file = open_file(path);
    if (!file) {
        return err(file.error());
    }
    
    return parse_data(file.value())
        .map_error([&](auto e) {
            return ErrorCode(ERR_PARSE_FAILED).with_context(path.string());
        });
}

// Usage
auto result = load_data("config.json")
    .and_then([](auto data) { return validate(data); })
    .map([](auto data) { return process(data); })
    .or_else([](auto error) {
        log_error(error);
        return load_defaults();
    });
```

### Exception-Based with Conversion
```cpp
Expected<int> safe_divide(int a, int b) {
    return Expected<int>::try_invoke([=]() {
        if (b == 0) {
            throw LogicError("Division by zero");
        }
        return a / b;
    });
}
```

### Assertion and Contracts
```cpp
class BankAccount {
    double balance_;
    
public:
    void withdraw(double amount) {
        PRECONDITION(amount > 0, "Amount must be positive");
        PRECONDITION(amount <= balance_, "Insufficient funds");
        
        balance_ -= amount;
        
        POSTCONDITION(balance_ >= 0, "Balance cannot be negative");
        INVARIANT(is_valid(), "Account invariant violated");
    }
};
```

## ⚡ Performance Considerations

- **Result overhead**: Size of largest variant member + discriminator
- **Exception cost**: Zero-cost when not thrown (table-based)
- **Stack trace**: ~1-5ms to capture (optional)
- **Error codes**: Single integer comparison
- **Assertion cost**: Compiled out in release builds

## 🧪 Testing Strategy

- **Success path testing**: Verify zero overhead
- **Error propagation**: Test error chain through call stack
- **Exception safety**: Verify cleanup on throw
- **Thread safety**: Concurrent error handling
- **Performance**: Measure error handling overhead

## 📝 Usage Guidelines

1. **Use Result for expected errors**: File not found, network timeout
2. **Use exceptions for bugs**: Null pointer, out of bounds
3. **Always provide context**: Add location and relevant data
4. **Fail fast**: Check preconditions early
5. **Clean error messages**: User-facing vs developer-facing

## 🚫 Anti-patterns to Avoid

- Ignoring error returns
- Empty catch blocks
- Using exceptions for control flow
- Generic error messages
- Missing error context

## 🔗 Dependencies
- `base/` - For Object, Registry patterns
- Standard library (C++20)

## 📈 Future Enhancements
- Structured error reporting
- Error recovery strategies
- Automatic retry mechanisms
- Error telemetry integration
- Machine-readable error formats