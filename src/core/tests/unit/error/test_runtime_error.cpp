#include <gtest/gtest.h>
#include <core/error/runtime_error.h>
#include <core/error/error_code.h>
#include <chrono>
#include <thread>
#include <vector>
#include <memory>
#include <future>
#include <sstream>
#include <fstream>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <random>

using namespace fem::core::error;
using namespace std::chrono_literals;

class RuntimeErrorTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_file_path_ = "/tmp/test_runtime_" + std::to_string(std::random_device{}());
        test_dir_path_ = "/tmp/test_dir_" + std::to_string(std::random_device{}());
    }

    void TearDown() override {
        std::remove(test_file_path_.c_str());
        std::remove(test_dir_path_.c_str());
    }

    void create_test_file() {
        std::ofstream file(test_file_path_);
        file << "test content for runtime error testing";
        file.close();
    }

    std::string test_file_path_;
    std::string test_dir_path_;
};

// ========== Basic RuntimeError tests ==========
TEST_F(RuntimeErrorTest, RuntimeErrorConstruction) {
    RuntimeError ex("Runtime error message");
    EXPECT_STREQ(ex.what(), "Runtime error message");
    EXPECT_EQ(ex.code(), ErrorCode::Unknown);
}

TEST_F(RuntimeErrorTest, RuntimeErrorInheritance) {
    RuntimeError ex("Runtime error");
    Exception* base = &ex;
    std::exception* std_base = &ex;

    EXPECT_STREQ(base->what(), "Runtime error");
    EXPECT_EQ(base->code(), ErrorCode::Unknown);
    EXPECT_STREQ(std_base->what(), "Runtime error");
}

TEST_F(RuntimeErrorTest, RuntimeErrorWithSourceLocation) {
    RuntimeError ex("Test error");
    const auto& loc = ex.where();
    EXPECT_TRUE(loc.file_name() != nullptr);
    EXPECT_TRUE(loc.function_name() != nullptr);
    EXPECT_GT(loc.line(), 0u);
}

// ========== IOError tests ==========
TEST_F(RuntimeErrorTest, IOErrorReadOperation) {
    IOError ex(IOError::Operation::Read, "file.txt", "file not found");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("read") != std::string::npos);
    EXPECT_TRUE(message.find("file.txt") != std::string::npos);
    EXPECT_TRUE(message.find("file not found") != std::string::npos);
    EXPECT_EQ(ex.operation(), IOError::Operation::Read);
    EXPECT_EQ(ex.resource(), "file.txt");
}

TEST_F(RuntimeErrorTest, IOErrorWriteOperation) {
    IOError ex(IOError::Operation::Write, "/tmp/readonly.txt", "permission denied");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("write") != std::string::npos);
    EXPECT_TRUE(message.find("/tmp/readonly.txt") != std::string::npos);
    EXPECT_TRUE(message.find("permission denied") != std::string::npos);
    EXPECT_EQ(ex.operation(), IOError::Operation::Write);
}

TEST_F(RuntimeErrorTest, IOErrorAllOperations) {
    std::vector<IOError::Operation> operations = {
        IOError::Operation::Read,
        IOError::Operation::Write,
        IOError::Operation::Open,
        IOError::Operation::Close,
        IOError::Operation::Seek,
        IOError::Operation::Flush,
        IOError::Operation::Other
    };

    for (auto op : operations) {
        IOError ex(op, "test_resource", "test_reason");
        EXPECT_EQ(ex.operation(), op);
        EXPECT_EQ(ex.resource(), "test_resource");
        EXPECT_FALSE(std::string(ex.what()).empty());
    }
}

// ========== NotFoundError tests ==========
TEST_F(RuntimeErrorTest, NotFoundErrorFile) {
    NotFoundError ex(NotFoundError::ResourceType::File, "missing.txt");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("File") != std::string::npos);
    EXPECT_TRUE(message.find("missing.txt") != std::string::npos);
    EXPECT_TRUE(message.find("not found") != std::string::npos);
    EXPECT_EQ(ex.resource_type(), NotFoundError::ResourceType::File);
    EXPECT_EQ(ex.resource_name(), "missing.txt");
}

TEST_F(RuntimeErrorTest, NotFoundErrorDirectory) {
    NotFoundError ex(NotFoundError::ResourceType::Directory, "/path/to/nowhere");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("Directory") != std::string::npos);
    EXPECT_TRUE(message.find("/path/to/nowhere") != std::string::npos);
    EXPECT_EQ(ex.resource_type(), NotFoundError::ResourceType::Directory);
}

TEST_F(RuntimeErrorTest, NotFoundErrorAllTypes) {
    std::vector<NotFoundError::ResourceType> types = {
        NotFoundError::ResourceType::File,
        NotFoundError::ResourceType::Directory,
        NotFoundError::ResourceType::Key,
        NotFoundError::ResourceType::Element,
        NotFoundError::ResourceType::Object,
        NotFoundError::ResourceType::Other
    };

    for (auto type : types) {
        NotFoundError ex(type, "test_resource");
        EXPECT_EQ(ex.resource_type(), type);
        EXPECT_EQ(ex.resource_name(), "test_resource");
        EXPECT_FALSE(std::string(ex.what()).empty());
    }
}

// ========== AlreadyExistsError tests ==========
TEST_F(RuntimeErrorTest, AlreadyExistsErrorConstruction) {
    AlreadyExistsError ex("File", "existing.txt");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("File") != std::string::npos);
    EXPECT_TRUE(message.find("existing.txt") != std::string::npos);
    EXPECT_TRUE(message.find("already exists") != std::string::npos);
    EXPECT_EQ(ex.resource_type(), "File");
    EXPECT_EQ(ex.resource_name(), "existing.txt");
}

TEST_F(RuntimeErrorTest, AlreadyExistsErrorVariousTypes) {
    std::vector<std::string> resource_types = {
        "File", "Directory", "Table", "User", "Database"
    };

    for (const auto& type : resource_types) {
        AlreadyExistsError ex(type, "test_name");
        EXPECT_EQ(ex.resource_type(), type);
        EXPECT_EQ(ex.resource_name(), "test_name");
        EXPECT_TRUE(std::string(ex.what()).find(type) != std::string::npos);
    }
}

// ========== TimeoutError tests ==========
TEST_F(RuntimeErrorTest, TimeoutErrorConstruction) {
    TimeoutError ex("database connection", 5000ms);
    std::string message = ex.what();

    EXPECT_TRUE(message.find("database connection") != std::string::npos);
    EXPECT_TRUE(message.find("5000") != std::string::npos);
    EXPECT_TRUE(message.find("timed out") != std::string::npos);
    EXPECT_EQ(ex.operation(), "database connection");
    EXPECT_EQ(ex.timeout(), 5000ms);
}

TEST_F(RuntimeErrorTest, TimeoutErrorVariousDurations) {
    std::vector<std::chrono::milliseconds> timeouts = {
        100ms, 1000ms, 30000ms, 60000ms
    };

    for (auto timeout : timeouts) {
        TimeoutError ex("test_operation", timeout);
        EXPECT_EQ(ex.timeout(), timeout);
        EXPECT_EQ(ex.operation(), "test_operation");
        EXPECT_TRUE(std::string(ex.what()).find(std::to_string(timeout.count())) != std::string::npos);
    }
}

// ========== PermissionError tests ==========
TEST_F(RuntimeErrorTest, PermissionErrorRead) {
    PermissionError ex(PermissionError::Permission::Read, "/etc/shadow");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("read") != std::string::npos);
    EXPECT_TRUE(message.find("/etc/shadow") != std::string::npos);
    EXPECT_TRUE(message.find("Permission denied") != std::string::npos);
    EXPECT_EQ(ex.permission(), PermissionError::Permission::Read);
    EXPECT_EQ(ex.resource(), "/etc/shadow");
}

TEST_F(RuntimeErrorTest, PermissionErrorAllPermissions) {
    std::vector<PermissionError::Permission> permissions = {
        PermissionError::Permission::Read,
        PermissionError::Permission::Write,
        PermissionError::Permission::Execute,
        PermissionError::Permission::Delete,
        PermissionError::Permission::Create,
        PermissionError::Permission::Other
    };

    for (auto perm : permissions) {
        PermissionError ex(perm, "test_resource");
        EXPECT_EQ(ex.permission(), perm);
        EXPECT_EQ(ex.resource(), "test_resource");
        EXPECT_FALSE(std::string(ex.what()).empty());
    }
}

// ========== ResourceExhaustedError tests ==========
TEST_F(RuntimeErrorTest, ResourceExhaustedErrorMemory) {
    ResourceExhaustedError ex(ResourceExhaustedError::Resource::Memory, 1024);
    std::string message = ex.what();

    EXPECT_TRUE(message.find("Memory") != std::string::npos);
    EXPECT_TRUE(message.find("exhausted") != std::string::npos);
    EXPECT_TRUE(message.find("1024") != std::string::npos);
    EXPECT_EQ(ex.resource(), ResourceExhaustedError::Resource::Memory);
    EXPECT_EQ(ex.limit(), 1024);
}

TEST_F(RuntimeErrorTest, ResourceExhaustedErrorNoLimit) {
    ResourceExhaustedError ex(ResourceExhaustedError::Resource::DiskSpace);
    std::string message = ex.what();

    EXPECT_TRUE(message.find("Disk space") != std::string::npos);
    EXPECT_TRUE(message.find("exhausted") != std::string::npos);
    EXPECT_EQ(ex.resource(), ResourceExhaustedError::Resource::DiskSpace);
    EXPECT_FALSE(ex.limit().has_value());
}

TEST_F(RuntimeErrorTest, ResourceExhaustedErrorAllTypes) {
    std::vector<ResourceExhaustedError::Resource> resources = {
        ResourceExhaustedError::Resource::Memory,
        ResourceExhaustedError::Resource::DiskSpace,
        ResourceExhaustedError::Resource::FileHandles,
        ResourceExhaustedError::Resource::Threads,
        ResourceExhaustedError::Resource::Connections,
        ResourceExhaustedError::Resource::Other
    };

    for (auto resource : resources) {
        ResourceExhaustedError ex(resource, 100);
        EXPECT_EQ(ex.resource(), resource);
        EXPECT_EQ(ex.limit(), 100);
        EXPECT_FALSE(std::string(ex.what()).empty());
    }
}

// ========== ConfigurationError tests ==========
TEST_F(RuntimeErrorTest, ConfigurationErrorConstruction) {
    ConfigurationError ex("database_url", "invalid format");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("database_url") != std::string::npos);
    EXPECT_TRUE(message.find("invalid format") != std::string::npos);
    EXPECT_TRUE(message.find("Configuration error") != std::string::npos);
    EXPECT_EQ(ex.parameter(), "database_url");
}

TEST_F(RuntimeErrorTest, ConfigurationErrorVariousParameters) {
    std::vector<std::pair<std::string, std::string>> configs = {
        {"port", "must be between 1 and 65535"},
        {"timeout", "must be positive"},
        {"log_level", "must be one of: debug, info, warning, error"},
        {"max_connections", "exceeds system limit"}
    };

    for (const auto& [param, reason] : configs) {
        ConfigurationError ex(param, reason);
        EXPECT_EQ(ex.parameter(), param);
        EXPECT_TRUE(std::string(ex.what()).find(param) != std::string::npos);
        EXPECT_TRUE(std::string(ex.what()).find(reason) != std::string::npos);
    }
}

// ========== ParseError tests ==========
TEST_F(RuntimeErrorTest, ParseErrorConstruction) {
    ParseError ex("malformed json {", 15, "closing brace");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("15") != std::string::npos);
    EXPECT_TRUE(message.find("closing brace") != std::string::npos);
    EXPECT_TRUE(message.find("Parse error") != std::string::npos);
    EXPECT_EQ(ex.input(), "malformed json {");
    EXPECT_EQ(ex.position(), 15);
    EXPECT_EQ(ex.expected(), "closing brace");
}

TEST_F(RuntimeErrorTest, ParseErrorVariousInputs) {
    std::vector<std::tuple<std::string, size_t, std::string>> test_cases = {
        {"123.456.789", 7, "end of number"},
        {"[1,2,3,", 7, "closing bracket"},
        {"SELECT * FORM table", 9, "FROM keyword"},
        {"<xml><tag></xml>", 10, "closing tag"}
    };

    for (const auto& [input, pos, expected] : test_cases) {
        ParseError ex(input, pos, expected);
        EXPECT_EQ(ex.input(), input);
        EXPECT_EQ(ex.position(), pos);
        EXPECT_EQ(ex.expected(), expected);
        EXPECT_TRUE(std::string(ex.what()).find(std::to_string(pos)) != std::string::npos);
    }
}

// ========== NetworkError tests ==========
TEST_F(RuntimeErrorTest, NetworkErrorConnectionRefused) {
    NetworkError ex(NetworkError::Type::ConnectionRefused, "localhost:8080");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("Connection refused") != std::string::npos);
    EXPECT_TRUE(message.find("localhost:8080") != std::string::npos);
    EXPECT_EQ(ex.type(), NetworkError::Type::ConnectionRefused);
    EXPECT_EQ(ex.endpoint(), "localhost:8080");
}

TEST_F(RuntimeErrorTest, NetworkErrorWithDetails) {
    NetworkError ex(NetworkError::Type::Timeout, "api.example.com", "after 30 seconds");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("timeout") != std::string::npos);
    EXPECT_TRUE(message.find("api.example.com") != std::string::npos);
    EXPECT_TRUE(message.find("after 30 seconds") != std::string::npos);
}

TEST_F(RuntimeErrorTest, NetworkErrorAllTypes) {
    std::vector<NetworkError::Type> types = {
        NetworkError::Type::ConnectionRefused,
        NetworkError::Type::ConnectionLost,
        NetworkError::Type::HostNotFound,
        NetworkError::Type::Timeout,
        NetworkError::Type::ProtocolError,
        NetworkError::Type::Other
    };

    for (auto type : types) {
        NetworkError ex(type, "test.endpoint.com");
        EXPECT_EQ(ex.type(), type);
        EXPECT_EQ(ex.endpoint(), "test.endpoint.com");
        EXPECT_FALSE(std::string(ex.what()).empty());
    }
}

// ========== ConcurrencyError tests ==========
TEST_F(RuntimeErrorTest, ConcurrencyErrorDeadlock) {
    ConcurrencyError ex(ConcurrencyError::Type::Deadlock, "circular dependency between threads");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("Deadlock") != std::string::npos);
    EXPECT_TRUE(message.find("circular dependency") != std::string::npos);
    EXPECT_EQ(ex.type(), ConcurrencyError::Type::Deadlock);
}

TEST_F(RuntimeErrorTest, ConcurrencyErrorAllTypes) {
    std::vector<ConcurrencyError::Type> types = {
        ConcurrencyError::Type::Deadlock,
        ConcurrencyError::Type::RaceCondition,
        ConcurrencyError::Type::LockTimeout,
        ConcurrencyError::Type::ThreadCreationFailed,
        ConcurrencyError::Type::Other
    };

    for (auto type : types) {
        ConcurrencyError ex(type, "test details");
        EXPECT_EQ(ex.type(), type);
        EXPECT_FALSE(std::string(ex.what()).empty());
    }
}

// ========== NumericalError tests ==========
TEST_F(RuntimeErrorTest, NumericalErrorOverflow) {
    NumericalError ex(NumericalError::Type::Overflow, "matrix multiplication");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("overflow") != std::string::npos);
    EXPECT_TRUE(message.find("matrix multiplication") != std::string::npos);
    EXPECT_EQ(ex.type(), NumericalError::Type::Overflow);
    EXPECT_EQ(ex.operation(), "matrix multiplication");
}

TEST_F(RuntimeErrorTest, NumericalErrorWithDetails) {
    NumericalError ex(NumericalError::Type::DivisionByZero, "determinant calculation", "singular matrix");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("Division by zero") != std::string::npos);
    EXPECT_TRUE(message.find("determinant calculation") != std::string::npos);
    EXPECT_TRUE(message.find("singular matrix") != std::string::npos);
}

TEST_F(RuntimeErrorTest, NumericalErrorAllTypes) {
    std::vector<NumericalError::Type> types = {
        NumericalError::Type::Overflow,
        NumericalError::Type::Underflow,
        NumericalError::Type::DivisionByZero,
        NumericalError::Type::InvalidOperation,
        NumericalError::Type::Convergence,
        NumericalError::Type::Singularity,
        NumericalError::Type::Other
    };

    for (auto type : types) {
        NumericalError ex(type, "test_operation");
        EXPECT_EQ(ex.type(), type);
        EXPECT_EQ(ex.operation(), "test_operation");
        EXPECT_FALSE(std::string(ex.what()).empty());
    }
}

// ========== Exception throwing and catching tests ==========
TEST_F(RuntimeErrorTest, ThrowAndCatchRuntimeError) {
    bool caught = false;
    try {
        throw RuntimeError("Test runtime error");
    } catch (const RuntimeError& e) {
        caught = true;
        EXPECT_STREQ(e.what(), "Test runtime error");
        EXPECT_EQ(e.code(), ErrorCode::Unknown);
    }
    EXPECT_TRUE(caught);
}

TEST_F(RuntimeErrorTest, ThrowAndCatchSpecificError) {
    bool caught_io = false;
    bool caught_runtime = false;

    try {
        throw IOError(IOError::Operation::Read, "test.txt", "file not found");
    } catch (const IOError& e) {
        caught_io = true;
        EXPECT_EQ(e.operation(), IOError::Operation::Read);
        EXPECT_EQ(e.resource(), "test.txt");
    } catch (const RuntimeError& e) {
        caught_runtime = true;
        FAIL() << "Should have caught IOError specifically";
    }

    EXPECT_TRUE(caught_io);
    EXPECT_FALSE(caught_runtime);
}

TEST_F(RuntimeErrorTest, ThrowAndCatchAsException) {
    bool caught = false;
    try {
        throw TimeoutError("operation", 1000ms);
    } catch (const Exception& e) {
        caught = true;
        EXPECT_EQ(e.code(), ErrorCode::Unknown);
    }
    EXPECT_TRUE(caught);
}

// ========== Polymorphic behavior tests ==========
TEST_F(RuntimeErrorTest, PolymorphicBehavior) {
    std::vector<std::unique_ptr<RuntimeError>> errors;

    errors.push_back(std::make_unique<IOError>(IOError::Operation::Read, "file.txt", "not found"));
    errors.push_back(std::make_unique<TimeoutError>("operation", 1000ms));
    errors.push_back(std::make_unique<PermissionError>(PermissionError::Permission::Write, "file.txt"));

    for (const auto& error : errors) {
        EXPECT_EQ(error->code(), ErrorCode::Unknown);
        EXPECT_FALSE(std::string(error->what()).empty());
    }
}

// ========== Real-world scenario tests ==========
TEST_F(RuntimeErrorTest, FileOperationErrors) {
    auto safe_file_read = [](const std::string& filename) -> std::string {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw IOError(IOError::Operation::Open, filename, "file not found");
        }

        std::string content;
        if (!std::getline(file, content)) {
            throw IOError(IOError::Operation::Read, filename, "read failed");
        }

        return content;
    };

    EXPECT_THROW(safe_file_read("/nonexistent/file.txt"), IOError);
}

TEST_F(RuntimeErrorTest, NetworkOperationSimulation) {
    auto simulate_network_request = [](const std::string& url, bool should_timeout) {
        if (should_timeout) {
            throw TimeoutError("HTTP request to " + url, 5000ms);
        }
        if (url.find("invalid") != std::string::npos) {
            throw NetworkError(NetworkError::Type::HostNotFound, url);
        }
        return "Success";
    };

    EXPECT_NO_THROW(simulate_network_request("https://example.com", false));
    EXPECT_THROW(simulate_network_request("https://example.com", true), TimeoutError);
    EXPECT_THROW(simulate_network_request("https://invalid.domain", false), NetworkError);
}

// ========== Context and formatting tests ==========
TEST_F(RuntimeErrorTest, ErrorWithContext) {
    IOError ex(IOError::Operation::Write, "output.txt", "disk full");
    ex.with_context("in function save_document")
      .with_context("called from user interface");

    std::string full_msg = ex.full_message();
    EXPECT_TRUE(full_msg.find("output.txt") != std::string::npos);
    EXPECT_TRUE(full_msg.find("save_document") != std::string::npos);
    EXPECT_TRUE(full_msg.find("user interface") != std::string::npos);
}

// ========== Edge cases and stress tests ==========
TEST_F(RuntimeErrorTest, VeryLongErrorMessages) {
    std::string long_resource(1000, 'a');
    std::string long_reason(1000, 'b');

    IOError ex(IOError::Operation::Read, long_resource, long_reason);
    std::string message = ex.what();

    EXPECT_TRUE(message.find(long_resource) != std::string::npos);
    EXPECT_TRUE(message.find(long_reason) != std::string::npos);
}

TEST_F(RuntimeErrorTest, UnicodeInErrorMessages) {
    NotFoundError ex(NotFoundError::ResourceType::File, "файл.txt");
    std::string message = ex.what();

    EXPECT_TRUE(message.find("файл.txt") != std::string::npos);
}

TEST_F(RuntimeErrorTest, EmptyStringsHandling) {
    ConfigurationError ex("", "");
    EXPECT_NO_THROW(ex.what());
    EXPECT_EQ(ex.parameter(), "");
}

TEST_F(RuntimeErrorTest, ExceptionChaining) {
    IOError inner(IOError::Operation::Read, "config.txt", "file corrupted");
    ConfigurationError outer("database_url", "invalid configuration");
    outer.with_nested(inner);

    std::string full_msg = outer.full_message();
    EXPECT_TRUE(full_msg.find("database_url") != std::string::npos);
    EXPECT_TRUE(full_msg.find("config.txt") != std::string::npos);
}

// ========== Performance and threading tests ==========
TEST_F(RuntimeErrorTest, ConcurrentExceptionCreation) {
    const int num_threads = 10;
    const int exceptions_per_thread = 100;
    std::vector<std::thread> threads;
    std::atomic<int> total_exceptions{0};

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&total_exceptions, exceptions_per_thread, i]() {
            for (int j = 0; j < exceptions_per_thread; ++j) {
                try {
                    throw RuntimeError("Thread " + std::to_string(i) + " exception " + std::to_string(j));
                } catch (const RuntimeError& e) {
                    total_exceptions++;
                    EXPECT_FALSE(std::string(e.what()).empty());
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(total_exceptions, num_threads * exceptions_per_thread);
}

TEST_F(RuntimeErrorTest, ErrorCodeConsistency) {
    std::vector<std::string> messages = {
        "Database connection failed",
        "Network timeout",
        "File not found",
        "Permission denied",
        "Resource exhausted"
    };

    for (const auto& msg : messages) {
        RuntimeError ex(msg);
        EXPECT_EQ(ex.code(), ErrorCode::Unknown);
        EXPECT_STREQ(ex.what(), msg.c_str());
    }
}

TEST_F(RuntimeErrorTest, DifferentMessages) {
    RuntimeError ex1("First error");
    RuntimeError ex2("Second error");
    RuntimeError ex3("Third error");

    EXPECT_STREQ(ex1.what(), "First error");
    EXPECT_STREQ(ex2.what(), "Second error");
    EXPECT_STREQ(ex3.what(), "Third error");

    EXPECT_NE(std::string(ex1.what()), std::string(ex2.what()));
    EXPECT_NE(std::string(ex2.what()), std::string(ex3.what()));
}