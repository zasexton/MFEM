#include <gtest/gtest.h>
#include <core/error/exception_base.h>
#include <core/error/error_code.h>
#include <cerrno>
#include <system_error>
#include <cstring>
#include <vector>
#include <thread>
#include <memory>
#include <atomic>
#include <chrono>
#include <limits>

using namespace fem::core::error;

class SystemErrorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Save errno state
        saved_errno_ = errno;
    }

    void TearDown() override {
        // Restore errno state
        errno = saved_errno_;
    }

private:
    int saved_errno_;
};

// ========== Basic SystemError tests ==========
TEST_F(SystemErrorTest, SystemErrorConstruction) {
    SystemError ex("System operation failed", ENOENT);
    EXPECT_EQ(ex.code(), ErrorCode::SystemError);
    EXPECT_EQ(ex.system_error_code(), ENOENT);
}

TEST_F(SystemErrorTest, SystemErrorMessage) {
    SystemError ex("System operation failed", ENOENT);
    std::string message = ex.what();
    EXPECT_TRUE(message.find("System operation failed") != std::string::npos);
}

TEST_F(SystemErrorTest, SystemErrorInheritance) {
    SystemError ex("Test", ENOENT);
    Exception* base = &ex;
    std::exception* std_base = &ex;

    EXPECT_EQ(base->code(), ErrorCode::SystemError);
    EXPECT_STREQ(std_base->what(), "Test");
}

TEST_F(SystemErrorTest, SystemErrorWithSourceLocation) {
    SystemError ex("Test error", EINVAL);
    const auto& loc = ex.where();
    EXPECT_TRUE(loc.file_name() != nullptr);
    EXPECT_TRUE(loc.function_name() != nullptr);
    EXPECT_GT(loc.line(), 0u);
}

// ========== Exception throwing and catching tests ==========
TEST_F(SystemErrorTest, ThrowAndCatchSystemError) {
    bool caught = false;
    try {
        throw SystemError("Test system error", EINVAL);
    } catch (const SystemError& e) {
        caught = true;
        EXPECT_EQ(e.system_error_code(), EINVAL);
        EXPECT_EQ(e.code(), ErrorCode::SystemError);
    }
    EXPECT_TRUE(caught);
}

TEST_F(SystemErrorTest, ThrowAndCatchAsException) {
    bool caught = false;
    try {
        throw SystemError("Test", ENOENT);
    } catch (const Exception& e) {
        caught = true;
        EXPECT_EQ(e.code(), ErrorCode::SystemError);
    }
    EXPECT_TRUE(caught);
}

TEST_F(SystemErrorTest, ThrowAndCatchAsStdException) {
    bool caught = false;
    try {
        throw SystemError("Test system error", EACCES);
    } catch (const std::exception& e) {
        caught = true;
        EXPECT_TRUE(std::string(e.what()).find("Test system error") != std::string::npos);
    }
    EXPECT_TRUE(caught);
}

// ========== Common system error codes ==========
TEST_F(SystemErrorTest, FileNotFoundError) {
    SystemError ex("File operation", ENOENT);
    EXPECT_EQ(ex.system_error_code(), ENOENT);
    EXPECT_EQ(ex.code(), ErrorCode::SystemError);
}

TEST_F(SystemErrorTest, PermissionDeniedError) {
    SystemError ex("Permission check", EACCES);
    EXPECT_EQ(ex.system_error_code(), EACCES);
    EXPECT_EQ(ex.code(), ErrorCode::SystemError);
}

TEST_F(SystemErrorTest, InvalidArgumentError) {
    SystemError ex("System call", EINVAL);
    EXPECT_EQ(ex.system_error_code(), EINVAL);
    EXPECT_EQ(ex.code(), ErrorCode::SystemError);
}

TEST_F(SystemErrorTest, NotDirectoryError) {
    SystemError ex("Directory operation", ENOTDIR);
    EXPECT_EQ(ex.system_error_code(), ENOTDIR);
    EXPECT_EQ(ex.code(), ErrorCode::SystemError);
}

TEST_F(SystemErrorTest, IsDirectoryError) {
    SystemError ex("File operation", EISDIR);
    EXPECT_EQ(ex.system_error_code(), EISDIR);
    EXPECT_EQ(ex.code(), ErrorCode::SystemError);
}

TEST_F(SystemErrorTest, ResourceTemporarilyUnavailable) {
    SystemError ex("Resource access", EAGAIN);
    EXPECT_EQ(ex.system_error_code(), EAGAIN);
    EXPECT_EQ(ex.code(), ErrorCode::SystemError);
}

TEST_F(SystemErrorTest, InterruptedSystemCall) {
    SystemError ex("System call", EINTR);
    EXPECT_EQ(ex.system_error_code(), EINTR);
    EXPECT_EQ(ex.code(), ErrorCode::SystemError);
}

TEST_F(SystemErrorTest, NoSpaceLeftOnDevice) {
    SystemError ex("Write operation", ENOSPC);
    EXPECT_EQ(ex.system_error_code(), ENOSPC);
    EXPECT_EQ(ex.code(), ErrorCode::SystemError);
}

// ========== Error code variations ==========
TEST_F(SystemErrorTest, DifferentErrorCodes) {
    SystemError ex1("Test", ENOENT);
    SystemError ex2("Test", EACCES);
    SystemError ex3("Test", EINVAL);

    EXPECT_EQ(ex1.system_error_code(), ENOENT);
    EXPECT_EQ(ex2.system_error_code(), EACCES);
    EXPECT_EQ(ex3.system_error_code(), EINVAL);

    EXPECT_NE(ex1.system_error_code(), ex2.system_error_code());
    EXPECT_NE(ex2.system_error_code(), ex3.system_error_code());
}

TEST_F(SystemErrorTest, ErrorCodeConsistency) {
    std::vector<int> error_codes = {ENOENT, EACCES, EINVAL, ENOTDIR, EISDIR, EAGAIN, EINTR, ENOSPC};

    for (int err_code : error_codes) {
        SystemError ex("Test operation", err_code);
        EXPECT_EQ(ex.system_error_code(), err_code);
        EXPECT_EQ(ex.code(), ErrorCode::SystemError);
    }
}

TEST_F(SystemErrorTest, ZeroErrorCode) {
    SystemError ex("Success case", 0);
    EXPECT_EQ(ex.system_error_code(), 0);
    EXPECT_EQ(ex.code(), ErrorCode::SystemError);
}

TEST_F(SystemErrorTest, NegativeErrorCode) {
    SystemError ex("Negative error", -1);
    EXPECT_EQ(ex.system_error_code(), -1);
    EXPECT_EQ(ex.code(), ErrorCode::SystemError);
}

// ========== Context and formatting tests ==========
TEST_F(SystemErrorTest, ErrorWithContext) {
    SystemError ex("File operation failed", ENOENT);
    ex.with_context("in function open_file")
      .with_context("called from main");

    std::string full_msg = ex.full_message();
    EXPECT_TRUE(full_msg.find("File operation failed") != std::string::npos);
    EXPECT_TRUE(full_msg.find("open_file") != std::string::npos);
    EXPECT_TRUE(full_msg.find("called from main") != std::string::npos);
}

TEST_F(SystemErrorTest, MessageFormatting) {
    SystemError ex("Operation failed", EACCES);
    std::string message = ex.what();
    EXPECT_TRUE(message.find("Operation failed") != std::string::npos);
}

TEST_F(SystemErrorTest, FullMessageFormatting) {
    SystemError ex("File read failed", ENOENT);
    std::string full_msg = ex.full_message();

    EXPECT_TRUE(full_msg.find("File read failed") != std::string::npos);
    EXPECT_TRUE(full_msg.find("SystemError") != std::string::npos);
}

// ========== Real-world scenario tests ==========
TEST_F(SystemErrorTest, FileOperationScenarios) {
    auto simulate_file_operation = [](const std::string& operation, int error_code) {
        if (error_code != 0) {
            throw SystemError(operation + " failed", error_code);
        }
        return "success";
    };

    EXPECT_NO_THROW(simulate_file_operation("file read", 0));
    EXPECT_THROW(simulate_file_operation("file read", ENOENT), SystemError);
    EXPECT_THROW(simulate_file_operation("file write", EACCES), SystemError);
    EXPECT_THROW(simulate_file_operation("directory create", EEXIST), SystemError);
}

TEST_F(SystemErrorTest, NetworkOperationScenarios) {
    auto simulate_network_operation = [](const std::string& operation, int error_code) {
        if (error_code != 0) {
            throw SystemError(operation + " failed", error_code);
        }
        return "success";
    };

    EXPECT_NO_THROW(simulate_network_operation("socket connect", 0));
    EXPECT_THROW(simulate_network_operation("socket connect", ECONNREFUSED), SystemError);
    EXPECT_THROW(simulate_network_operation("socket bind", EADDRINUSE), SystemError);
}

// ========== System call simulation tests ==========
TEST_F(SystemErrorTest, FileSystemCallErrors) {
    std::vector<std::pair<std::string, int>> file_errors = {
        {"open", ENOENT},
        {"read", EBADF},
        {"write", ENOSPC},
        {"close", EIO},
        {"stat", ENOTDIR},
        {"chmod", EPERM},
        {"unlink", EACCES}
    };

    for (const auto& [syscall, error_code] : file_errors) {
        SystemError ex(syscall + " system call", error_code);
        EXPECT_EQ(ex.system_error_code(), error_code);
        EXPECT_TRUE(std::string(ex.what()).find(syscall) != std::string::npos);
    }
}

TEST_F(SystemErrorTest, MemoryOperationErrors) {
    std::vector<std::pair<std::string, int>> memory_errors = {
        {"malloc", ENOMEM},
        {"mmap", ENOMEM},
        {"mprotect", EACCES},
        {"munmap", EINVAL}
    };

    for (const auto& [operation, error_code] : memory_errors) {
        SystemError ex(operation + " failed", error_code);
        EXPECT_EQ(ex.system_error_code(), error_code);
        EXPECT_TRUE(std::string(ex.what()).find(operation) != std::string::npos);
    }
}

TEST_F(SystemErrorTest, ProcessOperationErrors) {
    std::vector<std::pair<std::string, int>> process_errors = {
        {"fork", EAGAIN},
        {"execve", ENOENT},
        {"waitpid", ECHILD},
        {"kill", ESRCH}
    };

    for (const auto& [operation, error_code] : process_errors) {
        SystemError ex(operation + " failed", error_code);
        EXPECT_EQ(ex.system_error_code(), error_code);
        EXPECT_TRUE(std::string(ex.what()).find(operation) != std::string::npos);
    }
}

// ========== Error propagation tests ==========
TEST_F(SystemErrorTest, ErrorPropagationChain) {
    auto low_level_operation = [](bool should_fail) {
        if (should_fail) {
            throw SystemError("Low-level I/O error", EIO);
        }
    };

    auto mid_level_operation = [&](bool should_fail) {
        try {
            low_level_operation(should_fail);
        } catch (const SystemError& e) {
            SystemError wrapped("Mid-level operation failed", ENOTRECOVERABLE);
            wrapped.with_nested(e);
            throw wrapped;
        }
    };

    EXPECT_NO_THROW(mid_level_operation(false));

    try {
        mid_level_operation(true);
        FAIL() << "Expected SystemError to be thrown";
    } catch (const SystemError& e) {
        EXPECT_EQ(e.system_error_code(), ENOTRECOVERABLE);
        std::string full_msg = e.full_message();
        EXPECT_TRUE(full_msg.find("Mid-level operation failed") != std::string::npos);
        EXPECT_TRUE(full_msg.find("Low-level I/O error") != std::string::npos);
    }
}

// ========== Threading and concurrency tests ==========
TEST_F(SystemErrorTest, ConcurrentSystemErrors) {
    const int num_threads = 10;
    const int operations_per_thread = 50;
    std::vector<std::thread> threads;
    std::atomic<int> total_errors{0};

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&total_errors, operations_per_thread, i]() {
            for (int j = 0; j < operations_per_thread; ++j) {
                try {
                    throw SystemError("Thread " + std::to_string(i) + " error " + std::to_string(j), ENOENT + (j % 10));
                } catch (const SystemError& e) {
                    total_errors++;
                    EXPECT_EQ(e.code(), ErrorCode::SystemError);
                    EXPECT_FALSE(std::string(e.what()).empty());
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(total_errors, num_threads * operations_per_thread);
}

// ========== Polymorphic behavior tests ==========
TEST_F(SystemErrorTest, PolymorphicBehavior) {
    std::vector<std::unique_ptr<Exception>> errors;

    errors.push_back(std::make_unique<SystemError>("File error", ENOENT));
    errors.push_back(std::make_unique<SystemError>("Permission error", EACCES));
    errors.push_back(std::make_unique<SystemError>("Invalid argument", EINVAL));

    for (const auto& error : errors) {
        EXPECT_EQ(error->code(), ErrorCode::SystemError);
        EXPECT_FALSE(std::string(error->what()).empty());
    }
}

// ========== Copy and move semantics tests ==========
TEST_F(SystemErrorTest, CopySemantics) {
    SystemError original("Original error", ENOENT);
    original.with_context("original context");

    SystemError copied = original;
    EXPECT_EQ(copied.system_error_code(), ENOENT);
    EXPECT_EQ(copied.code(), ErrorCode::SystemError);
    EXPECT_STREQ(copied.what(), original.what());

    std::string original_full = original.full_message();
    std::string copied_full = copied.full_message();
    EXPECT_EQ(original_full, copied_full);
}

TEST_F(SystemErrorTest, MoveSemantics) {
    SystemError original("Original error", EACCES);
    original.with_context("original context");

    std::string expected_message = original.what();
    std::string expected_full = original.full_message();

    SystemError moved = std::move(original);
    EXPECT_EQ(moved.system_error_code(), EACCES);
    EXPECT_EQ(moved.code(), ErrorCode::SystemError);
    EXPECT_STREQ(moved.what(), expected_message.c_str());
    EXPECT_EQ(moved.full_message(), expected_full);
}

// ========== Edge cases and error handling ==========
TEST_F(SystemErrorTest, EmptyMessage) {
    SystemError ex("", EINVAL);
    EXPECT_STREQ(ex.what(), "");
    EXPECT_EQ(ex.system_error_code(), EINVAL);
    EXPECT_EQ(ex.code(), ErrorCode::SystemError);
}

TEST_F(SystemErrorTest, LongMessage) {
    std::string long_message(1000, 'x');
    SystemError ex(long_message, ENOENT);

    EXPECT_EQ(std::string(ex.what()), long_message);
    EXPECT_EQ(ex.system_error_code(), ENOENT);
}

TEST_F(SystemErrorTest, UnicodeMessage) {
    SystemError ex("Ошибка системы", ENOENT);
    std::string message = ex.what();
    EXPECT_TRUE(message.find("Ошибка системы") != std::string::npos);
    EXPECT_EQ(ex.system_error_code(), ENOENT);
}

// ========== Error code boundary tests ==========
TEST_F(SystemErrorTest, MinMaxErrorCodes) {
    SystemError ex_min("Minimum error", std::numeric_limits<int>::min());
    SystemError ex_max("Maximum error", std::numeric_limits<int>::max());

    EXPECT_EQ(ex_min.system_error_code(), std::numeric_limits<int>::min());
    EXPECT_EQ(ex_max.system_error_code(), std::numeric_limits<int>::max());
    EXPECT_EQ(ex_min.code(), ErrorCode::SystemError);
    EXPECT_EQ(ex_max.code(), ErrorCode::SystemError);
}

// ========== Performance tests ==========
TEST_F(SystemErrorTest, ExceptionCreationPerformance) {
    const int num_exceptions = 1000;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_exceptions; ++i) {
        SystemError ex("Performance test " + std::to_string(i), ENOENT);
        std::string message = ex.what();
        EXPECT_FALSE(message.empty());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Should complete within reasonable time (adjust threshold as needed)
    EXPECT_LT(duration.count(), 1000); // Less than 1 second for 1000 exceptions
}