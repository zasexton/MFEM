#include <gtest/gtest.h>
#include <core/error/error_message.h>
#include <chrono>
#include <thread>
#include <regex>

using namespace fem::core::error;

class ErrorMessageTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Clear any existing catalog entries
        // Note: No clear method provided, so we work with what we have
    }
};

// Template tests
TEST_F(ErrorMessageTest, TemplateConstruction) {
    ErrorMessage::Template tmpl("Error: {message} at {location}");

    auto placeholders = tmpl.placeholders();
    ASSERT_EQ(placeholders.size(), 2);
    EXPECT_TRUE(std::find(placeholders.begin(), placeholders.end(), "message") != placeholders.end());
    EXPECT_TRUE(std::find(placeholders.begin(), placeholders.end(), "location") != placeholders.end());
}

TEST_F(ErrorMessageTest, TemplateDefaultConstruction) {
    ErrorMessage::Template tmpl;  // Default constructor

    auto placeholders = tmpl.placeholders();
    EXPECT_TRUE(placeholders.empty());
}

TEST_F(ErrorMessageTest, TemplateFormat) {
    ErrorMessage::Template tmpl("Error {code}: {message} at {location}");

    std::map<std::string, std::string> values = {
        {"code", "404"},
        {"message", "Not found"},
        {"location", "file.cpp:42"}
    };

    std::string result = tmpl.format(values);
    EXPECT_EQ(result, "Error 404: Not found at file.cpp:42");
}

TEST_F(ErrorMessageTest, TemplateFormatMissingValues) {
    ErrorMessage::Template tmpl("Error {code}: {message}");

    std::map<std::string, std::string> values = {
        {"code", "404"}
        // missing "message"
    };

    std::string result = tmpl.format(values);
    EXPECT_EQ(result, "Error 404: {message}");  // Placeholder remains
}

TEST_F(ErrorMessageTest, TemplateFormatExtraValues) {
    ErrorMessage::Template tmpl("Error {code}");

    std::map<std::string, std::string> values = {
        {"code", "404"},
        {"extra", "ignored"}
    };

    std::string result = tmpl.format(values);
    EXPECT_EQ(result, "Error 404");
}

TEST_F(ErrorMessageTest, TemplateValidation) {
    ErrorMessage::Template tmpl("Error {code}: {message}");

    std::map<std::string, std::string> complete = {
        {"code", "404"},
        {"message", "Not found"}
    };
    EXPECT_TRUE(tmpl.validate(complete));

    std::map<std::string, std::string> incomplete = {
        {"code", "404"}
    };
    EXPECT_FALSE(tmpl.validate(incomplete));

    std::map<std::string, std::string> extra = {
        {"code", "404"},
        {"message", "Not found"},
        {"extra", "value"}
    };
    EXPECT_TRUE(tmpl.validate(extra));  // Extra values are OK
}

TEST_F(ErrorMessageTest, TemplateWithDuplicatePlaceholders) {
    ErrorMessage::Template tmpl("{value} equals {value} and {other}");

    auto placeholders = tmpl.placeholders();
    EXPECT_EQ(placeholders.size(), 2);  // "value" should appear only once

    std::map<std::string, std::string> values = {
        {"value", "42"},
        {"other", "100"}
    };

    std::string result = tmpl.format(values);
    EXPECT_EQ(result, "42 equals 42 and 100");
}

TEST_F(ErrorMessageTest, TemplateWithNestedBraces) {
    ErrorMessage::Template tmpl("Error {{not_a_placeholder}} {real}");

    auto placeholders = tmpl.placeholders();
    EXPECT_EQ(placeholders.size(), 1);
    EXPECT_EQ(placeholders[0], "real");
}

// Builder tests
TEST_F(ErrorMessageTest, BuilderBasicMessage) {
    ErrorMessage::Builder builder;
    std::string msg = builder
        .set_code(ErrorCode::FileNotFound)
        .set_message("File not found")
        .build();

    EXPECT_TRUE(msg.find("[ERROR]") != std::string::npos);
    EXPECT_TRUE(msg.find("[E0200]") != std::string::npos);  // FileNotFound = 200
    EXPECT_TRUE(msg.find("File not found") != std::string::npos);
}

TEST_F(ErrorMessageTest, BuilderWithDetails) {
    ErrorMessage::Builder builder;
    std::string msg = builder
        .set_message("Operation failed")
        .set_details("Detailed explanation of the failure")
        .build();

    EXPECT_TRUE(msg.find("Operation failed") != std::string::npos);
    EXPECT_TRUE(msg.find("Details: Detailed explanation") != std::string::npos);
}

TEST_F(ErrorMessageTest, BuilderWithContext) {
    ErrorMessage::Builder builder;
    std::string msg = builder
        .set_message("Error occurred")
        .add_context("file", "test.cpp")
        .add_context("line", "42")
        .add_context("function", "process")
        .build();

    EXPECT_TRUE(msg.find("Error occurred") != std::string::npos);
    EXPECT_TRUE(msg.find("Context:") != std::string::npos);
    EXPECT_TRUE(msg.find("file: test.cpp") != std::string::npos);
    EXPECT_TRUE(msg.find("line: 42") != std::string::npos);
    EXPECT_TRUE(msg.find("function: process") != std::string::npos);
}

TEST_F(ErrorMessageTest, BuilderWithSuggestion) {
    ErrorMessage::Builder builder;
    std::string msg = builder
        .set_message("Invalid configuration")
        .set_suggestion("Check the configuration file format")
        .build();

    EXPECT_TRUE(msg.find("Invalid configuration") != std::string::npos);
    EXPECT_TRUE(msg.find("Suggestion: Check the configuration file format") != std::string::npos);
}

TEST_F(ErrorMessageTest, BuilderWithTimestamp) {
    auto now = std::chrono::system_clock::now();

    ErrorMessage::Builder builder;
    std::string msg = builder
        .set_message("Timed event")
        .set_timestamp(now)
        .build();

    EXPECT_TRUE(msg.find("Timed event") != std::string::npos);
    // Should have timestamp in format [YYYY-MM-DD HH:MM:SS]
    std::regex timestamp_regex(R"(\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\])");
    EXPECT_TRUE(std::regex_search(msg, timestamp_regex));
}

TEST_F(ErrorMessageTest, BuilderWithSeverity) {
    struct TestCase {
        ErrorMessage::Severity severity;
        std::string expected;
    };

    TestCase cases[] = {
        {ErrorMessage::Severity::Debug, "[DEBUG]"},
        {ErrorMessage::Severity::Info, "[INFO]"},
        {ErrorMessage::Severity::Warning, "[WARNING]"},
        {ErrorMessage::Severity::Error, "[ERROR]"},
        {ErrorMessage::Severity::Critical, "[CRITICAL]"},
        {ErrorMessage::Severity::Fatal, "[FATAL]"}
    };

    for (const auto& tc : cases) {
        ErrorMessage::Builder builder;
        std::string msg = builder
            .set_severity(tc.severity)
            .set_message("Test message")
            .build();

        EXPECT_TRUE(msg.find(tc.expected) != std::string::npos)
            << "Failed for severity: " << tc.expected;
    }
}

TEST_F(ErrorMessageTest, BuilderCompleteMessage) {
    auto now = std::chrono::system_clock::now();

    ErrorMessage::Builder builder;
    std::string msg = builder
        .set_code(ErrorCode::InvalidArgument)
        .set_severity(ErrorMessage::Severity::Warning)
        .set_message("Invalid input provided")
        .set_details("The input value exceeds maximum allowed")
        .add_context("parameter", "size")
        .add_context("value", "1000")
        .add_context("maximum", "100")
        .set_suggestion("Use a value between 0 and 100")
        .set_timestamp(now)
        .build();

    // Check all components are present
    EXPECT_TRUE(msg.find("[WARNING]") != std::string::npos);
    EXPECT_TRUE(msg.find("[E0003]") != std::string::npos);  // InvalidArgument = 3
    EXPECT_TRUE(msg.find("Invalid input provided") != std::string::npos);
    EXPECT_TRUE(msg.find("Details:") != std::string::npos);
    EXPECT_TRUE(msg.find("Context:") != std::string::npos);
    EXPECT_TRUE(msg.find("parameter: size") != std::string::npos);
    EXPECT_TRUE(msg.find("Suggestion:") != std::string::npos);
}

TEST_F(ErrorMessageTest, BuilderStructured) {
    auto now = std::chrono::system_clock::now();

    ErrorMessage::Builder builder;
    auto structured = builder
        .set_code(ErrorCode::OutOfMemory)
        .set_severity(ErrorMessage::Severity::Critical)
        .set_message("Memory allocation failed")
        .set_details("Attempted to allocate 1GB")
        .add_context("available", "256MB")
        .add_context("requested", "1024MB")
        .set_suggestion("Reduce memory usage")
        .set_timestamp(now)
        .build_structured();

    EXPECT_EQ(structured["severity"], "CRITICAL");
    EXPECT_EQ(structured["code"], "E0100");  // OutOfMemory = 100
    EXPECT_EQ(structured["message"], "Memory allocation failed");
    EXPECT_EQ(structured["details"], "Attempted to allocate 1GB");
    EXPECT_EQ(structured["suggestion"], "Reduce memory usage");
    EXPECT_EQ(structured["context.available"], "256MB");
    EXPECT_EQ(structured["context.requested"], "1024MB");
    EXPECT_TRUE(structured.find("timestamp") != structured.end());
}

// Formatter tests
TEST_F(ErrorMessageTest, FormatterFileError) {
    std::string msg = ErrorMessage::Formatter::file_error(
        "open", "/path/to/file.txt", "Permission denied");

    EXPECT_EQ(msg, "Failed to open file '/path/to/file.txt': Permission denied");
}

TEST_F(ErrorMessageTest, FormatterValidationError) {
    std::string msg = ErrorMessage::Formatter::validation_error(
        "email", "must be valid email format", "not-an-email");

    EXPECT_EQ(msg, "Validation failed for 'email': must be valid email format (got: not-an-email)");
}

TEST_F(ErrorMessageTest, FormatterRangeError) {
    std::string msg = ErrorMessage::Formatter::range_error(
        "temperature", 150, -273, 100);

    EXPECT_EQ(msg, "temperature out of range: 150 not in [-273, 100]");

    // Test with different types
    std::string msg_double = ErrorMessage::Formatter::range_error(
        "value", 3.14, 0.0, 1.0);

    EXPECT_TRUE(msg_double.find("3.14") != std::string::npos);
    EXPECT_TRUE(msg_double.find("[0, 1]") != std::string::npos);
}

TEST_F(ErrorMessageTest, FormatterNullPointerError) {
    std::string msg = ErrorMessage::Formatter::null_pointer_error("data_ptr");

    EXPECT_EQ(msg, "Null pointer: 'data_ptr'");
}

TEST_F(ErrorMessageTest, FormatterTypeMismatchError) {
    std::string msg = ErrorMessage::Formatter::type_mismatch_error(
        "function argument", "std::string", "int");

    EXPECT_EQ(msg, "Type mismatch in function argument: expected 'std::string', got 'int'");
}

TEST_F(ErrorMessageTest, FormatterTimeoutError) {
    std::string msg = ErrorMessage::Formatter::timeout_error(
        "database query", std::chrono::milliseconds(5000));

    EXPECT_EQ(msg, "Operation 'database query' timed out after 5000ms");
}

TEST_F(ErrorMessageTest, FormatterResourceError) {
    std::string msg = ErrorMessage::Formatter::resource_error(
        "Database", "connection", "server unreachable");

    EXPECT_EQ(msg, "Database connection failed: server unreachable");
}

TEST_F(ErrorMessageTest, FormatterConfigError) {
    std::string msg = ErrorMessage::Formatter::config_error(
        "max_connections", "1000", "exceeds system limit");

    EXPECT_EQ(msg, "Configuration error for 'max_connections' = '1000': exceeds system limit");
}

TEST_F(ErrorMessageTest, FormatterNumericalError) {
    std::string msg = ErrorMessage::Formatter::numerical_error(
        "matrix inversion", "singular matrix");

    EXPECT_EQ(msg, "Numerical error in matrix inversion: singular matrix");
}

TEST_F(ErrorMessageTest, FormatterDependencyError) {
    std::string msg = ErrorMessage::Formatter::dependency_error(
        "ModuleA", "ModuleB", "version mismatch");

    EXPECT_EQ(msg, "'ModuleA' dependency on 'ModuleB' failed: version mismatch");
}

// Catalog tests
TEST_F(ErrorMessageTest, CatalogRegisterAndGet) {
    auto& catalog = ErrorMessage::catalog();

    catalog.register_template("error.file_not_found",
                             "File '{filename}' not found in '{directory}'");

    auto tmpl = catalog.get_template("error.file_not_found");
    ASSERT_TRUE(tmpl.has_value());

    std::map<std::string, std::string> values = {
        {"filename", "config.ini"},
        {"directory", "/etc/app"}
    };

    std::string msg = tmpl->format(values);
    EXPECT_EQ(msg, "File 'config.ini' not found in '/etc/app'");
}

TEST_F(ErrorMessageTest, CatalogMultiLanguage) {
    auto& catalog = ErrorMessage::catalog();

    catalog.register_template("greeting", "Hello, {name}!", "en");
    catalog.register_template("greeting", "Bonjour, {name}!", "fr");
    catalog.register_template("greeting", "Hola, {name}!", "es");

    std::map<std::string, std::string> values = {{"name", "World"}};

    auto en_tmpl = catalog.get_template("greeting", "en");
    ASSERT_TRUE(en_tmpl.has_value());
    EXPECT_EQ(en_tmpl->format(values), "Hello, World!");

    auto fr_tmpl = catalog.get_template("greeting", "fr");
    ASSERT_TRUE(fr_tmpl.has_value());
    EXPECT_EQ(fr_tmpl->format(values), "Bonjour, World!");

    auto es_tmpl = catalog.get_template("greeting", "es");
    ASSERT_TRUE(es_tmpl.has_value());
    EXPECT_EQ(es_tmpl->format(values), "Hola, World!");
}

TEST_F(ErrorMessageTest, CatalogFormat) {
    auto& catalog = ErrorMessage::catalog();

    catalog.register_template("error.invalid_range",
                             "Value {value} is outside range [{min}, {max}]");

    std::map<std::string, std::string> values = {
        {"value", "150"},
        {"min", "0"},
        {"max", "100"}
    };

    std::string msg = catalog.format("error.invalid_range", values);
    EXPECT_EQ(msg, "Value 150 is outside range [0, 100]");
}

TEST_F(ErrorMessageTest, CatalogFormatMissingTemplate) {
    auto& catalog = ErrorMessage::catalog();

    std::map<std::string, std::string> values;
    std::string msg = catalog.format("non.existent.key", values);

    // Should return the key itself as fallback
    EXPECT_EQ(msg, "non.existent.key");
}

TEST_F(ErrorMessageTest, CatalogLoadFromConfig) {
    auto& catalog = ErrorMessage::catalog();

    std::map<std::string, std::map<std::string, std::string>> config = {
        {
            "error.network",
            {
                {"en", "Network error: {message}"},
                {"fr", "Erreur réseau: {message}"},
                {"de", "Netzwerkfehler: {message}"}
            }
        },
        {
            "error.auth",
            {
                {"en", "Authentication failed for user '{user}'"},
                {"fr", "Échec de l'authentification pour l'utilisateur '{user}'"}
            }
        }
    };

    catalog.load_from_config(config);

    // Test English
    std::map<std::string, std::string> network_values = {{"message", "timeout"}};
    EXPECT_EQ(catalog.format("error.network", network_values, "en"),
              "Network error: timeout");

    // Test French
    EXPECT_EQ(catalog.format("error.network", network_values, "fr"),
              "Erreur réseau: timeout");

    // Test German
    EXPECT_EQ(catalog.format("error.network", network_values, "de"),
              "Netzwerkfehler: timeout");

    // Test auth message
    std::map<std::string, std::string> auth_values = {{"user", "admin"}};
    EXPECT_EQ(catalog.format("error.auth", auth_values, "en"),
              "Authentication failed for user 'admin'");
}

// Helper function tests
TEST_F(ErrorMessageTest, FormatError) {
    std::map<std::string, std::string> context = {
        {"file", "data.txt"},
        {"line", "42"}
    };

    std::string msg = format_error(ErrorCode::FileNotFound,
                                   "Unable to locate file", context);

    EXPECT_TRUE(msg.find("[ERROR]") != std::string::npos);
    EXPECT_TRUE(msg.find("[E0200]") != std::string::npos);
    EXPECT_TRUE(msg.find("Unable to locate file") != std::string::npos);
    EXPECT_TRUE(msg.find("file: data.txt") != std::string::npos);
    EXPECT_TRUE(msg.find("line: 42") != std::string::npos);
}

TEST_F(ErrorMessageTest, DetailedError) {
    std::string msg = detailed_error(
        ErrorCode::InvalidArgument,
        "Invalid input",
        "The provided value is negative",
        "Use a positive value");

    EXPECT_TRUE(msg.find("[ERROR]") != std::string::npos);
    EXPECT_TRUE(msg.find("[E0003]") != std::string::npos);
    EXPECT_TRUE(msg.find("Invalid input") != std::string::npos);
    EXPECT_TRUE(msg.find("Details: The provided value is negative") != std::string::npos);
    EXPECT_TRUE(msg.find("Suggestion: Use a positive value") != std::string::npos);
}

TEST_F(ErrorMessageTest, DetailedErrorNoSuggestion) {
    std::string msg = detailed_error(
        ErrorCode::Unknown,
        "Something went wrong",
        "An unexpected error occurred");

    EXPECT_TRUE(msg.find("Something went wrong") != std::string::npos);
    EXPECT_TRUE(msg.find("Details: An unexpected error occurred") != std::string::npos);
    EXPECT_FALSE(msg.find("Suggestion:") != std::string::npos);
}

// Edge cases
TEST_F(ErrorMessageTest, TemplateEmptyPattern) {
    ErrorMessage::Template tmpl("");

    auto placeholders = tmpl.placeholders();
    EXPECT_TRUE(placeholders.empty());

    std::map<std::string, std::string> values;
    EXPECT_EQ(tmpl.format(values), "");
}

TEST_F(ErrorMessageTest, TemplateNoPlaceholders) {
    ErrorMessage::Template tmpl("This is a static message");

    auto placeholders = tmpl.placeholders();
    EXPECT_TRUE(placeholders.empty());

    std::map<std::string, std::string> values = {{"ignored", "value"}};
    EXPECT_EQ(tmpl.format(values), "This is a static message");
}

TEST_F(ErrorMessageTest, TemplateMalformedPlaceholders) {
    ErrorMessage::Template tmpl("Error {incomplete and {valid}");

    auto placeholders = tmpl.placeholders();
    EXPECT_EQ(placeholders.size(), 1);
    EXPECT_EQ(placeholders[0], "valid");
}

TEST_F(ErrorMessageTest, BuilderEmptyMessage) {
    ErrorMessage::Builder builder;
    std::string msg = builder.build();

    // Should still produce valid output with defaults
    EXPECT_TRUE(msg.find("[ERROR]") != std::string::npos);
    EXPECT_TRUE(msg.find(":") != std::string::npos);
}

TEST_F(ErrorMessageTest, BuilderChaining) {
    ErrorMessage::Builder builder;
    auto& same_builder = builder
        .set_code(ErrorCode::Success)
        .set_severity(ErrorMessage::Severity::Info)
        .set_message("Test");

    EXPECT_EQ(&builder, &same_builder);
}

TEST_F(ErrorMessageTest, CatalogThreadSafety) {
    auto& catalog = ErrorMessage::catalog();

    const int num_threads = 10;
    const int operations_per_thread = 100;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&catalog, t, operations_per_thread]() {
            for (int i = 0; i < operations_per_thread; ++i) {
                std::string key = "thread_" + std::to_string(t) + "_" + std::to_string(i);
                std::string pattern = "Message from thread {tid} operation {oid}";
                catalog.register_template(key, pattern);

                auto tmpl = catalog.get_template(key);
                EXPECT_TRUE(tmpl.has_value());
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Verify all templates were registered
    for (int t = 0; t < num_threads; ++t) {
        for (int i = 0; i < operations_per_thread; ++i) {
            std::string key = "thread_" + std::to_string(t) + "_" + std::to_string(i);
            auto tmpl = catalog.get_template(key);
            EXPECT_TRUE(tmpl.has_value());
        }
    }
}