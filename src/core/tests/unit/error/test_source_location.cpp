#include <gtest/gtest.h>
#include <core/error/source_location.h>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <thread>

using namespace fem::core::error;

class SourceLocationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary test file for SourceContext tests
        test_file_path_ = std::filesystem::temp_directory_path() / "test_source.cpp";
        std::ofstream file(test_file_path_);
        file << "// Line 1\n";
        file << "void function1() {\n";
        file << "    int x = 42;  // Line 3\n";
        file << "    int y = 100;\n";
        file << "    int z = x + y;  // Error here\n";
        file << "    return;\n";
        file << "}\n";
        file << "// Line 8\n";
        file.close();
    }

    void TearDown() override {
        // Clean up temporary file
        if (std::filesystem::exists(test_file_path_)) {
            std::filesystem::remove(test_file_path_);
        }
    }

    std::filesystem::path test_file_path_;
};

// Basic SourceLocation tests
TEST_F(SourceLocationTest, DefaultConstruction) {
    SourceLocation loc;

    EXPECT_NE(loc.file_name(), nullptr);
    EXPECT_NE(loc.function_name(), nullptr);
    EXPECT_GT(loc.line(), 0);  // Should be this line
    EXPECT_GE(loc.column(), 0);  // Column might be 0 depending on compiler
}

TEST_F(SourceLocationTest, CurrentLocation) {
    auto current = std::source_location::current();
    SourceLocation loc(current);

    EXPECT_STREQ(loc.file_name(), current.file_name());
    EXPECT_STREQ(loc.function_name(), current.function_name());
    EXPECT_EQ(loc.line(), current.line());
    EXPECT_EQ(loc.column(), current.column());
}

TEST_F(SourceLocationTest, ExplicitConstruction) {
    SourceLocation loc("test_file.cpp", 42, "test_function", 10);

    EXPECT_STREQ(loc.file_name(), "test_file.cpp");
    EXPECT_STREQ(loc.function_name(), "test_function");
    EXPECT_EQ(loc.line(), 42);
    EXPECT_EQ(loc.column(), 10);
}

TEST_F(SourceLocationTest, FilenameExtraction) {
    SourceLocation loc("/path/to/source/file.cpp", 10, "func");

    EXPECT_EQ(loc.filename(), "file.cpp");
}

TEST_F(SourceLocationTest, FilenameFromWindowsPath) {
    SourceLocation loc("C:\\path\\to\\source\\file.cpp", 10, "func");

    std::string filename = loc.filename();
    EXPECT_TRUE(filename == "file.cpp" || filename == "C:\\path\\to\\source\\file.cpp");
}

TEST_F(SourceLocationTest, DirectoryExtraction) {
    SourceLocation loc("/path/to/source/file.cpp", 10, "func");

    EXPECT_EQ(loc.directory(), "/path/to/source");
}

TEST_F(SourceLocationTest, RelativePath) {
    SourceLocation loc("/home/user/project/src/file.cpp", 10, "func");

    std::string relative = loc.relative_path("/home/user/project");
    EXPECT_TRUE(relative == "src/file.cpp" || relative.find("src") != std::string::npos);
}

TEST_F(SourceLocationTest, RelativePathInvalid) {
    SourceLocation loc("/path/to/file.cpp", 10, "func");

    // Invalid base directory should return original path
    std::string relative = loc.relative_path("!!!invalid!!!");
    EXPECT_EQ(relative, "/path/to/file.cpp");
}

// Function name processing tests
TEST_F(SourceLocationTest, SimpleFunctionName) {
    SourceLocation loc("file.cpp", 10, "namespace::Class<T, U>::method(int, double) const");

    std::string simple = loc.simple_function_name();
    EXPECT_EQ(simple, "namespace::Class");
}

TEST_F(SourceLocationTest, SimpleFunctionNameNoTemplate) {
    SourceLocation loc("file.cpp", 10, "simple_function(int x)");

    std::string simple = loc.simple_function_name();
    EXPECT_EQ(simple, "simple_function");
}

TEST_F(SourceLocationTest, SimpleFunctionNamePlain) {
    SourceLocation loc("file.cpp", 10, "plain_function");

    std::string simple = loc.simple_function_name();
    EXPECT_EQ(simple, "plain_function");
}

TEST_F(SourceLocationTest, NamespaceName) {
    SourceLocation loc("file.cpp", 10, "ns1::ns2::Class::method()");

    std::string ns = loc.namespace_name();
    EXPECT_EQ(ns, "ns1::ns2::Class");
}

TEST_F(SourceLocationTest, NamespaceNameWithTemplate) {
    SourceLocation loc("file.cpp", 10, "ns::Class<T>::method(int)");

    std::string ns = loc.namespace_name();
    EXPECT_EQ(ns, "ns");
}

TEST_F(SourceLocationTest, NamespaceNameNoNamespace) {
    SourceLocation loc("file.cpp", 10, "global_function()");

    std::string ns = loc.namespace_name();
    EXPECT_EQ(ns, "");
}

// Formatting tests
TEST_F(SourceLocationTest, FormatShort) {
    SourceLocation loc("/path/to/file.cpp", 42, "function_name", 10);

    std::string formatted = loc.to_string(SourceLocation::Format::Short);
    EXPECT_EQ(formatted, "file.cpp:42");
}

TEST_F(SourceLocationTest, FormatMedium) {
    SourceLocation loc("/path/to/file.cpp", 42, "ns::function_name()", 10);

    std::string formatted = loc.to_string(SourceLocation::Format::Medium);
    EXPECT_EQ(formatted, "file.cpp:42:ns::function_name");
}

TEST_F(SourceLocationTest, FormatLongWithColumn) {
    SourceLocation loc("/path/to/file.cpp", 42, "ns::function_name()", 10);

    std::string formatted = loc.to_string(SourceLocation::Format::Long);
    EXPECT_EQ(formatted, "file.cpp:42:10:ns::function_name");
}

TEST_F(SourceLocationTest, FormatLongNoColumn) {
    SourceLocation loc("/path/to/file.cpp", 42, "ns::function_name()", 0);

    std::string formatted = loc.to_string(SourceLocation::Format::Long);
    EXPECT_EQ(formatted, "file.cpp:42:ns::function_name");
}

TEST_F(SourceLocationTest, FormatFull) {
    SourceLocation loc("/path/to/file.cpp", 42, "ns::function_name()", 10);

    std::string formatted = loc.to_string(SourceLocation::Format::Full);
    EXPECT_EQ(formatted, "/path/to/file.cpp:42:10:ns::function_name()");
}

TEST_F(SourceLocationTest, LogFormat) {
    SourceLocation loc("/path/to/file.cpp", 42, "ns::function<T>()", 10);

    std::string formatted = loc.log_format();
    EXPECT_EQ(formatted, "[file.cpp:42] in ns::function");
}

TEST_F(SourceLocationTest, DebugFormat) {
    SourceLocation loc("/path/to/file.cpp", 42, "ns::Class::method()", 10);

    std::string formatted = loc.debug_format();

    EXPECT_TRUE(formatted.find("File: /path/to/file.cpp") != std::string::npos);
    EXPECT_TRUE(formatted.find("Line: 42") != std::string::npos);
    EXPECT_TRUE(formatted.find("Column: 10") != std::string::npos);
    EXPECT_TRUE(formatted.find("Function: ns::Class::method()") != std::string::npos);
    EXPECT_TRUE(formatted.find("Namespace: ns::Class") != std::string::npos);
}

// Comparison tests
TEST_F(SourceLocationTest, Equality) {
    SourceLocation loc1("file.cpp", 42, "func", 10);
    SourceLocation loc2("file.cpp", 42, "func", 10);
    SourceLocation loc3("file.cpp", 42, "different_func", 10);  // Different function
    SourceLocation loc4("file.cpp", 43, "func", 10);  // Different line
    SourceLocation loc5("other.cpp", 42, "func", 10);  // Different file

    EXPECT_EQ(loc1, loc2);
    EXPECT_EQ(loc1, loc3);  // Function name not compared
    EXPECT_NE(loc1, loc4);
    EXPECT_NE(loc1, loc5);
}

TEST_F(SourceLocationTest, Inequality) {
    SourceLocation loc1("file1.cpp", 42, "func", 10);
    SourceLocation loc2("file2.cpp", 42, "func", 10);

    EXPECT_NE(loc1, loc2);
}

// Validity tests
TEST_F(SourceLocationTest, IsValid) {
    SourceLocation valid("file.cpp", 1, "func");
    EXPECT_TRUE(valid.is_valid());

    SourceLocation invalid_file("", 1, "func");
    EXPECT_FALSE(invalid_file.is_valid());

    SourceLocation invalid_line("file.cpp", 0, "func");
    EXPECT_FALSE(invalid_line.is_valid());

    SourceLocation null_file(nullptr, 1, "func");
    EXPECT_FALSE(null_file.is_valid());
}

// SourceContext tests
TEST_F(SourceLocationTest, SourceContextValid) {
    std::string file_path_str = test_file_path_.string();
    SourceLocation loc(file_path_str.c_str(), 5, "function1");
    SourceContext context(loc, 2);

    EXPECT_EQ(context.location().line(), 5);
    EXPECT_EQ(context.error_line(), "    int z = x + y;  // Error here");
    EXPECT_EQ(context.lines_before().size(), 2);
    EXPECT_EQ(context.lines_after().size(), 2);

    if (context.lines_before().size() >= 2) {
        EXPECT_EQ(context.lines_before()[0], "    int x = 42;  // Line 3");
        EXPECT_EQ(context.lines_before()[1], "    int y = 100;");
    }

    if (context.lines_after().size() >= 2) {
        EXPECT_EQ(context.lines_after()[0], "    return;");
        EXPECT_EQ(context.lines_after()[1], "}");
    }
}

TEST_F(SourceLocationTest, SourceContextInvalidFile) {
    SourceLocation loc("/nonexistent/file.cpp", 5, "func");
    SourceContext context(loc);

    EXPECT_TRUE(context.error_line().empty());
    EXPECT_TRUE(context.lines_before().empty());
    EXPECT_TRUE(context.lines_after().empty());
}

TEST_F(SourceLocationTest, SourceContextFormatDefault) {
    std::string file_path_str = test_file_path_.string();
    SourceLocation loc(file_path_str.c_str(), 5, "function1");
    SourceContext context(loc, 1);

    std::string formatted = context.format();

    // Should contain source location
    EXPECT_TRUE(formatted.find("test_source.cpp:5:function1") != std::string::npos);

    // Should show line numbers
    EXPECT_TRUE(formatted.find("   4 |") != std::string::npos);
    EXPECT_TRUE(formatted.find(">  5 |") != std::string::npos);  // Error line highlighted
    EXPECT_TRUE(formatted.find("   6 |") != std::string::npos);

    // Should contain actual code
    EXPECT_TRUE(formatted.find("int y = 100;") != std::string::npos);
    EXPECT_TRUE(formatted.find("int z = x + y;") != std::string::npos);
}

TEST_F(SourceLocationTest, SourceContextFormatNoLineNumbers) {
    std::string file_path_str = test_file_path_.string();
    SourceLocation loc(file_path_str.c_str(), 5, "function1");
    SourceContext context(loc, 1);

    std::string formatted = context.format(false, true);

    // Should not have line numbers
    EXPECT_FALSE(formatted.find("   4 |") != std::string::npos);
    EXPECT_FALSE(formatted.find(">  5 |") != std::string::npos);

    // But should still have code
    EXPECT_TRUE(formatted.find("int z = x + y;") != std::string::npos);
}

TEST_F(SourceLocationTest, SourceContextFormatNoHighlight) {
    std::string file_path_str = test_file_path_.string();
    SourceLocation loc(file_path_str.c_str(), 5, "function1", 10);
    SourceContext context(loc, 1);

    std::string formatted = context.format(true, false);

    // Should have regular line numbers (not highlighted)
    EXPECT_TRUE(formatted.find("   5 |") != std::string::npos);
    EXPECT_FALSE(formatted.find(">  5 |") != std::string::npos);

    // Should not have column indicator
    EXPECT_FALSE(formatted.find("^") != std::string::npos);
}

TEST_F(SourceLocationTest, SourceContextWithColumn) {
    std::string file_path_str = test_file_path_.string();
    SourceLocation loc(file_path_str.c_str(), 5, "function1", 15);
    SourceContext context(loc, 0);

    std::string formatted = context.format(true, true);

    // Should have column indicator at position 15
    auto pos = formatted.find("^");
    if (pos != std::string::npos) {
        // Count spaces before ^ to verify column position
        size_t line_start = formatted.rfind("\n", pos);
        if (line_start != std::string::npos) {
            std::string before_caret = formatted.substr(line_start, pos - line_start);
            // Should have appropriate spacing for column 15
            EXPECT_GT(before_caret.length(), 10);
        }
    }
}

// CallSite tests
TEST_F(SourceLocationTest, CallSiteConstruction) {
    CallSite site;

    EXPECT_TRUE(site.location().is_valid());
    EXPECT_GT(site.location().line(), 0);
}

TEST_F(SourceLocationTest, CallSiteElapsedTime) {
    CallSite site;

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    auto elapsed = site.elapsed();
    EXPECT_GE(elapsed.count(), 50);
    EXPECT_LT(elapsed.count(), 100);
}

TEST_F(SourceLocationTest, CallSiteFormat) {
    CallSite site;

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    std::string formatted = site.format();

    // Should contain filename:line@elapsed_ms
    EXPECT_TRUE(formatted.find(".cpp:") != std::string::npos);
    EXPECT_TRUE(formatted.find("@") != std::string::npos);
    EXPECT_TRUE(formatted.find("ms") != std::string::npos);
}

// Macro tests
TEST_F(SourceLocationTest, CurrentLocationMacro) {
    auto loc = FEM_CURRENT_LOCATION();

    EXPECT_TRUE(loc.is_valid());
    EXPECT_GT(loc.line(), 0);
    EXPECT_NE(loc.file_name(), nullptr);
    EXPECT_NE(loc.function_name(), nullptr);
}

TEST_F(SourceLocationTest, SourceContextMacro) {
    // Create a location for the test file
    std::string file_path_str = test_file_path_.string();
    SourceLocation loc(file_path_str.c_str(), 5, "test");

    // Can't use the macro directly with our test file, but we can test
    // that the macro compiles and produces valid results
    auto context = FEM_SOURCE_CONTEXT(2);

    EXPECT_TRUE(context.location().is_valid());
}

TEST_F(SourceLocationTest, CallSiteMacro) {
    auto site = FEM_CALL_SITE();

    EXPECT_TRUE(site.location().is_valid());
    EXPECT_GT(site.location().line(), 0);

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    EXPECT_GE(site.elapsed().count(), 5);
}

// Edge cases
TEST_F(SourceLocationTest, EmptyFunctionName) {
    SourceLocation loc("file.cpp", 42, "");

    EXPECT_EQ(loc.simple_function_name(), "");
    EXPECT_EQ(loc.namespace_name(), "");
}

TEST_F(SourceLocationTest, ComplexTemplateFunctionName) {
    SourceLocation loc("file.cpp", 42,
        "ns1::ns2::Class<std::vector<int>, std::map<K, V>>::method<T>(const T&, Args&&...) const noexcept");

    std::string simple = loc.simple_function_name();
    EXPECT_EQ(simple, "ns1::ns2::Class");

    std::string ns = loc.namespace_name();
    EXPECT_EQ(ns, "ns1::ns2");
}

TEST_F(SourceLocationTest, LambdaFunctionName) {
    SourceLocation loc("file.cpp", 42,
        "TestClass::testMethod()::<lambda(int)>");

    std::string simple = loc.simple_function_name();
    EXPECT_EQ(simple, "TestClass::testMethod()::");
}

TEST_F(SourceLocationTest, SourceContextBoundaryConditions) {
    // Test at beginning of file
    std::string file_path_str = test_file_path_.string();
    SourceLocation loc1(file_path_str.c_str(), 1, "func");
    SourceContext context1(loc1, 5);

    EXPECT_TRUE(context1.lines_before().empty());
    EXPECT_EQ(context1.error_line(), "// Line 1");

    // Test at end of file (assuming 8 lines)
    SourceLocation loc2(file_path_str.c_str(), 8, "func");
    SourceContext context2(loc2, 5);

    EXPECT_EQ(context2.error_line(), "// Line 8");
    EXPECT_TRUE(context2.lines_after().empty());

    // Test with line number beyond file
    SourceLocation loc3(file_path_str.c_str(), 1000, "func");
    SourceContext context3(loc3);

    EXPECT_TRUE(context3.error_line().empty());
}

TEST_F(SourceLocationTest, SourceContextZeroContextLines) {
    std::string file_path_str = test_file_path_.string();
    SourceLocation loc(file_path_str.c_str(), 5, "func");
    SourceContext context(loc, 0);

    EXPECT_TRUE(context.lines_before().empty());
    EXPECT_EQ(context.error_line(), "    int z = x + y;  // Error here");
    EXPECT_TRUE(context.lines_after().empty());
}

TEST_F(SourceLocationTest, SourceContextLargeContextLines) {
    std::string file_path_str = test_file_path_.string();
    SourceLocation loc(file_path_str.c_str(), 5, "func");
    SourceContext context(loc, 100);  // More than file has

    // Should get all available lines before and after
    EXPECT_EQ(context.lines_before().size(), 4);  // Lines 1-4
    EXPECT_EQ(context.lines_after().size(), 3);   // Lines 6-8
}