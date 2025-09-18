#include <gtest/gtest.h>
#include <core/error/error_category.h>
#include <core/error/error_code.h>
#include <memory>
#include <thread>
#include <vector>

using namespace fem::core::error;

class ErrorCategoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Clear registry for each test
        ErrorCategoryRegistry::instance().clear();
    }

    void TearDown() override {
        // Clean up registry after each test
        ErrorCategoryRegistry::instance().clear();
    }
};

// Custom error category for testing
class TestErrorCategory : public ExtendedErrorCategory {
public:
    TestErrorCategory() : ExtendedErrorCategory("TestErrorCategory") {}

    const char* name() const noexcept override {
        return "TestCategory";
    }

    std::string message(int code) const override {
        switch (code) {
            case 1: return "Test error 1";
            case 2: return "Test error 2";
            case 3: return "Test error 3";
            default: return "Unknown test error";
        }
    }

    std::error_condition default_error_condition(int code) const noexcept override {
        return std::error_condition(code, *this);
    }
};

// Another custom category for testing
class AnotherTestCategory : public ExtendedErrorCategory {
public:
    AnotherTestCategory() : ExtendedErrorCategory("AnotherTestCategory") {}

    const char* name() const noexcept override {
        return "AnotherTest";
    }

    std::string message(int code) const override {
        return "Another error: " + std::to_string(code);
    }
};

// Basic ErrorCategory tests
TEST_F(ErrorCategoryTest, CategoryName) {
    TestErrorCategory category;
    EXPECT_STREQ(category.name(), "TestCategory");
}

TEST_F(ErrorCategoryTest, CategoryMessage) {
    TestErrorCategory category;
    EXPECT_EQ(category.message(1), "Test error 1");
    EXPECT_EQ(category.message(2), "Test error 2");
    EXPECT_EQ(category.message(3), "Test error 3");
    EXPECT_EQ(category.message(99), "Unknown test error");
}

TEST_F(ErrorCategoryTest, CategoryComparison) {
    TestErrorCategory cat1;
    TestErrorCategory cat2;
    AnotherTestCategory cat3;

    // Same category type should compare equal by address
    EXPECT_EQ(&cat1, &cat1);
    EXPECT_NE(&cat1, &cat2);  // Different instances
    // Cast to common base type for comparison
    EXPECT_NE(static_cast<const std::error_category*>(&cat1),
              static_cast<const std::error_category*>(&cat3));  // Different types
}

TEST_F(ErrorCategoryTest, DefaultErrorCondition) {
    TestErrorCategory category;
    std::error_condition condition = category.default_error_condition(42);

    EXPECT_EQ(condition.value(), 42);
    EXPECT_EQ(&condition.category(), &category);
}

// ErrorCategoryRegistry tests
TEST_F(ErrorCategoryTest, RegisterCategory) {
    auto category = fem::core::base::make_object<TestErrorCategory>();

    ErrorCategoryRegistry::instance().register_category("test", category);

    auto retrieved = ErrorCategoryRegistry::instance().get_category("test");
    EXPECT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved, category);
}

TEST_F(ErrorCategoryTest, RegisterMultipleCategories) {
    auto cat1 = fem::core::base::make_object<TestErrorCategory>();
    auto cat2 = fem::core::base::make_object<AnotherTestCategory>();

    ErrorCategoryRegistry::instance().register_category("test1", cat1);
    ErrorCategoryRegistry::instance().register_category("test2", cat2);

    EXPECT_EQ(ErrorCategoryRegistry::instance().get_category("test1"), cat1);
    EXPECT_EQ(ErrorCategoryRegistry::instance().get_category("test2"), cat2);
}

TEST_F(ErrorCategoryTest, GetNonExistentCategory) {
    auto category = ErrorCategoryRegistry::instance().get_category("nonexistent");
    EXPECT_EQ(category, nullptr);
}

TEST_F(ErrorCategoryTest, RegisterDuplicateCategory) {
    auto cat1 = fem::core::base::make_object<TestErrorCategory>();
    auto cat2 = fem::core::base::make_object<AnotherTestCategory>();

    ErrorCategoryRegistry::instance().register_category("test", cat1);

    // Should return false when registering duplicate
    EXPECT_FALSE(
        ErrorCategoryRegistry::instance().register_category("test", cat2)
    );
}

TEST_F(ErrorCategoryTest, UnregisterCategory) {
    auto category = fem::core::base::make_object<TestErrorCategory>();

    ErrorCategoryRegistry::instance().register_category("test", category);
    EXPECT_NE(ErrorCategoryRegistry::instance().get_category("test"), nullptr);

    ErrorCategoryRegistry::instance().unregister_category("test");
    EXPECT_EQ(ErrorCategoryRegistry::instance().get_category("test"), nullptr);
}

TEST_F(ErrorCategoryTest, ListCategories) {
    auto cat1 = fem::core::base::make_object<TestErrorCategory>();
    auto cat2 = fem::core::base::make_object<AnotherTestCategory>();

    ErrorCategoryRegistry::instance().register_category("alpha", cat1);
    ErrorCategoryRegistry::instance().register_category("beta", cat2);

    auto categories = ErrorCategoryRegistry::instance().list_categories();

    EXPECT_EQ(categories.size(), 2);
    EXPECT_TRUE(std::find(categories.begin(), categories.end(), "alpha") != categories.end());
    EXPECT_TRUE(std::find(categories.begin(), categories.end(), "beta") != categories.end());
}

TEST_F(ErrorCategoryTest, ClearRegistry) {
    auto cat1 = fem::core::base::make_object<TestErrorCategory>();
    auto cat2 = fem::core::base::make_object<AnotherTestCategory>();

    ErrorCategoryRegistry::instance().register_category("test1", cat1);
    ErrorCategoryRegistry::instance().register_category("test2", cat2);

    EXPECT_EQ(ErrorCategoryRegistry::instance().list_categories().size(), 2);

    ErrorCategoryRegistry::instance().clear();

    EXPECT_EQ(ErrorCategoryRegistry::instance().list_categories().size(), 0);
    EXPECT_EQ(ErrorCategoryRegistry::instance().get_category("test1"), nullptr);
    EXPECT_EQ(ErrorCategoryRegistry::instance().get_category("test2"), nullptr);
}

// Test with std::error_code
TEST_F(ErrorCategoryTest, CreateErrorCode) {
    TestErrorCategory category;
    std::error_code ec(42, category);

    EXPECT_EQ(ec.value(), 42);
    EXPECT_EQ(&ec.category(), &category);
    EXPECT_EQ(ec.message(), "Unknown test error");
}

TEST_F(ErrorCategoryTest, CreateErrorCodeWithKnownValue) {
    TestErrorCategory category;
    std::error_code ec(1, category);

    EXPECT_EQ(ec.value(), 1);
    EXPECT_EQ(ec.message(), "Test error 1");
}

TEST_F(ErrorCategoryTest, ErrorCodeComparison) {
    TestErrorCategory cat1;
    AnotherTestCategory cat2;

    std::error_code ec1(1, cat1);
    std::error_code ec2(1, cat1);
    std::error_code ec3(2, cat1);
    std::error_code ec4(1, cat2);

    EXPECT_EQ(ec1, ec2);  // Same value and category
    EXPECT_NE(ec1, ec3);  // Different value
    EXPECT_NE(ec1, ec4);  // Different category
}

// Thread safety tests
TEST_F(ErrorCategoryTest, ThreadSafeRegistration) {
    const int num_threads = 10;
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([i]() {
            auto category = fem::core::base::make_object<TestErrorCategory>();
            std::string name = "category_" + std::to_string(i);

            // Registration returns bool, no exceptions
            ErrorCategoryRegistry::instance().register_category(name, category);
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Should have registered all categories
    auto categories = ErrorCategoryRegistry::instance().list_categories();
    EXPECT_EQ(categories.size(), num_threads);
}

TEST_F(ErrorCategoryTest, ThreadSafeAccess) {
    auto category = fem::core::base::make_object<TestErrorCategory>();
    ErrorCategoryRegistry::instance().register_category("shared", category);

    const int num_threads = 10;
    const int iterations = 100;
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([iterations]() {
            for (int j = 0; j < iterations; ++j) {
                auto cat = ErrorCategoryRegistry::instance().get_category("shared");
                EXPECT_NE(cat, nullptr);
                EXPECT_STREQ(cat->name(), "TestCategory");
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }
}

// Integration with fem::core::error::ErrorCode
TEST_F(ErrorCategoryTest, CoreErrorCategory) {
    std::error_code ec = make_error_code(ErrorCode::InvalidArgument);

    EXPECT_STREQ(ec.category().name(), "fem::core");
    EXPECT_FALSE(ec.message().empty());
}

// Custom domain-specific category
class DomainSpecificCategory : public ExtendedErrorCategory {
public:
    enum Code {
        Success = 0,
        ValidationFailed = 1,
        ProcessingError = 2,
        NetworkTimeout = 3
    };

    DomainSpecificCategory() : ExtendedErrorCategory("DomainSpecificCategory") {}

    const char* name() const noexcept override {
        return "DomainSpecific";
    }

    std::string message(int code) const override {
        switch (static_cast<Code>(code)) {
            case Success: return "Operation successful";
            case ValidationFailed: return "Validation failed";
            case ProcessingError: return "Processing error occurred";
            case NetworkTimeout: return "Network operation timed out";
            default: return "Unknown domain error";
        }
    }

    bool is_recoverable(int code) const noexcept override {
        return code != ProcessingError;
    }
};

TEST_F(ErrorCategoryTest, DomainSpecificUsage) {
    auto category = fem::core::base::make_object<DomainSpecificCategory>();
    ErrorCategoryRegistry::instance().register_category("domain", category);

    std::error_code ec1(DomainSpecificCategory::Success, *category);
    std::error_code ec2(DomainSpecificCategory::ValidationFailed, *category);
    std::error_code ec3(DomainSpecificCategory::ProcessingError, *category);

    EXPECT_EQ(ec1.message(), "Operation successful");
    EXPECT_EQ(ec2.message(), "Validation failed");
    EXPECT_EQ(ec3.message(), "Processing error occurred");

    EXPECT_FALSE(ec1);  // Success should evaluate to false
    EXPECT_TRUE(ec2);   // Error should evaluate to true
    EXPECT_TRUE(ec3);   // Error should evaluate to true

    // Test custom method
    EXPECT_TRUE(category->is_recoverable(DomainSpecificCategory::ValidationFailed));
    EXPECT_FALSE(category->is_recoverable(DomainSpecificCategory::ProcessingError));
}

// Test category equivalence
TEST_F(ErrorCategoryTest, CategoryEquivalence) {
    TestErrorCategory cat1;
    TestErrorCategory cat2;

    // Categories are compared by address (std::error_category comparison)
    EXPECT_FALSE(static_cast<const std::error_category&>(cat1) == static_cast<const std::error_category&>(cat2));
    EXPECT_TRUE(static_cast<const std::error_category&>(cat1) != static_cast<const std::error_category&>(cat2));
    EXPECT_TRUE(static_cast<const std::error_category&>(cat1) == static_cast<const std::error_category&>(cat1));

    // Error codes with same value but different categories are not equal
    std::error_code ec1(1, cat1);
    std::error_code ec2(1, cat2);
    EXPECT_NE(ec1, ec2);
}