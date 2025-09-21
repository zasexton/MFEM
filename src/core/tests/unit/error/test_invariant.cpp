#include <gtest/gtest.h>
#include <core/error/invariant.h>
#include <vector>
#include <set>
#include <algorithm>

using namespace fem::core::error;

class InvariantTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// InvariantChecker tests
TEST_F(InvariantTest, InvariantChecker_CheckAll_AllPass) {
    InvariantChecker checker;

    checker.add_invariant("inv1", []() { return true; });
    checker.add_invariant("inv2", []() { return true; });
    checker.add_invariant("inv3", []() { return true; });

    EXPECT_NO_THROW(checker.check_all());
    EXPECT_EQ(3u, checker.count());
}

TEST_F(InvariantTest, InvariantChecker_CheckAll_SomeFail) {
    InvariantChecker checker;

    checker.add_invariant("inv1", []() { return true; });
    checker.add_invariant("inv2", []() { return false; });
    checker.add_invariant("inv3", []() { return false; });

    EXPECT_THROW(checker.check_all(), InvariantError);
}

TEST_F(InvariantTest, InvariantChecker_CheckAll_WithDiagnostics) {
    InvariantChecker checker;

    checker.add_invariant(
        "inv1",
        []() { return false; },
        []() { return "diagnostic info"; }
    );

    try {
        checker.check_all();
        FAIL() << "Expected InvariantError to be thrown";
    } catch (const InvariantError& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("inv1") != std::string::npos);
        EXPECT_TRUE(msg.find("diagnostic info") != std::string::npos);
    }
}

TEST_F(InvariantTest, InvariantChecker_CheckSpecific_Pass) {
    InvariantChecker checker;

    checker.add_invariant("inv1", []() { return true; });
    checker.add_invariant("inv2", []() { return false; });

    EXPECT_NO_THROW(checker.check("inv1"));
}

TEST_F(InvariantTest, InvariantChecker_CheckSpecific_Fail) {
    InvariantChecker checker;

    checker.add_invariant("inv1", []() { return false; });

    EXPECT_THROW(checker.check("inv1"), InvariantError);
}

TEST_F(InvariantTest, InvariantChecker_CheckSpecific_Unknown) {
    InvariantChecker checker;

    EXPECT_THROW(checker.check("unknown"), std::logic_error);
}

TEST_F(InvariantTest, InvariantChecker_DebugCheck) {
    InvariantChecker checker;

    checker.add_invariant("inv1", []() { return true; });

#ifndef NDEBUG
    EXPECT_NO_THROW(checker.debug_check());
#else
    // In release mode, debug_check does nothing
    EXPECT_NO_THROW(checker.debug_check());
#endif
}

TEST_F(InvariantTest, InvariantChecker_Clear) {
    InvariantChecker checker;

    checker.add_invariant("inv1", []() { return true; });
    checker.add_invariant("inv2", []() { return true; });

    EXPECT_EQ(2u, checker.count());

    checker.clear();
    EXPECT_EQ(0u, checker.count());
}

// ClassInvariant tests
class TestClass {
public:
    int value;
    std::string name;
    std::vector<int> items;

    TestClass(int v = 0, const std::string& n = "")
        : value(v), name(n) {}

    bool is_valid() const { return value >= 0; }
    int get_sum() const {
        int sum = 0;
        for (int item : items) sum += item;
        return sum;
    }
};

TEST_F(InvariantTest, ClassInvariant_MemberInvariant_Valid) {
    TestClass obj(42, "test");
    ClassInvariant<TestClass> inv(&obj, "TestClass");

    inv.require(&TestClass::value,
                std::function<bool(const int&)>([](const int& v) { return v > 0; }),
                "value must be positive");

    EXPECT_NO_THROW(inv.check_all());
}

TEST_F(InvariantTest, ClassInvariant_MemberInvariant_Invalid) {
    TestClass obj(-5, "test");
    ClassInvariant<TestClass> inv(&obj, "TestClass");

    inv.require(&TestClass::value,
                std::function<bool(const int&)>([](const int& v) { return v > 0; }),
                "value must be positive");

    EXPECT_THROW(inv.check_all(), InvariantError);
}

TEST_F(InvariantTest, ClassInvariant_MethodInvariant_Valid) {
    TestClass obj(42, "test");
    ClassInvariant<TestClass> inv(&obj, "TestClass");

    inv.require_method(
        [](const TestClass* o) { return o->is_valid(); },
        "object must be valid");

    EXPECT_NO_THROW(inv.check_all());
}

TEST_F(InvariantTest, ClassInvariant_MethodInvariant_Invalid) {
    TestClass obj(-5, "test");
    ClassInvariant<TestClass> inv(&obj, "TestClass");

    inv.require_method(
        [](const TestClass* o) { return o->is_valid(); },
        "object must be valid");

    EXPECT_THROW(inv.check_all(), InvariantError);
}

TEST_F(InvariantTest, ClassInvariant_Guard) {
    TestClass obj(42, "test");
    ClassInvariant<TestClass> inv(&obj, "TestClass");

    inv.require(&TestClass::value,
                std::function<bool(const int&)>([](const int& v) { return v > 0; }),
                "value must be positive");

    {
        auto guard = inv.guard();
        // Guard checks invariants on construction and destruction
        obj.value = 10;  // Still valid
    }

    // Guard destructor checked invariants
    SUCCEED();
}

TEST_F(InvariantTest, ClassInvariant_Guard_ViolationInScope) {
    TestClass obj(42, "test");
    ClassInvariant<TestClass> inv(&obj, "TestClass");

    inv.require(&TestClass::value,
                std::function<bool(const int&)>([](const int& v) { return v > 0; }),
                "value must be positive");

    {
        auto guard = inv.guard();
        obj.value = -5;  // Violates invariant
    }
    // Guard destructor suppresses exception

    // Invariant is violated but destructor doesn't throw
    EXPECT_THROW(inv.check_all(), InvariantError);
}

// LoopInvariant tests
TEST_F(InvariantTest, LoopInvariant_BasicLoop) {
    int counter = 0;
    int sum = 0;

    fem::core::error::LoopInvariant loop("test_loop");
    loop.require([&]() { return counter >= 0; }, "counter non-negative")
        .require([&]() { return sum >= 0; }, "sum non-negative");

    EXPECT_NO_THROW(loop.enter());

    for (int i = 0; i < 5; ++i) {
        counter = i;
        sum += i;
        EXPECT_NO_THROW(loop.iterate());
    }

    EXPECT_NO_THROW(loop.exit());
    EXPECT_EQ(5u, loop.current_iteration());
}

TEST_F(InvariantTest, LoopInvariant_ViolationDuringIteration) {
    int value = 10;

    fem::core::error::LoopInvariant loop("test_loop");
    loop.require([&]() { return value > 0; }, "value must be positive");

    EXPECT_NO_THROW(loop.enter());

    value = 5;
    EXPECT_NO_THROW(loop.iterate());

    value = -1;  // Violates invariant
    EXPECT_THROW(loop.iterate(), InvariantError);
}

TEST_F(InvariantTest, LoopInvariant_ViolationAtExit) {
    bool valid = true;

    fem::core::error::LoopInvariant loop("test_loop");
    loop.require([&]() { return valid; }, "must be valid");

    EXPECT_NO_THROW(loop.enter());
    EXPECT_NO_THROW(loop.iterate());

    valid = false;
    EXPECT_THROW(loop.exit(), InvariantError);
}

// DataStructureInvariant tests
TEST_F(InvariantTest, DataStructureInvariant_SizeInRange_Valid) {
    std::vector<int> vec{1, 2, 3, 4, 5};
    DataStructureInvariant<std::vector<int>> inv(&vec, "test_vector");

    inv.size_in_range(1, 10);
    EXPECT_NO_THROW(inv.check_all());
}

TEST_F(InvariantTest, DataStructureInvariant_SizeInRange_Invalid) {
    std::vector<int> vec{1, 2, 3, 4, 5};
    DataStructureInvariant<std::vector<int>> inv(&vec, "test_vector");

    inv.size_in_range(10, 20);
    EXPECT_THROW(inv.check_all(), InvariantError);
}

TEST_F(InvariantTest, DataStructureInvariant_IsSorted_Valid) {
    std::vector<int> vec{1, 2, 3, 4, 5};
    DataStructureInvariant<std::vector<int>> inv(&vec, "test_vector");

    inv.is_sorted();
    EXPECT_NO_THROW(inv.check_all());
}

TEST_F(InvariantTest, DataStructureInvariant_IsSorted_Invalid) {
    std::vector<int> vec{1, 3, 2, 5, 4};
    DataStructureInvariant<std::vector<int>> inv(&vec, "test_vector");

    inv.is_sorted();
    EXPECT_THROW(inv.check_all(), InvariantError);
}

TEST_F(InvariantTest, DataStructureInvariant_AllUnique_Valid) {
    std::vector<int> vec{1, 2, 3, 4, 5};
    DataStructureInvariant<std::vector<int>> inv(&vec, "test_vector");

    inv.all_unique();
    EXPECT_NO_THROW(inv.check_all());
}

TEST_F(InvariantTest, DataStructureInvariant_AllUnique_Invalid) {
    std::vector<int> vec{1, 2, 3, 2, 5};
    DataStructureInvariant<std::vector<int>> inv(&vec, "test_vector");

    inv.all_unique();
    EXPECT_THROW(inv.check_all(), InvariantError);
}

TEST_F(InvariantTest, DataStructureInvariant_AllElements_Valid) {
    std::vector<int> vec{2, 4, 6, 8, 10};
    DataStructureInvariant<std::vector<int>> inv(&vec, "test_vector");

    inv.all_elements([](const int& x) { return x % 2 == 0; }, "all even");
    EXPECT_NO_THROW(inv.check_all());
}

TEST_F(InvariantTest, DataStructureInvariant_AllElements_Invalid) {
    std::vector<int> vec{2, 3, 6, 8, 10};
    DataStructureInvariant<std::vector<int>> inv(&vec, "test_vector");

    inv.all_elements([](const int& x) { return x % 2 == 0; }, "all even");
    EXPECT_THROW(inv.check_all(), InvariantError);
}

// NumericInvariant tests
TEST_F(InvariantTest, NumericInvariant_Conserves_Valid) {
    double total_energy = 100.0;

    NumericInvariant inv("physics_simulation");
    inv.conserves<double>(
        [&]() { return total_energy; },
        "energy",
        100.0,
        0.001
    );

    EXPECT_NO_THROW(inv.check_all());

    total_energy = 100.0001;  // Within tolerance
    EXPECT_NO_THROW(inv.check_all());
}

TEST_F(InvariantTest, NumericInvariant_Conserves_Invalid) {
    double total_energy = 105.0;

    NumericInvariant inv("physics_simulation");
    inv.conserves<double>(
        [&]() { return total_energy; },
        "energy",
        100.0,
        0.001
    );

    EXPECT_THROW(inv.check_all(), InvariantError);
}

class MockMatrix {
public:
    size_t rows() const { return 3; }
    size_t cols() const { return 3; }
    double operator()(size_t i, size_t j) const {
        if (i == j) return diagonal_value_;
        return 0.0;
    }
    void set_diagonal(double v) { diagonal_value_ = v; }
private:
    double diagonal_value_ = 1.0;
};

TEST_F(InvariantTest, NumericInvariant_PositiveDefinite_Valid) {
    MockMatrix mat;
    mat.set_diagonal(1.0);

    NumericInvariant inv("matrix_ops");
    inv.positive_definite<MockMatrix>(
        [&]() { return mat; },
        "covariance"
    );

    EXPECT_NO_THROW(inv.check_all());
}

TEST_F(InvariantTest, NumericInvariant_PositiveDefinite_Invalid) {
    MockMatrix mat;
    mat.set_diagonal(-1.0);

    NumericInvariant inv("matrix_ops");
    inv.positive_definite<MockMatrix>(
        [&]() { return mat; },
        "covariance"
    );

    EXPECT_THROW(inv.check_all(), InvariantError);
}

TEST_F(InvariantTest, NumericInvariant_Bounded_Valid) {
    double temperature = 300.0;

    NumericInvariant inv("thermodynamics");
    inv.bounded<double>(
        [&]() { return temperature; },
        "temperature",
        0.0,
        1000.0
    );

    EXPECT_NO_THROW(inv.check_all());
}

TEST_F(InvariantTest, NumericInvariant_Bounded_Invalid) {
    double temperature = -50.0;

    NumericInvariant inv("thermodynamics");
    inv.bounded<double>(
        [&]() { return temperature; },
        "temperature",
        0.0,
        1000.0
    );

    EXPECT_THROW(inv.check_all(), InvariantError);
}

// SystemInvariantManager tests
TEST_F(InvariantTest, SystemInvariantManager_RegisterAndCheck) {
    auto& manager = SystemInvariantManager::instance();

    auto checker = std::make_shared<InvariantChecker>();
    checker->add_invariant("test", []() { return true; });

    manager.register_invariant("test_invariant", checker);

    EXPECT_NO_THROW(manager.check("test_invariant"));
    EXPECT_NO_THROW(manager.check_all());

    auto names = manager.get_names();
    EXPECT_TRUE(std::find(names.begin(), names.end(), "test_invariant") != names.end());

    manager.unregister_invariant("test_invariant");
}

TEST_F(InvariantTest, SystemInvariantManager_UnregisterRemovesInvariant) {
    auto& manager = SystemInvariantManager::instance();

    auto checker = std::make_shared<InvariantChecker>();
    checker->add_invariant("test", []() { return true; });

    manager.register_invariant("temp_invariant", checker);

    auto names_before = manager.get_names();
    EXPECT_TRUE(std::find(names_before.begin(), names_before.end(), "temp_invariant") != names_before.end());

    manager.unregister_invariant("temp_invariant");

    auto names_after = manager.get_names();
    EXPECT_TRUE(std::find(names_after.begin(), names_after.end(), "temp_invariant") == names_after.end());
}

// InvariantScope tests
TEST_F(InvariantTest, InvariantScope_Valid) {
    bool condition = true;

    {
        InvariantScope scope(
            [&]() { return condition; },
            "test condition"
        );
        // Scope is valid
    }

    SUCCEED();
}

TEST_F(InvariantTest, InvariantScope_InvalidAtEntry) {
    bool condition = false;

    EXPECT_THROW({
        InvariantScope scope(
            [&]() { return condition; },
            "test condition"
        );
    }, InvariantError);
}

TEST_F(InvariantTest, InvariantScope_InvalidAtExit) {
    bool condition = true;

    {
        InvariantScope scope(
            [&]() { return condition; },
            "test condition"
        );

        condition = false;  // Violates invariant
    }
    // Destructor doesn't throw, just logs

    SUCCEED();
}

// Helper function tests
TEST_F(InvariantTest, MakeClassInvariant_Helper) {
    TestClass obj(42, "test");
    auto inv = make_class_invariant(&obj, "TestClass");

    inv.require(&TestClass::value,
                std::function<bool(const int&)>([](const int& v) { return v > 0; }),
                "value must be positive");

    EXPECT_NO_THROW(inv.check_all());
}

TEST_F(InvariantTest, MakeContainerInvariant_Helper) {
    std::vector<int> vec{1, 2, 3, 4, 5};
    auto inv = make_container_invariant(&vec, "test_vector");

    inv.is_sorted();
    EXPECT_NO_THROW(inv.check_all());
}

TEST_F(InvariantTest, InvariantScope_Helper) {
    bool condition = true;

    {
        auto scope = invariant_scope(
            [&]() { return condition; },
            "test condition"
        );
    }

    SUCCEED();
}

// Integration test
TEST_F(InvariantTest, IntegrationExample_ComplexClassInvariants) {
    class BankAccount {
    public:
        BankAccount(double initial) : balance_(initial) {}

        void deposit(double amount) {
            if (amount <= 0) throw std::invalid_argument("Invalid amount");
            balance_ += amount;
        }

        void withdraw(double amount) {
            if (amount <= 0) throw std::invalid_argument("Invalid amount");
            if (amount > balance_) throw std::runtime_error("Insufficient funds");
            balance_ -= amount;
        }

        double balance() const { return balance_; }

    private:
        double balance_;
    };

    BankAccount account(1000.0);

    ClassInvariant<BankAccount> inv(&account, "BankAccount");
    inv.require_method(
        [](const BankAccount* acc) { return acc->balance() >= 0; },
        "balance must be non-negative");

    {
        auto guard = inv.guard();
        account.deposit(500.0);
    }

    EXPECT_EQ(1500.0, account.balance());

    {
        auto guard = inv.guard();
        account.withdraw(200.0);
    }

    EXPECT_EQ(1300.0, account.balance());

    EXPECT_NO_THROW(inv.check_all());
}