/**
 * @file test_visitor.cpp
 * @brief Comprehensive unit tests for the Visitor pattern implementations
 *
 * Tests cover:
 * - BaseVisitor interface and type erasure
 * - Generic Visitor<T> with CRTP pattern
 * - Visitable<T> acceptance interface
 * - HierarchicalVisitor with tree traversal (pre/post/in-order)
 * - CompositeVisitor for sequential visitor application
 * - ConditionalVisitor with predicate filtering
 * - FunctionVisitor for lambda/function wrapping
 * - VisitorRegistry factory pattern
 * - VisitorCoordinator collection utilities
 * - Thread safety considerations
 * - Performance characteristics
 * - Memory management and RAII
 * - Error handling and edge cases
 * - Macro utilities for visitor declaration
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <typeindex>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>

#include "../../../base/visitor.h"

using namespace fem::core::base;
using namespace testing;

// ============================================================================
// Test Classes for Visitor Testing
// ============================================================================

// Base class for testing visitor pattern
class TestNode : public Visitable<TestNode> {
public:
    explicit TestNode(std::string name, int value = 0)
        : name_(std::move(name)), value_(value) {}

    virtual ~TestNode() = default;

    const std::string& get_name() const { return name_; }
    int get_value() const { return value_; }
    void set_value(int value) { value_ = value; }

    // Hierarchical support
    void add_child(std::shared_ptr<TestNode> child) {
        if (child) {
            children_.push_back(child);
            child->parent_ = shared_from_this();
        }
    }

    const std::vector<std::shared_ptr<TestNode>>& get_children() const { return children_; }
    std::weak_ptr<TestNode> get_parent() const { return parent_; }

    // Enable shared_from_this
    std::shared_ptr<TestNode> shared_from_this() { return std::shared_ptr<TestNode>(this, [](TestNode*){}); }

protected:
    std::string name_;
    int value_;
    std::vector<std::shared_ptr<TestNode>> children_;
    std::weak_ptr<TestNode> parent_;
};

// Derived test classes
class LeafNode : public TestNode {
public:
    explicit LeafNode(std::string name, int value = 0)
        : TestNode(std::move(name), value) {}

    std::string get_type() const { return "Leaf"; }
};

class BranchNode : public TestNode {
public:
    explicit BranchNode(std::string name, int value = 0)
        : TestNode(std::move(name), value) {}

    std::string get_type() const { return "Branch"; }
};

class SpecialNode : public TestNode {
public:
    explicit SpecialNode(std::string name, int value = 0, bool flag = false)
        : TestNode(std::move(name), value), special_flag_(flag) {}

    bool get_special_flag() const { return special_flag_; }
    void set_special_flag(bool flag) { special_flag_ = flag; }

private:
    bool special_flag_;
};

// Mock visitor for testing
class MockVisitor : public Visitor<TestNode> {
public:
    MOCK_METHOD(void, visit, (TestNode& node), (override));
    MOCK_METHOD(void, visit, (const TestNode& node), (override));
    MOCK_METHOD(std::string, get_name, (), (const, override));
    MOCK_METHOD(std::string, get_description, (), (const, override));
    MOCK_METHOD(bool, can_visit, (std::type_index type), (const, override));
};

// Concrete test visitors
class CountingVisitor : public Visitor<TestNode> {
public:
    explicit CountingVisitor(std::string name = "CountingVisitor") : name_(std::move(name)) {
        register_visitable_type<LeafNode>();
        register_visitable_type<BranchNode>();
        register_visitable_type<SpecialNode>();
    }

    void visit(TestNode& node) override {
        visit_count_++;
        visited_names_.push_back(node.get_name());
        node.set_value(node.get_value() + 1);
    }

    void visit(const TestNode& node) override {
        const_visit_count_++;
        const_visited_names_.push_back(node.get_name());
    }

    std::string get_name() const override { return name_; }

    // Test accessors
    int get_visit_count() const { return visit_count_; }
    int get_const_visit_count() const { return const_visit_count_; }
    const std::vector<std::string>& get_visited_names() const { return visited_names_; }
    const std::vector<std::string>& get_const_visited_names() const { return const_visited_names_; }

    void reset() {
        visit_count_ = 0;
        const_visit_count_ = 0;
        visited_names_.clear();
        const_visited_names_.clear();
    }

private:
    std::string name_;
    int visit_count_ = 0;
    int const_visit_count_ = 0;
    std::vector<std::string> visited_names_;
    std::vector<std::string> const_visited_names_;
};

class ValueModifyingVisitor : public Visitor<TestNode> {
public:
    explicit ValueModifyingVisitor(int increment) : increment_(increment) {}

    void visit(TestNode& node) override {
        node.set_value(node.get_value() + increment_);
        total_modifications_++;
    }

    using Visitor<TestNode>::visit; // Bring base class overloads into scope

    std::string get_name() const override {
        return "ValueModifyingVisitor(+" + std::to_string(increment_) + ")";
    }

    int get_total_modifications() const { return total_modifications_; }
    void reset() { total_modifications_ = 0; }

private:
    int increment_;
    int total_modifications_ = 0;
};

class StringCollectingVisitor : public Visitor<TestNode> {
public:
    void visit(TestNode& node) override {
        collected_strings_.push_back(node.get_name() + ":" + std::to_string(node.get_value()));
    }

    void visit(const TestNode& node) override {
        const_collected_strings_.push_back(node.get_name() + ":" + std::to_string(node.get_value()));
    }

    std::string get_name() const override { return "StringCollectingVisitor"; }

    const std::vector<std::string>& get_collected_strings() const { return collected_strings_; }
    const std::vector<std::string>& get_const_collected_strings() const { return const_collected_strings_; }
    void clear() { collected_strings_.clear(); const_collected_strings_.clear(); }

private:
    std::vector<std::string> collected_strings_;
    std::vector<std::string> const_collected_strings_;
};

// Test hierarchical visitor
class TestHierarchicalVisitor : public HierarchicalVisitor<TestNode> {
public:
    explicit TestHierarchicalVisitor(TraversalOrder order = TraversalOrder::PRE_ORDER)
        : HierarchicalVisitor<TestNode>(order) {}

    std::string get_name() const override { return "TestHierarchicalVisitor"; }

    const std::vector<std::string>& get_visit_order() const { return visit_order_; }
    void clear() { visit_order_.clear(); }

protected:
    void visit_object(TestNode& object) override {
        visit_order_.push_back(object.get_name());
    }

    std::vector<TestNode*> get_children(TestNode& object) override {
        std::vector<TestNode*> children;
        for (auto& child : object.get_children()) {
            children.push_back(child.get());
        }
        return children;
    }

private:
    std::vector<std::string> visit_order_;
};

// Thread-safe visitor for concurrency testing
class ThreadSafeVisitor : public Visitor<TestNode> {
public:
    void visit(TestNode& node) override {
        std::lock_guard<std::mutex> lock(mutex_);
        visit_count_++;
        visited_thread_ids_.insert(std::this_thread::get_id());
        total_value_ += node.get_value();
    }

    using Visitor<TestNode>::visit; // Bring base class overloads into scope

    std::string get_name() const override { return "ThreadSafeVisitor"; }

    int get_visit_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return visit_count_;
    }

    size_t get_unique_thread_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return visited_thread_ids_.size();
    }

    int get_total_value() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return total_value_;
    }

private:
    mutable std::mutex mutex_;
    int visit_count_ = 0;
    int total_value_ = 0;
    std::unordered_set<std::thread::id> visited_thread_ids_;
};

// ============================================================================
// Visitor Test Fixtures
// ============================================================================

class VisitorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test nodes
        root = std::make_shared<BranchNode>("root", 10);
        child1 = std::make_shared<LeafNode>("child1", 20);
        child2 = std::make_shared<BranchNode>("child2", 30);
        child3 = std::make_shared<SpecialNode>("child3", 40, true);
        grandchild1 = std::make_shared<LeafNode>("grandchild1", 50);
        grandchild2 = std::make_shared<LeafNode>("grandchild2", 60);

        // Build hierarchy
        root->add_child(child1);
        root->add_child(child2);
        root->add_child(child3);
        child2->add_child(grandchild1);
        child2->add_child(grandchild2);

        // Create test collections
        all_nodes = {root, child1, child2, child3, grandchild1, grandchild2};
        leaf_nodes = {child1, grandchild1, grandchild2};
    }

    // Test data
    std::shared_ptr<TestNode> root, child1, child2, child3, grandchild1, grandchild2;
    std::vector<std::shared_ptr<TestNode>> all_nodes;
    std::vector<std::shared_ptr<TestNode>> leaf_nodes;
};

// ============================================================================
// Basic Visitor Interface Tests
// ============================================================================

TEST_F(VisitorTest, BaseVisitorInterface) {
    CountingVisitor visitor("TestCountingVisitor");

    // Test base visitor interface
    EXPECT_EQ(visitor.get_name(), "TestCountingVisitor");
    EXPECT_EQ(visitor.get_description(), "TestCountingVisitor");

    // Test type checking
    EXPECT_TRUE(visitor.can_visit(std::type_index(typeid(TestNode))));
    EXPECT_TRUE(visitor.can_visit(std::type_index(typeid(LeafNode))));
    EXPECT_TRUE(visitor.can_visit(std::type_index(typeid(BranchNode))));
    EXPECT_TRUE(visitor.can_visit(std::type_index(typeid(SpecialNode))));
    EXPECT_FALSE(visitor.can_visit(std::type_index(typeid(int))));
}

TEST_F(VisitorTest, VisitorBasicFunctionality) {
    CountingVisitor visitor;

    // Test visiting individual nodes
    visitor.visit(*root);
    EXPECT_EQ(visitor.get_visit_count(), 1);
    EXPECT_EQ(visitor.get_visited_names().size(), 1);
    EXPECT_EQ(visitor.get_visited_names()[0], "root");
    EXPECT_EQ(root->get_value(), 11); // Should be incremented

    // Test const visiting
    const TestNode& const_node = *child1;
    visitor.visit(const_node);
    EXPECT_EQ(visitor.get_const_visit_count(), 1);
    EXPECT_EQ(visitor.get_const_visited_names()[0], "child1");

    // Test multiple visits
    visitor.visit(*child2);
    visitor.visit(*child3);
    EXPECT_EQ(visitor.get_visit_count(), 3);
    EXPECT_EQ(visitor.get_visited_names().size(), 3);
}

TEST_F(VisitorTest, VisitableAcceptance) {
    CountingVisitor visitor;

    // Test accept method
    root->accept(visitor);
    EXPECT_EQ(visitor.get_visit_count(), 1);
    EXPECT_EQ(visitor.get_visited_names()[0], "root");

    // Test const accept
    const TestNode& const_root = *root;
    const_root.accept(visitor);
    EXPECT_EQ(visitor.get_const_visit_count(), 1);

    // Test base visitor acceptance
    BaseVisitor& base_visitor = visitor;
    child1->accept_base(base_visitor);
    EXPECT_EQ(visitor.get_visit_count(), 2);
}

// ============================================================================
// Hierarchical Visitor Tests
// ============================================================================

TEST_F(VisitorTest, HierarchicalVisitorPreOrder) {
    TestHierarchicalVisitor visitor(HierarchicalVisitor<TestNode>::TraversalOrder::PRE_ORDER);

    visitor.visit(*root);
    auto order = visitor.get_visit_order();

    // Pre-order: root, child1, child2, grandchild1, grandchild2, child3
    std::vector<std::string> expected = {"root", "child1", "child2", "grandchild1", "grandchild2", "child3"};
    EXPECT_EQ(order, expected);
}

TEST_F(VisitorTest, HierarchicalVisitorPostOrder) {
    TestHierarchicalVisitor visitor(HierarchicalVisitor<TestNode>::TraversalOrder::POST_ORDER);

    visitor.visit(*root);
    auto order = visitor.get_visit_order();

    // Post-order: child1, grandchild1, grandchild2, child2, child3, root
    std::vector<std::string> expected = {"child1", "grandchild1", "grandchild2", "child2", "child3", "root"};
    EXPECT_EQ(order, expected);
}

TEST_F(VisitorTest, HierarchicalVisitorInOrder) {
    TestHierarchicalVisitor visitor(HierarchicalVisitor<TestNode>::TraversalOrder::IN_ORDER);

    visitor.visit(*root);
    auto order = visitor.get_visit_order();

    // In-order: child1, root, grandchild1, child2, grandchild2, child3
    // (first child, then parent, then remaining children traversed in order)
    std::vector<std::string> expected = {"child1", "root", "grandchild1", "child2", "grandchild2", "child3"};
    EXPECT_EQ(order, expected);
}

TEST_F(VisitorTest, HierarchicalVisitorTraversalOrderChange) {
    TestHierarchicalVisitor visitor;

    // Test initial order
    EXPECT_EQ(visitor.get_traversal_order(), HierarchicalVisitor<TestNode>::TraversalOrder::PRE_ORDER);

    // Change order and test
    visitor.set_traversal_order(HierarchicalVisitor<TestNode>::TraversalOrder::POST_ORDER);
    EXPECT_EQ(visitor.get_traversal_order(), HierarchicalVisitor<TestNode>::TraversalOrder::POST_ORDER);

    visitor.visit(*root);
    auto order = visitor.get_visit_order();
    std::vector<std::string> expected = {"child1", "grandchild1", "grandchild2", "child2", "child3", "root"};
    EXPECT_EQ(order, expected);
}

// ============================================================================
// Composite Visitor Tests
// ============================================================================

TEST_F(VisitorTest, CompositeVisitorBasicFunctionality) {
    auto composite = std::make_unique<CompositeVisitor<TestNode>>();
    auto counting_visitor = std::make_unique<CountingVisitor>("Counter");
    auto modifying_visitor = std::make_unique<ValueModifyingVisitor>(5);

    // Add visitors to composite
    composite->add_visitor(std::move(counting_visitor));
    composite->add_visitor(std::move(modifying_visitor));

    EXPECT_EQ(composite->get_visitor_count(), 2);
    EXPECT_EQ(composite->get_name(), "CompositeVisitor(2 visitors)");

    // Test visiting
    int original_value = root->get_value();
    composite->visit(*root);

    // Should be modified by both visitors
    EXPECT_EQ(root->get_value(), original_value + 1 + 5); // CountingVisitor +1, ValueModifyingVisitor +5
}

TEST_F(VisitorTest, CompositeVisitorWithReferences) {
    CompositeVisitor<TestNode> composite;
    CountingVisitor counting_visitor("Counter");
    ValueModifyingVisitor modifying_visitor(10);

    // Add visitors by reference
    composite.add_visitor(counting_visitor);
    composite.add_visitor(modifying_visitor);

    EXPECT_EQ(composite.get_visitor_count(), 2);

    // Test visiting
    composite.visit(*child1);

    EXPECT_EQ(counting_visitor.get_visit_count(), 1);
    EXPECT_EQ(modifying_visitor.get_total_modifications(), 1);
    EXPECT_EQ(child1->get_value(), 20 + 1 + 10); // Original + counting + modifying
}

TEST_F(VisitorTest, CompositeVisitorConstVisiting) {
    CompositeVisitor<TestNode> composite;
    CountingVisitor counting_visitor;
    composite.add_visitor(counting_visitor);

    const TestNode& const_node = *child2;
    composite.visit(const_node);

    EXPECT_EQ(counting_visitor.get_const_visit_count(), 1);
}

TEST_F(VisitorTest, CompositeVisitorClear) {
    auto composite = std::make_unique<CompositeVisitor<TestNode>>();
    composite->add_visitor(std::make_unique<CountingVisitor>());
    composite->add_visitor(std::make_unique<ValueModifyingVisitor>(1));

    EXPECT_EQ(composite->get_visitor_count(), 2);

    composite->clear();
    EXPECT_EQ(composite->get_visitor_count(), 0);
}

// ============================================================================
// Conditional Visitor Tests
// ============================================================================

TEST_F(VisitorTest, ConditionalVisitorPredicate) {
    auto base_visitor = std::make_unique<CountingVisitor>("Base");
    auto* base_ptr = base_visitor.get();

    // Create conditional visitor that only visits nodes with value > 25
    auto conditional = std::make_unique<ConditionalVisitor<TestNode>>(
        std::move(base_visitor),
        [](const TestNode& node) { return node.get_value() > 25; }
    );

    EXPECT_EQ(conditional->get_name(), "ConditionalVisitor(Base)");

    // Visit all nodes
    for (auto& node : all_nodes) {
        conditional->visit(*node);
    }

    // Should only visit nodes with value > 25: child2(30), child3(40), grandchild1(50), grandchild2(60)
    EXPECT_EQ(base_ptr->get_visit_count(), 4);

    auto visited_names = base_ptr->get_visited_names();
    EXPECT_THAT(visited_names, UnorderedElementsAre("child2", "child3", "grandchild1", "grandchild2"));
}

TEST_F(VisitorTest, ConditionalVisitorTypeBasedPredicate) {
    auto base_visitor = std::make_unique<StringCollectingVisitor>();
    auto* base_ptr = base_visitor.get();

    // Create conditional visitor that only visits SpecialNode objects
    auto conditional = std::make_unique<ConditionalVisitor<TestNode>>(
        std::move(base_visitor),
        [](const TestNode& node) { return dynamic_cast<const SpecialNode*>(&node) != nullptr; }
    );

    // Visit all nodes
    for (auto& node : all_nodes) {
        conditional->visit(*node);
    }

    // Should only visit child3 (SpecialNode)
    EXPECT_EQ(base_ptr->get_collected_strings().size(), 1);
    EXPECT_EQ(base_ptr->get_collected_strings()[0], "child3:40");
}

// ============================================================================
// Function Visitor Tests
// ============================================================================

TEST_F(VisitorTest, FunctionVisitorLambda) {
    std::vector<std::string> visited_names;

    auto func_visitor = std::make_unique<FunctionVisitor<TestNode>>(
        [&visited_names](TestNode& node) {
            visited_names.push_back(node.get_name());
            node.set_value(node.get_value() * 2);
        },
        "MultiplyingVisitor"
    );

    EXPECT_EQ(func_visitor->get_name(), "MultiplyingVisitor");

    int original_value = root->get_value();
    func_visitor->visit(*root);

    EXPECT_EQ(visited_names.size(), 1);
    EXPECT_EQ(visited_names[0], "root");
    EXPECT_EQ(root->get_value(), original_value * 2);
}

TEST_F(VisitorTest, FunctionVisitorWithConstFunction) {
    std::vector<std::string> mutable_visits;
    std::vector<std::string> const_visits;

    auto func_visitor = std::make_unique<FunctionVisitor<TestNode>>(
        [&mutable_visits](TestNode& node) { mutable_visits.push_back(node.get_name()); },
        [&const_visits](const TestNode& node) { const_visits.push_back(node.get_name()); },
        "DualFunctionVisitor"
    );

    func_visitor->visit(*child1);
    const TestNode& const_child = *child2;
    func_visitor->visit(const_child);

    EXPECT_EQ(mutable_visits.size(), 1);
    EXPECT_EQ(mutable_visits[0], "child1");
    EXPECT_EQ(const_visits.size(), 1);
    EXPECT_EQ(const_visits[0], "child2");
}

TEST_F(VisitorTest, FunctionVisitorConstFallback) {
    std::vector<std::string> visits;

    // Function visitor with only mutable function
    auto func_visitor = std::make_unique<FunctionVisitor<TestNode>>(
        [&visits](TestNode& node) { visits.push_back(node.get_name()); },
        "FallbackVisitor"
    );

    // Visit const object - should fallback to mutable function
    const TestNode& const_child = *child3;
    func_visitor->visit(const_child);

    EXPECT_EQ(visits.size(), 1);
    EXPECT_EQ(visits[0], "child3");
}

// ============================================================================
// Visitor Registry Tests
// ============================================================================

TEST_F(VisitorTest, VisitorRegistryBasicFunctionality) {
    VisitorRegistry<TestNode> registry;

    // Register visitor types
    registry.register_visitor_type<CountingVisitor>("counting");
    registry.register_visitor("modifying", []() -> std::unique_ptr<Visitor<TestNode>> {
        return std::make_unique<ValueModifyingVisitor>(3);
    });

    // Test registration queries
    EXPECT_TRUE(registry.is_registered("counting"));
    EXPECT_TRUE(registry.is_registered("modifying"));
    EXPECT_FALSE(registry.is_registered("nonexistent"));

    auto names = registry.get_registered_names();
    EXPECT_EQ(names.size(), 2);
    EXPECT_THAT(names, UnorderedElementsAre("counting", "modifying"));
}

TEST_F(VisitorTest, VisitorRegistryCreation) {
    VisitorRegistry<TestNode> registry;
    registry.register_visitor_type<CountingVisitor>("counting");

    // Create visitor
    auto visitor = registry.create_visitor("counting");
    ASSERT_NE(visitor, nullptr);
    EXPECT_EQ(visitor->get_name(), "CountingVisitor");

    // Test non-existent visitor
    auto null_visitor = registry.create_visitor("nonexistent");
    EXPECT_EQ(null_visitor, nullptr);
}

TEST_F(VisitorTest, VisitorRegistryCustomFactory) {
    VisitorRegistry<TestNode> registry;

    // Register custom factory with parameters
    registry.register_visitor("custom_incrementer", []() -> std::unique_ptr<Visitor<TestNode>> {
        return std::make_unique<ValueModifyingVisitor>(42);
    });

    auto visitor = registry.create_visitor("custom_incrementer");
    ASSERT_NE(visitor, nullptr);

    int original_value = root->get_value();
    visitor->visit(*root);
    EXPECT_EQ(root->get_value(), original_value + 42);
}

TEST_F(VisitorTest, VisitorRegistryClear) {
    VisitorRegistry<TestNode> registry;
    registry.register_visitor_type<CountingVisitor>("test");

    EXPECT_TRUE(registry.is_registered("test"));

    registry.clear();
    EXPECT_FALSE(registry.is_registered("test"));
    EXPECT_EQ(registry.get_registered_names().size(), 0);
}

// ============================================================================
// Visitor Coordinator Tests
// ============================================================================

TEST_F(VisitorTest, VisitorCoordinatorApplyToCollection) {
    CountingVisitor visitor;

    // Apply to vector of shared_ptr
    VisitorCoordinator<TestNode>::apply_visitor(visitor, all_nodes);

    EXPECT_EQ(visitor.get_visit_count(), all_nodes.size());

    // Check that all nodes were visited
    auto visited_names = visitor.get_visited_names();
    EXPECT_EQ(visited_names.size(), 6);
}

TEST_F(VisitorTest, VisitorCoordinatorApplyWithPredicate) {
    StringCollectingVisitor visitor;

    // Apply only to nodes with value >= 40
    VisitorCoordinator<TestNode>::apply_visitor_if(visitor, all_nodes,
        [](const TestNode& node) { return node.get_value() >= 40; });

    auto collected = visitor.get_collected_strings();
    EXPECT_EQ(collected.size(), 3); // child3(40), grandchild1(50), grandchild2(60)

    EXPECT_THAT(collected, UnorderedElementsAre("child3:40", "grandchild1:50", "grandchild2:60"));
}

TEST_F(VisitorTest, VisitorCoordinatorApplyMultipleVisitors) {
    std::vector<std::unique_ptr<Visitor<TestNode>>> visitors;
    visitors.push_back(std::make_unique<CountingVisitor>());
    visitors.push_back(std::make_unique<ValueModifyingVisitor>(5));

    // Store original values
    std::vector<int> original_values;
    for (auto& node : all_nodes) {
        original_values.push_back(node->get_value());
    }

    VisitorCoordinator<TestNode>::apply_visitors(visitors, all_nodes);

    // Check that all values were modified by both visitors
    for (size_t i = 0; i < all_nodes.size(); ++i) {
        EXPECT_EQ(all_nodes[i]->get_value(), original_values[i] + 1 + 5);
    }
}

TEST_F(VisitorTest, VisitorCoordinatorRawPointerCollection) {
    std::vector<TestNode*> raw_pointers;
    for (auto& node : all_nodes) {
        raw_pointers.push_back(node.get());
    }

    CountingVisitor visitor;
    VisitorCoordinator<TestNode>::apply_visitor(visitor, raw_pointers);

    EXPECT_EQ(visitor.get_visit_count(), raw_pointers.size());
}

// ============================================================================
// Convenience Factory Function Tests
// ============================================================================

TEST_F(VisitorTest, MakeFunctionVisitor) {
    std::vector<std::string> visits;

    auto visitor = make_function_visitor<TestNode>(
        [&visits](TestNode& node) { visits.push_back(node.get_name()); },
        "TestLambda"
    );

    EXPECT_EQ(visitor->get_name(), "TestLambda");

    visitor->visit(*root);
    visitor->visit(*child1);

    EXPECT_EQ(visits.size(), 2);
    EXPECT_THAT(visits, ElementsAre("root", "child1"));
}

TEST_F(VisitorTest, MakeConditionalVisitor) {
    auto base = std::make_unique<CountingVisitor>();
    auto* base_ptr = base.get();

    auto conditional = make_conditional_visitor<TestNode>(
        std::move(base),
        [](const TestNode& node) { return node.get_name().length() > 5; }
    );

    // Visit all nodes - only those with name length > 5 should be visited
    for (auto& node : all_nodes) {
        conditional->visit(*node);
    }

    EXPECT_EQ(base_ptr->get_visit_count(), 5); // child1, child2, child3, grandchild1, grandchild2 have length > 5
}

TEST_F(VisitorTest, MakeCompositeVisitor) {
    auto composite = make_composite_visitor<TestNode>();

    composite->add_visitor(std::make_unique<CountingVisitor>());
    composite->add_visitor(std::make_unique<ValueModifyingVisitor>(1));

    EXPECT_EQ(composite->get_visitor_count(), 2);

    int original_value = root->get_value();
    composite->visit(*root);
    EXPECT_EQ(root->get_value(), original_value + 1 + 1); // Both visitors modify value
}

// ============================================================================
// Thread Safety and Concurrency Tests
// ============================================================================

TEST_F(VisitorTest, ThreadSafeVisitorConcurrency) {
    ThreadSafeVisitor visitor;
    const int num_threads = 10;
    const int visits_per_thread = 100;

    std::vector<std::thread> threads;
    std::vector<std::shared_ptr<TestNode>> test_nodes;

    // Create test nodes for concurrent access
    for (int i = 0; i < visits_per_thread; ++i) {
        test_nodes.push_back(std::make_shared<TestNode>("node" + std::to_string(i), i));
    }

    // Launch threads
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&visitor, &test_nodes, visits_per_thread]() {
            for (int i = 0; i < visits_per_thread; ++i) {
                visitor.visit(*test_nodes[i]);
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify results
    EXPECT_EQ(visitor.get_visit_count(), num_threads * visits_per_thread);
    EXPECT_EQ(visitor.get_unique_thread_count(), num_threads);

    // Verify total value calculation
    int expected_total = 0;
    for (int i = 0; i < visits_per_thread; ++i) {
        expected_total += i * num_threads; // Each node visited by all threads
    }
    EXPECT_EQ(visitor.get_total_value(), expected_total);
}

TEST_F(VisitorTest, CompositeVisitorThreadSafety) {
    // Test that composite visitor properly handles concurrent access
    CompositeVisitor<TestNode> composite;
    composite.add_visitor(std::make_unique<ThreadSafeVisitor>());
    composite.add_visitor(std::make_unique<ThreadSafeVisitor>());

    const int num_threads = 5;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&composite, this]() {
            for (auto& node : all_nodes) {
                composite.visit(*node);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Should complete without crashes or deadlocks
    SUCCEED();
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST_F(VisitorTest, VisitorPerformance) {
    const int num_nodes = 10000;
    const int num_iterations = 1000;

    // Create large collection of nodes
    std::vector<std::shared_ptr<TestNode>> large_collection;
    for (int i = 0; i < num_nodes; ++i) {
        large_collection.push_back(std::make_shared<TestNode>("node" + std::to_string(i), i));
    }

    CountingVisitor visitor;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; ++i) {
        for (auto& node : large_collection) {
            visitor.visit(*node);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Should complete in reasonable time (less than 5 seconds for this scale)
    EXPECT_LT(duration.count(), 5000000);
    EXPECT_EQ(visitor.get_visit_count(), num_nodes * num_iterations);
}

TEST_F(VisitorTest, HierarchicalVisitorPerformance) {
    // Create deep hierarchy
    auto deep_root = std::make_shared<BranchNode>("deep_root", 0);
    auto current = deep_root;

    const int depth = 1000;
    for (int i = 0; i < depth; ++i) {
        auto child = std::make_shared<BranchNode>("child" + std::to_string(i), i);
        current->add_child(child);
        current = child;
    }

    TestHierarchicalVisitor visitor;

    auto start = std::chrono::high_resolution_clock::now();
    visitor.visit(*deep_root);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Should handle deep hierarchies efficiently
    EXPECT_LT(duration.count(), 100);
    EXPECT_EQ(visitor.get_visit_order().size(), depth + 1);
}

// ============================================================================
// Error Handling and Edge Cases
// ============================================================================

TEST_F(VisitorTest, NullPointerHandling) {
    CountingVisitor visitor;
    std::vector<std::shared_ptr<TestNode>> nodes_with_nulls = {root, nullptr, child1, nullptr, child2};

    // Should handle null pointers gracefully
    VisitorCoordinator<TestNode>::apply_visitor(visitor, nodes_with_nulls);

    // Should only visit non-null nodes
    EXPECT_EQ(visitor.get_visit_count(), 3);
}

TEST_F(VisitorTest, EmptyCollectionHandling) {
    CountingVisitor visitor;
    std::vector<std::shared_ptr<TestNode>> empty_collection;

    VisitorCoordinator<TestNode>::apply_visitor(visitor, empty_collection);
    EXPECT_EQ(visitor.get_visit_count(), 0);
}

TEST_F(VisitorTest, InvalidTypeVisiting) {
    MockVisitor mock_visitor;
    EXPECT_CALL(mock_visitor, can_visit(_)).WillRepeatedly(Return(false));

    // Test with invalid type
    EXPECT_FALSE(mock_visitor.can_visit(std::type_index(typeid(std::string))));
}

TEST_F(VisitorTest, HierarchicalVisitorEmptyHierarchy) {
    auto isolated_node = std::make_shared<LeafNode>("isolated", 42);
    TestHierarchicalVisitor visitor;

    visitor.visit(*isolated_node);

    // Should visit just the single node
    EXPECT_EQ(visitor.get_visit_order().size(), 1);
    EXPECT_EQ(visitor.get_visit_order()[0], "isolated");
}

TEST_F(VisitorTest, CompositeVisitorEmptyComposite) {
    CompositeVisitor<TestNode> empty_composite;

    EXPECT_EQ(empty_composite.get_visitor_count(), 0);

    // Should handle visiting with no contained visitors
    empty_composite.visit(*root);
    SUCCEED(); // Should not crash
}

// ============================================================================
// Memory Management Tests
// ============================================================================

TEST_F(VisitorTest, VisitorLifecycleManagement) {
    std::unique_ptr<CompositeVisitor<TestNode>> composite;

    {
        // Create composite and add visitors in inner scope
        composite = std::make_unique<CompositeVisitor<TestNode>>();
        composite->add_visitor(std::make_unique<CountingVisitor>());
        composite->add_visitor(std::make_unique<ValueModifyingVisitor>(1));

        EXPECT_EQ(composite->get_visitor_count(), 2);
    }

    // Visitors should still be alive after scope exit
    composite->visit(*root);
    EXPECT_EQ(composite->get_visitor_count(), 2);

    // Clear should properly destroy visitors
    composite->clear();
    EXPECT_EQ(composite->get_visitor_count(), 0);
}

TEST_F(VisitorTest, VisitorRegistryLifecycle) {
    auto registry = std::make_unique<VisitorRegistry<TestNode>>();
    registry->register_visitor_type<CountingVisitor>("test");

    auto visitor = registry->create_visitor("test");
    ASSERT_NE(visitor, nullptr);

    // Visitor should remain valid after registry destruction
    registry.reset();
    visitor->visit(*root);
    SUCCEED();
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(VisitorTest, ComplexVisitorWorkflow) {
    // Complex workflow combining multiple visitor types
    VisitorRegistry<TestNode> registry;
    registry.register_visitor_type<CountingVisitor>("counter");
    registry.register_visitor("incrementer", []() {
        return std::make_unique<ValueModifyingVisitor>(10);
    });

    // Create composite visitor with registry-created visitors
    auto composite = make_composite_visitor<TestNode>();
    composite->add_visitor(registry.create_visitor("counter"));
    composite->add_visitor(registry.create_visitor("incrementer"));

    // Wrap in conditional visitor
    auto conditional = make_conditional_visitor<TestNode>(
        std::move(composite),
        [](const TestNode& node) { return node.get_value() >= 30; }
    );

    // Apply to hierarchy using coordinator
    std::vector<TestNode*> raw_nodes;
    for (auto& node : all_nodes) {
        raw_nodes.push_back(node.get());
    }

    VisitorCoordinator<TestNode>::apply_visitor(*conditional, raw_nodes);

    // Verify complex workflow results
    // Only nodes with value >= 30 should be modified: child2(30), child3(40), grandchild1(50), grandchild2(60)
    EXPECT_EQ(root->get_value(), 10);      // Not modified (< 30)
    EXPECT_EQ(child1->get_value(), 20);    // Not modified (< 30)
    EXPECT_EQ(child2->get_value(), 30 + 1 + 10);    // Modified
    EXPECT_EQ(child3->get_value(), 40 + 1 + 10);    // Modified
    EXPECT_EQ(grandchild1->get_value(), 50 + 1 + 10); // Modified
    EXPECT_EQ(grandchild2->get_value(), 60 + 1 + 10); // Modified
}

TEST_F(VisitorTest, HierarchicalWithCompositeIntegration) {
    // Test hierarchical visitor containing composite visitor
    class CompositeHierarchicalVisitor : public HierarchicalVisitor<TestNode> {
    public:
        CompositeHierarchicalVisitor() {
            composite_.add_visitor(std::make_unique<CountingVisitor>());
            composite_.add_visitor(std::make_unique<ValueModifyingVisitor>(2));
        }

        std::string get_name() const override { return "CompositeHierarchicalVisitor"; }

    protected:
        void visit_object(TestNode& object) override {
            composite_.visit(object);
        }

        std::vector<TestNode*> get_children(TestNode& object) override {
            std::vector<TestNode*> children;
            for (auto& child : object.get_children()) {
                children.push_back(child.get());
            }
            return children;
        }

    private:
        CompositeVisitor<TestNode> composite_;
    };

    CompositeHierarchicalVisitor hierarchical_composite;

    // Store original values
    std::vector<int> original_values;
    for (auto& node : all_nodes) {
        original_values.push_back(node->get_value());
    }

    hierarchical_composite.visit(*root);

    // All nodes in hierarchy should be visited and modified
    for (size_t i = 0; i < all_nodes.size(); ++i) {
        EXPECT_EQ(all_nodes[i]->get_value(), original_values[i] + 1 + 2);
    }
}

// ============================================================================
// Mock Visitor Integration Tests
// ============================================================================

TEST_F(VisitorTest, MockVisitorIntegration) {
    MockVisitor mock_visitor;

    // Set up expectations
    EXPECT_CALL(mock_visitor, get_name())
        .WillRepeatedly(Return("MockTestVisitor"));

    EXPECT_CALL(mock_visitor, visit(A<TestNode&>()))
        .Times(2);

    // Test visiting
    mock_visitor.visit(*root);
    mock_visitor.visit(*child1);

    EXPECT_EQ(mock_visitor.get_name(), "MockTestVisitor");
}

TEST_F(VisitorTest, MockVisitorInComposite) {
    auto mock_visitor = std::make_unique<MockVisitor>();
    auto* mock_ptr = mock_visitor.get();

    EXPECT_CALL(*mock_ptr, visit(A<TestNode&>()))
        .Times(1);

    EXPECT_CALL(*mock_ptr, get_name())
        .WillRepeatedly(Return("MockInComposite"));

    CompositeVisitor<TestNode> composite;
    composite.add_visitor(std::move(mock_visitor));

    composite.visit(*root);
}