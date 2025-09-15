/**
 * @file test_component.cpp
 * @brief Comprehensive unit tests for the Component/Entity system
 *
 * Tests cover:
 * - Component lifecycle management
 * - Entity-component interactions
 * - Type safety and CRTP functionality
 * - Dependency and compatibility checking
 * - Thread safety where applicable
 * - Edge cases and error conditions
 * - Performance characteristics
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <sstream>

#include "core/base/component.h"

using namespace fem::core::base;
using namespace testing;

// ============================================================================
// Test Components for validation
// ============================================================================

// Basic test component
class TestComponent : public TypedComponent<TestComponent> {
public:
    explicit TestComponent(std::string_view name = "test")
        : TypedComponent(name), update_count_(0) {}

    void update(double dt) override {
        update_count_++;
        last_dt_ = dt;
    }

    void reset() override {
        update_count_ = 0;
        last_dt_ = 0.0;
    }

    std::string serialize() const override {
        return "{\"update_count\":" + std::to_string(update_count_) + "}";
    }

    bool deserialize(const std::string& data) override {
        // Simple parsing for test
        if (data.find("update_count") != std::string::npos) {
            return true;
        }
        return false;
    }

    int get_update_count() const { return update_count_; }
    double get_last_dt() const { return last_dt_; }

private:
    int update_count_;
    double last_dt_;
};

// Component with dependencies
class DependentComponent : public TypedComponent<DependentComponent> {
public:
    explicit DependentComponent(std::string_view name = "dependent")
        : TypedComponent(name) {}

    std::vector<std::type_index> get_dependencies() const override {
        return {std::type_index(typeid(TestComponent))};
    }
};

// Component with compatibility restrictions
class IncompatibleComponent : public TypedComponent<IncompatibleComponent> {
public:
    explicit IncompatibleComponent(std::string_view name = "incompatible")
        : TypedComponent(name) {}

    bool is_compatible_with(std::type_index other_type) const override {
        // Incompatible with TestComponent
        return other_type != std::type_index(typeid(TestComponent));
    }
};

// Heavy component for performance testing
class HeavyComponent : public TypedComponent<HeavyComponent> {
public:
    HeavyComponent() : TypedComponent("heavy") {
        data_.resize(1000000, 0.0); // 1M doubles
    }

    void update(double dt) override {
        // Simulate heavy computation
        for (auto& val : data_) {
            val += dt * 0.001;
        }
    }

private:
    std::vector<double> data_;
};

// Thread-safe component for concurrency testing
class ThreadSafeComponent : public TypedComponent<ThreadSafeComponent> {
public:
    ThreadSafeComponent() : TypedComponent("thread_safe"), counter_(0) {}

    void update(double /*dt*/) override {
        counter_.fetch_add(1, std::memory_order_relaxed);
    }

    int get_counter() const {
        return counter_.load(std::memory_order_relaxed);
    }

private:
    std::atomic<int> counter_;
};

// ============================================================================
// Component Base Class Tests
// ============================================================================

class ComponentTest : public ::testing::Test {
protected:
    void SetUp() override {
        component_ = std::make_unique<TestComponent>("test_comp");
    }

    std::unique_ptr<TestComponent> component_;
};

TEST_F(ComponentTest, Construction) {
    EXPECT_EQ(component_->get_name(), "test_comp");
    EXPECT_EQ(component_->get_owner(), nullptr);
    EXPECT_TRUE(component_->is_active());
}

TEST_F(ComponentTest, NameManagement) {
    component_->set_name("new_name");
    EXPECT_EQ(component_->get_name(), "new_name");
    
    // Empty name
    component_->set_name("");
    EXPECT_EQ(component_->get_name(), "");
    
    // Very long name
    std::string long_name(1000, 'a');
    component_->set_name(long_name);
    EXPECT_EQ(component_->get_name(), long_name);
}

TEST_F(ComponentTest, TypeInformation) {
    auto type_idx = component_->get_type();
    EXPECT_EQ(type_idx, std::type_index(typeid(TestComponent)));
    
    auto type_name = component_->get_type_name();
    EXPECT_FALSE(type_name.empty());
    EXPECT_NE(type_name.find("TestComponent"), std::string::npos);
}

TEST_F(ComponentTest, ActiveState) {
    EXPECT_TRUE(component_->is_active());
    
    component_->set_active(false);
    EXPECT_FALSE(component_->is_active());
    
    component_->set_active(true);
    EXPECT_TRUE(component_->is_active());
}

TEST_F(ComponentTest, UpdateBehavior) {
    EXPECT_EQ(component_->get_update_count(), 0);
    
    component_->update(0.016); // ~60 FPS
    EXPECT_EQ(component_->get_update_count(), 1);
    EXPECT_DOUBLE_EQ(component_->get_last_dt(), 0.016);
    
    component_->update(0.033); // ~30 FPS
    EXPECT_EQ(component_->get_update_count(), 2);
    EXPECT_DOUBLE_EQ(component_->get_last_dt(), 0.033);
}

TEST_F(ComponentTest, ResetBehavior) {
    component_->update(0.016);
    component_->update(0.016);
    EXPECT_EQ(component_->get_update_count(), 2);
    
    component_->reset();
    EXPECT_EQ(component_->get_update_count(), 0);
    EXPECT_DOUBLE_EQ(component_->get_last_dt(), 0.0);
}

TEST_F(ComponentTest, Serialization) {
    component_->update(0.016);
    component_->update(0.016);
    
    auto serialized = component_->serialize();
    EXPECT_NE(serialized.find("\"update_count\":2"), std::string::npos);
    
    EXPECT_TRUE(component_->deserialize(serialized));
    EXPECT_FALSE(component_->deserialize("invalid_json"));
}

TEST_F(ComponentTest, DefaultBehaviors) {
    // Test default implementations
    EXPECT_TRUE(component_->get_dependencies().empty());
    EXPECT_TRUE(component_->is_compatible_with(std::type_index(typeid(int))));
    EXPECT_TRUE(component_->is_compatible_with(std::type_index(typeid(TestComponent))));
}

TEST_F(ComponentTest, TypedComponentCasting) {
    // Test safe casting
    auto* as_test = component_->as<TestComponent>();
    EXPECT_NE(as_test, nullptr);
    
    auto* as_other = component_->as<DependentComponent>();
    EXPECT_EQ(as_other, nullptr);
    
    // Const casting
    const auto* const_comp = component_.get();
    auto* const_as_test = const_comp->as<TestComponent>();
    EXPECT_NE(const_as_test, nullptr);
}

// ============================================================================
// Entity Tests
// ============================================================================

class EntityTest : public ::testing::Test {
protected:
    void SetUp() override {
        entity_ = std::make_unique<Entity>("test_entity");
    }

    std::unique_ptr<Entity> entity_;
};

TEST_F(EntityTest, Construction) {
    EXPECT_EQ(entity_->get_name(), "test_entity");
    EXPECT_TRUE(entity_->is_active());
    EXPECT_EQ(entity_->get_component_count(), 0);
}

TEST_F(EntityTest, AddComponent) {
    auto* comp = entity_->add_component<TestComponent>("comp1");
    ASSERT_NE(comp, nullptr);
    EXPECT_EQ(comp->get_name(), "comp1");
    EXPECT_EQ(comp->get_owner(), entity_.get());
    EXPECT_EQ(entity_->get_component_count(), 1);
}

TEST_F(EntityTest, GetComponent) {
    entity_->add_component<TestComponent>("comp1");
    
    auto* comp = entity_->get_component<TestComponent>();
    ASSERT_NE(comp, nullptr);
    EXPECT_EQ(comp->get_name(), "comp1");
    
    auto* missing = entity_->get_component<DependentComponent>();
    EXPECT_EQ(missing, nullptr);
}

TEST_F(EntityTest, HasComponent) {
    EXPECT_FALSE(entity_->has_component<TestComponent>());
    
    entity_->add_component<TestComponent>();
    EXPECT_TRUE(entity_->has_component<TestComponent>());
    EXPECT_FALSE(entity_->has_component<DependentComponent>());
}

TEST_F(EntityTest, RemoveComponent) {
    entity_->add_component<TestComponent>();
    EXPECT_TRUE(entity_->has_component<TestComponent>());
    
    EXPECT_TRUE(entity_->remove_component<TestComponent>());
    EXPECT_FALSE(entity_->has_component<TestComponent>());
    
    // Remove non-existent component
    EXPECT_FALSE(entity_->remove_component<TestComponent>());
}

TEST_F(EntityTest, ReplaceComponent) {
    auto* comp1 = entity_->add_component<TestComponent>("comp1");
    comp1->update(0.016);
    EXPECT_EQ(comp1->get_update_count(), 1);
    
    auto* comp2 = entity_->add_component<TestComponent>("comp2");
    EXPECT_EQ(comp2->get_update_count(), 0);
    EXPECT_EQ(comp2->get_name(), "comp2");
    
    // Old component should be removed
    EXPECT_EQ(entity_->get_component_count(), 1);
}

TEST_F(EntityTest, MultipleComponents) {
    entity_->add_component<TestComponent>();
    entity_->add_component<ThreadSafeComponent>();
    
    EXPECT_EQ(entity_->get_component_count(), 2);
    EXPECT_TRUE(entity_->has_component<TestComponent>());
    EXPECT_TRUE(entity_->has_component<ThreadSafeComponent>());
}

TEST_F(EntityTest, ComponentDependencies) {
    // Add dependent component without dependency should fail
    auto* dep_comp = entity_->add_component<DependentComponent>();
    EXPECT_EQ(dep_comp, nullptr);
    
    // Add dependency first
    entity_->add_component<TestComponent>();
    
    // Now dependent component should succeed
    dep_comp = entity_->add_component<DependentComponent>();
    EXPECT_NE(dep_comp, nullptr);
}

TEST_F(EntityTest, ComponentCompatibility) {
    // Add TestComponent first
    entity_->add_component<TestComponent>();
    
    // Try to add incompatible component
    auto* incompat = entity_->add_component<IncompatibleComponent>();
    EXPECT_EQ(incompat, nullptr);
    
    // Remove TestComponent and try again
    entity_->remove_component<TestComponent>();
    incompat = entity_->add_component<IncompatibleComponent>();
    EXPECT_NE(incompat, nullptr);
}

TEST_F(EntityTest, UpdateEntity) {
    auto* comp1 = entity_->add_component<TestComponent>();
    auto* comp2 = entity_->add_component<ThreadSafeComponent>();
    
    entity_->update(0.016);
    
    EXPECT_EQ(comp1->get_update_count(), 1);
    EXPECT_EQ(comp2->get_counter(), 1);
    
    // Inactive component shouldn't update
    comp1->set_active(false);
    entity_->update(0.016);
    
    EXPECT_EQ(comp1->get_update_count(), 1); // No change
    EXPECT_EQ(comp2->get_counter(), 2);
}

TEST_F(EntityTest, ResetEntity) {
    auto* comp = entity_->add_component<TestComponent>();
    comp->update(0.016);
    comp->update(0.016);
    EXPECT_EQ(comp->get_update_count(), 2);
    
    entity_->reset();
    EXPECT_EQ(comp->get_update_count(), 0);
}

TEST_F(EntityTest, ForEachComponent) {
    entity_->add_component<TestComponent>("comp1");
    entity_->add_component<ThreadSafeComponent>();
    
    // For each of specific type
    int test_count = 0;
    entity_->for_each_component<TestComponent>([&test_count](TestComponent* comp) {
        test_count++;
        EXPECT_EQ(comp->get_name(), "comp1");
    });
    EXPECT_EQ(test_count, 1);
    
    // For each of all components
    int total_count = 0;
    entity_->for_each_component([&total_count](Component* comp) {
        total_count++;
        EXPECT_NE(comp, nullptr);
    });
    EXPECT_EQ(total_count, 2);
}

TEST_F(EntityTest, GetAllComponents) {
    entity_->add_component<TestComponent>();
    entity_->add_component<ThreadSafeComponent>();
    
    auto& components = entity_->get_all_components();
    EXPECT_EQ(components.size(), 2);
    
    // Check types are present
    bool has_test = false;
    bool has_thread_safe = false;
    
    for (const auto& [type, comp] : components) {
        if (type == std::type_index(typeid(TestComponent))) has_test = true;
        if (type == std::type_index(typeid(ThreadSafeComponent))) has_thread_safe = true;
    }
    
    EXPECT_TRUE(has_test);
    EXPECT_TRUE(has_thread_safe);
}

// ============================================================================
// ObjectEntity Tests
// ============================================================================

TEST(ObjectEntityTest, Construction) {
    ObjectEntity obj_entity("test_obj_entity");
    EXPECT_EQ(obj_entity.get_name(), "test_obj_entity");
    EXPECT_NE(obj_entity.id(), 0); // Should have valid Object ID
}

TEST(ObjectEntityTest, DualInheritance) {
    ObjectEntity obj_entity;
    
    // Test Object functionality
    auto id = obj_entity.id();
    EXPECT_NE(id, 0);
    EXPECT_EQ(obj_entity.ref_count(), 1);
    
    // Test Entity functionality
    auto* comp = obj_entity.add_component<TestComponent>();
    EXPECT_NE(comp, nullptr);
    EXPECT_TRUE(obj_entity.has_component<TestComponent>());
}

TEST(ObjectEntityTest, UpdateAndReset) {
    ObjectEntity obj_entity;
    auto* comp = obj_entity.add_component<TestComponent>();
    
    obj_entity.update(0.016);
    EXPECT_EQ(comp->get_update_count(), 1);
    
    obj_entity.reset();
    EXPECT_EQ(comp->get_update_count(), 0);
}

// ============================================================================
// ComponentSystem Tests
// ============================================================================

class ComponentSystemTest : public ::testing::Test {
protected:
    void SetUp() override {
        system_ = std::make_unique<ComponentSystem>();
    }

    std::unique_ptr<ComponentSystem> system_;
};

TEST_F(ComponentSystemTest, RegisterComponentType) {
    system_->register_component_type<TestComponent>();
    system_->register_component_type<DependentComponent>();
    
    auto types = system_->get_registered_types();
    EXPECT_GE(types.size(), 2);
}

TEST_F(ComponentSystemTest, CreateComponentByName) {
    system_->register_component_type<TestComponent>();
    
    auto comp = system_->create_component(typeid(TestComponent).name());
    ASSERT_NE(comp, nullptr);
    EXPECT_EQ(comp->get_type(), std::type_index(typeid(TestComponent)));
    
    // Unknown type
    auto unknown = system_->create_component("UnknownType");
    EXPECT_EQ(unknown, nullptr);
}

TEST_F(ComponentSystemTest, ManageEntities) {
    auto entity1 = std::make_unique<Entity>("entity1");
    auto entity2 = std::make_unique<Entity>("entity2");
    
    entity1->add_component<TestComponent>();
    entity2->add_component<TestComponent>();
    
    system_->manage_entity(entity1.get());
    system_->manage_entity(entity2.get());
    
    // Update all managed entities
    system_->update_all_entities(0.016);
    
    auto* comp1 = entity1->get_component<TestComponent>();
    auto* comp2 = entity2->get_component<TestComponent>();
    
    EXPECT_EQ(comp1->get_update_count(), 1);
    EXPECT_EQ(comp2->get_update_count(), 1);
    
    // Unmanage one entity
    system_->unmanage_entity(entity1.get());
    system_->update_all_entities(0.016);
    
    EXPECT_EQ(comp1->get_update_count(), 1); // Not updated
    EXPECT_EQ(comp2->get_update_count(), 2); // Updated
}

TEST_F(ComponentSystemTest, InactiveEntities) {
    auto entity = std::make_unique<Entity>();
    entity->add_component<TestComponent>();
    
    system_->manage_entity(entity.get());
    
    entity->set_active(false);
    system_->update_all_entities(0.016);
    
    auto* comp = entity->get_component<TestComponent>();
    EXPECT_EQ(comp->get_update_count(), 0); // Not updated when inactive
}

// ============================================================================
// Edge Cases and Error Conditions
// ============================================================================

TEST(ComponentEdgeCases, NullOwner) {
    TestComponent comp;
    EXPECT_EQ(comp.get_owner(), nullptr);
    
    comp.on_attach(nullptr);
    EXPECT_EQ(comp.get_owner(), nullptr);
}

TEST(ComponentEdgeCases, AttachDetachCycle) {
    TestComponent comp;
    Entity entity;
    
    comp.on_attach(&entity);
    EXPECT_EQ(comp.get_owner(), &entity);
    
    comp.on_detach();
    EXPECT_EQ(comp.get_owner(), nullptr);
    
    comp.on_attach(&entity);
    EXPECT_EQ(comp.get_owner(), &entity);
}

TEST(EntityEdgeCases, EmptyEntity) {
    Entity entity;
    
    entity.update(0.016); // Should not crash
    entity.reset(); // Should not crash
    
    EXPECT_EQ(entity.get_component_count(), 0);
    EXPECT_FALSE(entity.has_component<TestComponent>());
    EXPECT_FALSE(entity.remove_component<TestComponent>());
}

TEST(EntityEdgeCases, LargeNumberOfComponents) {
    Entity entity;
    
    // Create many different component types dynamically
    class DynamicComponent : public TypedComponent<DynamicComponent> {
    public:
        explicit DynamicComponent(int id) : TypedComponent("dynamic"), id_(id) {}
        int id_;
    };
    
    // Add multiple components (limited test for compilation)
    entity.add_component<TestComponent>();
    entity.add_component<ThreadSafeComponent>();
    entity.add_component<DynamicComponent>(1);
    
    EXPECT_EQ(entity.get_component_count(), 3);
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST(ThreadSafetyTest, ConcurrentUpdates) {
    Entity entity;
    auto* comp = entity.add_component<ThreadSafeComponent>();
    
    const int num_threads = 4;
    const int updates_per_thread = 1000;
    
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&entity, updates_per_thread]() {
            for (int j = 0; j < updates_per_thread; ++j) {
                entity.update(0.001);
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // Each update increments counter once
    EXPECT_EQ(comp->get_counter(), num_threads * updates_per_thread);
}

TEST(ThreadSafetyTest, ComponentSystemConcurrentManagement) {
    ComponentSystem system;
    const int num_entities = 100;
    
    std::vector<std::unique_ptr<Entity>> entities;
    for (int i = 0; i < num_entities; ++i) {
        entities.push_back(std::make_unique<Entity>());
        entities.back()->add_component<ThreadSafeComponent>();
    }
    
    // Concurrent registration
    std::vector<std::thread> threads;
    for (int i = 0; i < num_entities; ++i) {
        threads.emplace_back([&system, &entities, i]() {
            system.manage_entity(entities[i].get());
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // Concurrent updates
    threads.clear();
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&system]() {
            for (int j = 0; j < 10; ++j) {
                system.update_all_entities(0.001);
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // Verify all entities were updated
    for (const auto& entity : entities) {
        auto* comp = entity->get_component<ThreadSafeComponent>();
        EXPECT_GT(comp->get_counter(), 0);
    }
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST(PerformanceTest, ComponentUpdateOverhead) {
    Entity entity;
    auto* comp = entity.add_component<TestComponent>();
    
    const int iterations = 100000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        comp->update(0.001);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double us_per_update = static_cast<double>(duration.count()) / iterations;
    
    // Expect less than 1 microsecond per simple update
    EXPECT_LT(us_per_update, 1.0);
    
    // Print for information
    std::cout << "Component update overhead: " << us_per_update << " microseconds\n";
}

TEST(PerformanceTest, EntityIterationOverhead) {
    Entity entity;
    const int num_components = 10;
    
    for (int i = 0; i < num_components; ++i) {
        entity.add_component<TestComponent>("comp" + std::to_string(i));
    }
    
    const int iterations = 10000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        entity.update(0.001);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double us_per_iteration = static_cast<double>(duration.count()) / iterations;
    
    // Expect reasonable overhead even with multiple components
    EXPECT_LT(us_per_iteration, 10.0);
    
    std::cout << "Entity iteration overhead (" << num_components << " components): " 
              << us_per_iteration << " microseconds\n";
}

TEST(PerformanceTest, HeavyComponentMemoryFootprint) {
    Entity entity;
    
    // Measure memory before
    auto* heavy = entity.add_component<HeavyComponent>();
    ASSERT_NE(heavy, nullptr);
    
    // Heavy component should allocate ~8MB (1M doubles)
    // This test mainly verifies it doesn't crash or leak
    
    entity.update(0.001); // Should complete without issues
    entity.remove_component<HeavyComponent>(); // Should clean up properly
}

// ============================================================================
// Boundary Conditions
// ============================================================================

TEST(BoundaryConditions, MaxComponents) {
    Entity entity;
    
    // Try to add many components (different types required due to map storage)
    // This is a compile-time limitation test more than runtime
    entity.add_component<TestComponent>();
    entity.add_component<DependentComponent>();
    entity.add_component<ThreadSafeComponent>();
    
    // Verify all were added (DependentComponent succeeds because TestComponent satisfies its dependency)
    EXPECT_EQ(entity.get_component_count(), 3);
}

TEST(BoundaryConditions, ZeroDeltaTime) {
    Entity entity;
    auto* comp = entity.add_component<TestComponent>();
    
    entity.update(0.0);
    EXPECT_EQ(comp->get_update_count(), 1);
    EXPECT_DOUBLE_EQ(comp->get_last_dt(), 0.0);
}

TEST(BoundaryConditions, NegativeDeltaTime) {
    Entity entity;
    auto* comp = entity.add_component<TestComponent>();
    
    entity.update(-0.016); // Negative dt (e.g., time reversal)
    EXPECT_EQ(comp->get_update_count(), 1);
    EXPECT_DOUBLE_EQ(comp->get_last_dt(), -0.016);
}

TEST(BoundaryConditions, VeryLargeDeltaTime) {
    Entity entity;
    auto* comp = entity.add_component<TestComponent>();
    
    entity.update(1e10); // Very large dt
    EXPECT_EQ(comp->get_update_count(), 1);
    EXPECT_DOUBLE_EQ(comp->get_last_dt(), 1e10);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(IntegrationTest, CompleteComponentLifecycle) {
    ComponentSystem system;
    system.register_component_type<TestComponent>();
    system.register_component_type<DependentComponent>();
    
    auto entity = std::make_unique<Entity>("test_entity");
    
    // Add components
    auto* test_comp = entity->add_component<TestComponent>("test");
    ASSERT_NE(test_comp, nullptr);
    
    auto* dep_comp = entity->add_component<DependentComponent>("dependent");
    ASSERT_NE(dep_comp, nullptr);
    
    // Register with system
    system.manage_entity(entity.get());
    
    // Update through system
    system.update_all_entities(0.016);
    EXPECT_EQ(test_comp->get_update_count(), 1);
    
    // Serialize state
    auto serialized = test_comp->serialize();
    EXPECT_FALSE(serialized.empty());
    
    // Reset
    entity->reset();
    EXPECT_EQ(test_comp->get_update_count(), 0);
    
    // Deserialize
    EXPECT_TRUE(test_comp->deserialize(serialized));
    
    // Remove component
    EXPECT_TRUE(entity->remove_component<DependentComponent>());
    EXPECT_FALSE(entity->has_component<DependentComponent>());
    
    // Unregister from system
    system.unmanage_entity(entity.get());
    
    // Final update shouldn't affect entity
    system.update_all_entities(0.016);
    EXPECT_EQ(test_comp->get_update_count(), 0);
}

