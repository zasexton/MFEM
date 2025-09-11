#include <benchmark/benchmark.h>
#include "core/base/object.h"

#include <vector>
#include <memory>

using namespace fem::core::base;

// ============================================================================
// Object Creation Benchmarks
// ============================================================================

static void BM_ObjectCreation(benchmark::State& state) {
    for (auto _ : state) {
        Object obj("BenchmarkObject");
        benchmark::DoNotOptimize(obj.id());
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ObjectCreation);

static void BM_ObjectCreationWithSourceLocation(benchmark::State& state) {
    for (auto _ : state) {
        Object obj("BenchmarkObject", std::source_location::current());
        benchmark::DoNotOptimize(obj.id());
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ObjectCreationWithSourceLocation);

// ============================================================================
// ID Generation Benchmarks
// ============================================================================

static void BM_IDGenerationSingleThread(benchmark::State& state) {
    std::vector<Object::id_type> ids;
    ids.reserve(state.iterations());
    
    for (auto _ : state) {
        Object obj;
        ids.push_back(obj.id());
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetLabel("IDs generated");
}
BENCHMARK(BM_IDGenerationSingleThread);

static void BM_IDGenerationMultiThread(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<std::unique_ptr<Object>> objects;
        objects.reserve(state.range(0));
        state.ResumeTiming();
        
        for (int i = 0; i < state.range(0); ++i) {
            objects.push_back(std::make_unique<Object>());
        }
    }
    
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_IDGenerationMultiThread)->Range(1, 1000)->ThreadRange(1, 8);

// ============================================================================
// Type Checking Benchmarks
// ============================================================================

class TestObject : public Object {
public:
    TestObject() : Object("TestObject"), value_(42) {}
    int value_;
};

class AnotherObject : public Object {
public:
    AnotherObject() : Object("AnotherObject") {}
};

static void BM_TypeCheckingIsType(benchmark::State& state) {
    TestObject obj;
    Object* base = &obj;
    
    for (auto _ : state) {
        bool result = base->is_type<TestObject>();
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_TypeCheckingIsType);

static void BM_TypeCheckingAs(benchmark::State& state) {
    TestObject obj;
    Object* base = &obj;
    
    for (auto _ : state) {
        auto* derived = base->as<TestObject>();
        benchmark::DoNotOptimize(derived);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_TypeCheckingAs);

static void BM_TypeCheckingAsRef(benchmark::State& state) {
    TestObject obj;
    Object& base = obj;
    
    for (auto _ : state) {
        try {
            auto& derived = base.as_ref<TestObject>();
            benchmark::DoNotOptimize(derived.value_);
        } catch (...) {
            // Should not happen in this benchmark
        }
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_TypeCheckingAsRef);

// ============================================================================
// Reference Counting Benchmarks
// ============================================================================

static void BM_ReferenceCountingAddRelease(benchmark::State& state) {
    Object obj;
    
    for (auto _ : state) {
        obj.add_ref();
        obj.release();
    }
    state.SetItemsProcessed(state.iterations() * 2);  // 2 ops per iteration
}
BENCHMARK(BM_ReferenceCountingAddRelease);

static void BM_ReferenceCountingConcurrent(benchmark::State& state) {
    Object obj;
    
    for (auto _ : state) {
        obj.add_ref();
        benchmark::DoNotOptimize(obj.ref_count());
        obj.release();
    }
    state.SetItemsProcessed(state.iterations() * 2);
}
BENCHMARK(BM_ReferenceCountingConcurrent)->ThreadRange(1, 8);

// ============================================================================
// object_ptr Benchmarks
// ============================================================================

static void BM_ObjectPtrCreation(benchmark::State& state) {
    for (auto _ : state) {
        auto ptr = make_object<TestObject>();
        benchmark::DoNotOptimize(ptr.get());
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ObjectPtrCreation);

static void BM_ObjectPtrCopy(benchmark::State& state) {
    auto original = make_object<TestObject>();
    
    for (auto _ : state) {
        object_ptr<TestObject> copy(original);
        benchmark::DoNotOptimize(copy.get());
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ObjectPtrCopy);

static void BM_ObjectPtrMove(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        auto ptr1 = make_object<TestObject>();
        state.ResumeTiming();
        
        object_ptr<TestObject> ptr2(std::move(ptr1));
        benchmark::DoNotOptimize(ptr2.get());
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ObjectPtrMove);

static void BM_ObjectPtrDereference(benchmark::State& state) {
    auto ptr = make_object<TestObject>();
    
    for (auto _ : state) {
        int value = ptr->value_;
        benchmark::DoNotOptimize(value);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ObjectPtrDereference);

// ============================================================================
// Comparison and Hashing Benchmarks
// ============================================================================

static void BM_ObjectComparison(benchmark::State& state) {
    Object obj1;
    Object obj2;
    
    for (auto _ : state) {
        bool equal = (obj1 == obj2);
        benchmark::DoNotOptimize(equal);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ObjectComparison);

static void BM_ObjectHashing(benchmark::State& state) {
    Object obj;
    std::hash<Object> hasher;
    
    for (auto _ : state) {
        auto hash = hasher(obj);
        benchmark::DoNotOptimize(hash);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ObjectHashing);

static void BM_ObjectPtrHashing(benchmark::State& state) {
    auto ptr = make_object<TestObject>();
    std::hash<object_ptr<TestObject>> hasher;
    
    for (auto _ : state) {
        auto hash = hasher(ptr);
        benchmark::DoNotOptimize(hash);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ObjectPtrHashing);

// ============================================================================
// Complex Scenarios
// ============================================================================

static void BM_PolymorphicVectorIteration(benchmark::State& state) {
    // Setup polymorphic vector
    std::vector<object_ptr<Object>> objects;
    objects.reserve(state.range(0));
    
    for (int i = 0; i < state.range(0); ++i) {
        if (i % 2 == 0) {
            objects.push_back(make_object<TestObject>());
        } else {
            objects.push_back(make_object<AnotherObject>());
        }
    }
    
    for (auto _ : state) {
        int count = 0;
        for (const auto& obj : objects) {
            if (obj->is_type<TestObject>()) {
                ++count;
            }
        }
        benchmark::DoNotOptimize(count);
    }
    
    state.SetItemsProcessed(state.iterations() * state.range(0));
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_PolymorphicVectorIteration)
    ->Range(10, 10000)
    ->Complexity();

static void BM_MassObjectCreationDestruction(benchmark::State& state) {
    for (auto _ : state) {
        std::vector<object_ptr<Object>> objects;
        objects.reserve(state.range(0));
        
        for (int i = 0; i < state.range(0); ++i) {
            objects.push_back(make_object<Object>());
        }
        
        // Destruction happens automatically
    }
    
    state.SetItemsProcessed(state.iterations() * state.range(0));
    state.SetLabel(std::to_string(state.range(0)) + " objects");
}
BENCHMARK(BM_MassObjectCreationDestruction)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000);

// ============================================================================
// Custom Counters and Statistics
// ============================================================================

static void BM_ObjectMemoryFootprint(benchmark::State& state) {
    std::vector<std::unique_ptr<Object>> objects;
    
    for (auto _ : state) {
        state.PauseTiming();
        objects.clear();
        objects.reserve(state.range(0));
        state.ResumeTiming();
        
        for (int i = 0; i < state.range(0); ++i) {
            objects.push_back(std::make_unique<Object>("TestObject"));
        }
    }
    
    // Report memory usage
    state.counters["Objects"] = state.range(0);
    state.counters["BytesPerObject"] = sizeof(Object);
    state.counters["TotalBytes"] = state.range(0) * sizeof(Object);
    state.counters["ObjectsPerSec"] = benchmark::Counter(
        state.range(0), benchmark::Counter::kIsRate);
}
BENCHMARK(BM_ObjectMemoryFootprint)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000);

// ============================================================================
// Main function (provided by benchmark::benchmark_main)
// ============================================================================
// BENCHMARK_MAIN();  // Not needed if linking with benchmark::benchmark_main