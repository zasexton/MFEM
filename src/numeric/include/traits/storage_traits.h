#pragma once

#ifndef NUMERIC_STORAGE_TRAITS_H
#define NUMERIC_STORAGE_TRAITS_H

#include <type_traits>
#include <memory>
#include <cstddef>
#include <limits>

#include "../base/storage_base.h"
#include "../base/allocator_base.h"
#include "../base/numeric_base.h"
#include "../base/traits_base.h"

#include "type_traits.h"
#include "numeric_traits.h"

namespace fem::numeric::traits {

    /**
     * @brief Storage categories for dispatch
     */
    enum class StorageCategory {
        Dense,          // Contiguous memory
        Sparse,         // Sparse formats (CSR, COO, etc.)
        Static,         // Compile-time fixed size
        Dynamic,        // Runtime allocated
        Aligned,        // SIMD-aligned memory
        Strided,        // Non-contiguous with strides
        Pooled,         // Memory pool allocated
        View,           // Non-owning reference
        Hybrid,         // Small buffer optimization
        Mapped,         // Memory-mapped storage
        Distributed,    // Distributed across nodes
        Device,         // GPU/accelerator memory
        Unknown
    };

    /**
     * @brief Memory ownership model
     */
    enum class OwnershipModel {
        Owning,         // Owns and manages memory
        NonOwning,      // References external memory
        SharedOwning,   // Shared ownership (ref counted)
        WeakReference,  // Weak reference (may be invalidated)
        Borrowed,       // Temporary borrowing
        Unknown
    };

    /**
     * @brief Memory access patterns
     */
    enum class AccessPattern {
        Sequential,     // Linear access
        Random,         // Random access
        Strided,        // Fixed stride access
        Blocked,        // Block-wise access
        Streaming,      // One-time sequential
        Unknown
    };

    /**
     * @brief Detect if type is a storage type
     */
    template<typename T>
    struct is_storage {
    private:
        template<typename S>
        static std::true_type test(const StorageBase<S>*);
        static std::false_type test(...);

    public:
        static constexpr bool value = decltype(test(std::declval<T*>()))::value;
    };

    template<typename T>
    inline constexpr bool is_storage_v = is_storage<T>::value;

    /**
     * @brief Detect specific storage types from base
     */
    template<typename T>
    struct is_dynamic_storage : std::is_same<T, DynamicStorage<typename T::value_type>> {};

    template<typename T>
    inline constexpr bool is_dynamic_storage_v = is_dynamic_storage<T>::value;

    template<typename T, size_t N>
    struct is_static_storage : std::is_same<T, StaticStorage<typename T::value_type, N>> {};

    template<typename T>
    struct is_aligned_storage : std::false_type {};

    template<typename V, size_t A>
    struct is_aligned_storage<AlignedStorage<V, A>> : std::true_type {};

    template<typename T>
    inline constexpr bool is_aligned_storage_v = is_aligned_storage<T>::value;

    /**
     * @brief Extract value type from storage
     */
    template<typename Storage, typename = void>
    struct storage_value_type {
        using type = void;
    };

    template<typename Storage>
    struct storage_value_type<Storage, std::void_t<typename Storage::value_type>> {
        using type = typename Storage::value_type;
    };

    template<typename Storage>
    using storage_value_type_t = typename storage_value_type<Storage>::type;

    /**
     * @brief Get storage capacity (static or dynamic)
     */
    template<typename Storage>
    struct storage_capacity {
        static constexpr size_t static_capacity = 0;
        static constexpr bool is_static = false;

        static size_t capacity(const Storage& s) {
            if constexpr (requires { s.capacity(); }) {
                return s.capacity();
            } else if constexpr (requires { s.size(); }) {
                return s.size();
            } else {
                return 0;
            }
        }
    };

    // Specialization for static storage
    template<typename T, size_t N>
    struct storage_capacity<StaticStorage<T, N>> {
        static constexpr size_t static_capacity = N;
        static constexpr bool is_static = true;

        static constexpr size_t capacity(const StaticStorage<T, N>&) {
            return N;
        }
    };

    /**
     * @brief Memory alignment requirements
     */
    template<typename Storage>
    struct alignment_traits {
        using value_type = storage_value_type_t<Storage>;

        static constexpr size_t required_alignment = [] {
            if constexpr (is_aligned_storage_v<Storage>) {
                return Storage::alignment;
            } else if constexpr (requires { Storage::alignment; }) {
                return Storage::alignment;
            } else {
                return alignof(value_type);
            }
        }();

        static constexpr bool is_simd_aligned =
                required_alignment >= 16;  // Minimum for SSE

        static constexpr bool is_cache_aligned =
                required_alignment >= 64;  // Typical cache line

        static constexpr bool is_page_aligned =
                required_alignment >= 4096;  // Typical page size
    };

    /**
     * @brief Growth strategy for dynamic storage
     */
    template<typename Storage>
    struct growth_strategy {
        enum Strategy {
            None,           // Fixed size
            Linear,         // Grow by constant amount
            Geometric,      // Grow by factor (e.g., 1.5x or 2x)
            OnDemand,       // Grow exactly as needed
            Chunked         // Grow in fixed chunks
        };

        static constexpr Strategy strategy = [] {
            if constexpr (storage_capacity<Storage>::is_static) {
                return None;
            } else if constexpr (is_dynamic_storage_v<Storage>) {
                return Geometric;  // std::vector-like growth
            } else {
                return OnDemand;
            }
        }();

        static constexpr double growth_factor =
                strategy == Geometric ? 2.0 : 1.0;

        static size_t next_capacity(size_t current) {
            switch (strategy) {
                case Geometric:
                    return static_cast<size_t>(current * growth_factor);
                case Linear:
                    return current + 1024;  // Example linear growth
                case Chunked:
                    return ((current / 4096) + 1) * 4096;  // 4KB chunks
                default:
                    return current;
            }
        }
    };

    /**
     * @brief Storage properties aggregator
     */
    template<typename Storage>
    struct storage_properties {
        using value_type = storage_value_type_t<Storage>;
        using alignment_info = alignment_traits<Storage>;
        using capacity_info = storage_capacity<Storage>;
        using growth_info = growth_strategy<Storage>;

        // Basic classification
        static constexpr StorageCategory category = [] {
            if constexpr (capacity_info::is_static) {
                return StorageCategory::Static;
            } else if constexpr (is_aligned_storage_v<Storage>) {
                return StorageCategory::Aligned;
            } else if constexpr (is_dynamic_storage_v<Storage>) {
                return StorageCategory::Dynamic;
            } else {
                return StorageCategory::Unknown;
            }
        }();

        // Memory characteristics
        static constexpr bool is_contiguous =
                category == StorageCategory::Dense ||
                category == StorageCategory::Static ||
                category == StorageCategory::Dynamic ||
                category == StorageCategory::Aligned;

        static constexpr bool is_resizable =
                category == StorageCategory::Dynamic ||
                category == StorageCategory::Sparse;

        static constexpr size_t alignment = alignment_info::required_alignment;
        static constexpr bool simd_ready = alignment_info::is_simd_aligned;

        // Ownership
        static constexpr OwnershipModel ownership = [] {
            if constexpr (category == StorageCategory::View) {
                return OwnershipModel::NonOwning;
            } else {
                return OwnershipModel::Owning;
            }
        }();

        // Performance hints
        static constexpr bool zero_copy_move = is_contiguous;
        static constexpr bool supports_parallel_access = is_contiguous;
        static constexpr bool cache_friendly =
                is_contiguous && alignment >= 64;
    };

    /**
     * @brief Check if storage is suitable for a given operation
     */
    template<typename Storage, typename Op>
    struct is_storage_suitable_for_operation {
        using props = storage_properties<Storage>;

        static constexpr bool value = [] {
            // SIMD operations need aligned storage
            if constexpr (requires { Op::requires_simd; }) {
                return props::simd_ready;
            }
            // Parallel operations need thread-safe storage
            else if constexpr (requires { Op::requires_parallel; }) {
                return props::supports_parallel_access;
            }
            // Default: any storage works
            else {
                return true;
            }
        }();
    };

    template<typename Storage, typename Op>
    inline constexpr bool is_storage_suitable_for_operation_v =
            is_storage_suitable_for_operation<Storage, Op>::value;

    /**
     * @brief Memory usage estimation
     */
    template<typename Storage>
    struct memory_footprint {
        using value_type = storage_value_type_t<Storage>;

        // Bytes per element (including padding)
        static constexpr size_t element_size = [] {
            if constexpr (requires { Storage::element_size; }) {
                return Storage::element_size;
            } else {
                return sizeof(value_type);
            }
        }();

        // Overhead bytes (metadata, pointers, etc.)
        static constexpr size_t overhead = [] {
            if constexpr (storage_capacity<Storage>::is_static) {
                return 0;  // No overhead for static storage
            } else if constexpr (is_dynamic_storage_v<Storage>) {
                return sizeof(void*) * 3;  // Typical vector overhead
            } else {
                return sizeof(void*);  // Minimum pointer overhead
            }
        }();

        // Total bytes for n elements
        static size_t bytes_required(size_t n) {
            size_t data_bytes = n * element_size;

            // Add alignment padding
            constexpr size_t align = alignment_traits<Storage>::required_alignment;
            if constexpr (align > alignof(value_type)) {
                data_bytes = ((data_bytes + align - 1) / align) * align;
            }

            return data_bytes + overhead;
        }

        // Memory efficiency (useful data / total allocated)
        static double efficiency(size_t n) {
            if (n == 0) return 0.0;
            size_t useful = n * sizeof(value_type);
            size_t total = bytes_required(n);
            return static_cast<double>(useful) / total;
        }
    };

    /**
     * @brief Storage conversion compatibility
     */
    template<typename From, typename To>
    struct can_convert_storage {
        static constexpr bool value = [] {
            using from_value = storage_value_type_t<From>;
            using to_value = storage_value_type_t<To>;

            // Check value type compatibility
            if constexpr (!std::is_convertible_v<from_value, to_value>) {
                return false;
            }

            // Check size compatibility for static storage
            if constexpr (storage_capacity<To>::is_static) {
                if constexpr (storage_capacity<From>::is_static) {
                    return storage_capacity<From>::static_capacity <=
                           storage_capacity<To>::static_capacity;
                } else {
                    return false;  // Can't convert dynamic to static
                }
            }

            return true;
        }();
    };

    template<typename From, typename To>
    inline constexpr bool can_convert_storage_v = can_convert_storage<From, To>::value;

    /**
     * @brief Optimal storage selector based on requirements
     */
    template<typename T, size_t Size = 0, bool NeedsSIMD = false, bool NeedsGrow = false>
    struct optimal_storage_selector {
        using type = std::conditional_t<
        Size == 0 || NeedsGrow,
        // Dynamic size or needs growth
        std::conditional_t<
        NeedsSIMD, AlignedStorage<T, 32>,  // AVX alignment
        DynamicStorage<T>>,
        // Static size
        std::conditional_t<
                Size * sizeof(T) <= 256,  // Small buffer optimization threshold
                StaticStorage<T, Size>,
        std::conditional_t<
        NeedsSIMD,
        AlignedStorage<T, 32>,
        DynamicStorage<T>>>>;
    };

    template<typename T, size_t Size = 0, bool NeedsSIMD = false, bool NeedsGrow = false>
    using optimal_storage_t = typename optimal_storage_selector<T, Size, NeedsSIMD, NeedsGrow>::type;

    /**
     * @brief Storage iterator traits
     */
    template<typename Storage>
    struct storage_iterator_traits {
        using value_type = storage_value_type_t<Storage>;

        using iterator = std::conditional_t<
                storage_properties<Storage>::is_contiguous,
                value_type*,
                typename Storage::iterator>;

        using const_iterator = std::conditional_t<
                storage_properties<Storage>::is_contiguous,
                const value_type*,
                typename Storage::const_iterator>;

        static constexpr bool has_random_access =
                storage_properties<Storage>::is_contiguous;
    };

    /**
     * @brief Cache behavior hints
     */
    template<typename Storage>
    struct cache_hints {
        static constexpr size_t cache_line_size = 64;

        // Elements per cache line
        static constexpr size_t elements_per_cache_line =
        cache_line_size / sizeof(storage_value_type_t<Storage>);

        // Prefetch distance for streaming
        static constexpr size_t prefetch_distance = [] {
            if constexpr (storage_properties<Storage>::is_contiguous) {
                return 4 * elements_per_cache_line;  // Prefetch 4 cache lines ahead
            } else {
                return 0;  // No prefetch for non-contiguous
            }
        }();

        // Whether to use non-temporal stores (bypass cache)
        static constexpr bool use_streaming_stores =
                storage_properties<Storage>::is_contiguous &&
                sizeof(storage_value_type_t<Storage>) >= 32;  // Large elements
    };

    /**
     * @brief Sparse storage detection
     */
    template<typename Storage>
    struct is_sparse_storage {
        static constexpr bool value =
                requires(Storage s) {
            { s.nnz() } -> std::convertible_to<size_t>;  // Non-zero count
            { s.format() };  // Storage format (CSR, COO, etc.)
        };
    };

    template<typename Storage>
    inline constexpr bool is_sparse_storage_v = is_sparse_storage<Storage>::value;

    /**
     * @brief Memory pool traits
     */
    template<typename Storage>
    struct is_pool_allocated {
        static constexpr bool value =
                requires(Storage s) {
            { s.pool() };  // Has pool accessor
            { s.chunk_size() } -> std::convertible_to<size_t>;
        };
    };

    template<typename Storage>
    inline constexpr bool is_pool_allocated_v = is_pool_allocated<Storage>::value;

    /**
     * @brief Device memory traits (GPU, etc.)
     */
    template<typename Storage>
    struct device_memory_traits {
        static constexpr bool is_device_memory =
                requires { typename Storage::device_type; };

        static constexpr bool is_unified_memory =
                requires { Storage::is_unified; } && Storage::is_unified;

        static constexpr bool requires_explicit_transfer =
                is_device_memory && !is_unified_memory;

        static constexpr Device device_type = [] {
            if constexpr (is_device_memory) {
                if constexpr (requires { Storage::device_type; }) {
                    return Storage::device_type;
                }
            }
            return Device::CPU;
        }();
    };

} // namespace fem::numeric::traits

#endif //NUMERIC_STORAGE_TRAITS_H
