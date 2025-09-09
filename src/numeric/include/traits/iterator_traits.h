#pragma once

#ifndef NUMERIC_ITERATOR_TRAITS_H
#define NUMERIC_ITERATOR_TRAITS_H

#include <iterator>
#include <type_traits>
#include <memory>
#include <concepts>
#include <cstring>  // Added for std::memcpy

#include "../base/iterator_base.h"
#include "../base/numeric_base.h"

#include "type_traits.h"
#include "numeric_traits.h"

namespace fem::numeric::traits {

    /**
     * @brief Iterator categories extended for numeric operations
     */
    enum class NumericIteratorCategory {
        Input,              // Single-pass read
        Output,             // Single-pass write
        Forward,            // Multi-pass forward
        Bidirectional,      // Forward and backward
        RandomAccess,       // Arbitrary jumps
        Contiguous,         // Elements in contiguous memory
        Strided,            // Fixed stride between elements
        Indexed,            // Index-based access
        Checked,            // Runtime bounds checking
        Parallel,           // Thread-safe for parallel access
        Unknown
    };

    /**
     * @brief Memory access patterns for iterators
     */
    enum class IteratorAccessPattern {
        Sequential,         // Access elements in order
        Strided,           // Access with fixed stride
        Random,            // Random access pattern
        Blocked,           // Block-wise access
        Diagonal,          // Diagonal matrix access
        Triangular,        // Upper/lower triangular access
        Sparse,            // Sparse element access
        Unknown
    };

    /**
     * @brief Basic iterator traits using standard library
     */
    template<typename Iterator>
    struct basic_iterator_traits {
        using traits = std::iterator_traits<Iterator>;
        using value_type = typename traits::value_type;
        using difference_type = typename traits::difference_type;
        using pointer = typename traits::pointer;
        using reference = typename traits::reference;
        using iterator_category = typename traits::iterator_category;
    };

    /**
     * @brief Check if type is an iterator
     */
    template<typename T>
    struct is_iterator {
        static constexpr bool value = requires(T it) {
            typename std::iterator_traits<T>::value_type;
            typename std::iterator_traits<T>::iterator_category;
            { ++it } -> std::same_as<T&>;
            { *it };
        };
    };

    template<typename T>
    inline constexpr bool is_iterator_v = is_iterator<T>::value;

    // Forward declarations for template variables
    template<typename Iterator>
    inline constexpr bool is_checked_iterator_v = false;

    template<typename Iterator>
    inline constexpr bool is_multidim_iterator_v = false;

    // Specializations for our custom iterator types
    template<typename BaseIt>
    inline constexpr bool is_checked_iterator_v<fem::numeric::CheckedIterator<BaseIt>> = true;

    template<typename T, size_t Rank>
    inline constexpr bool is_multidim_iterator_v<fem::numeric::MultiDimIterator<T, Rank>> = true;

    /**
     * @brief Detect iterator category with numeric extensions
     * Builds upon numeric_iterator_traits from iterator_base.h
     */
    template<typename Iterator>
    struct numeric_iterator_category {
        using std_category = typename std::iterator_traits<Iterator>::iterator_category;
        using base_traits = numeric_iterator_traits<Iterator>;

        static constexpr NumericIteratorCategory value = [] {
            // Check for pointers first (always contiguous)
            if constexpr (std::is_pointer_v<Iterator>) {
                return NumericIteratorCategory::Contiguous;
            }
            // Check for our custom iterator types from iterator_base.h
            else if constexpr (base_traits::is_contiguous) {
                return NumericIteratorCategory::Contiguous;
            }
            else if constexpr (base_traits::is_strided) {
                return NumericIteratorCategory::Strided;
            }
            // Check for MultiDimIterator
            else if constexpr (is_multidim_iterator_v<Iterator>) {
                return NumericIteratorCategory::Indexed;
            }
            // Check for CheckedIterator
            else if constexpr (is_checked_iterator_v<Iterator>) {
                return NumericIteratorCategory::Checked;
            }
            // Check for contiguous iterator (C++20)
            else if constexpr (requires { typename std::contiguous_iterator_tag; } &&
                               std::is_base_of_v<std::contiguous_iterator_tag, std_category>) {
                return NumericIteratorCategory::Contiguous;
            }
            // Standard categories
            else if constexpr (std::is_base_of_v<std::random_access_iterator_tag, std_category>) {
                return NumericIteratorCategory::RandomAccess;
            }
            else if constexpr (std::is_base_of_v<std::bidirectional_iterator_tag, std_category>) {
                return NumericIteratorCategory::Bidirectional;
            }
            else if constexpr (std::is_base_of_v<std::forward_iterator_tag, std_category>) {
                return NumericIteratorCategory::Forward;
            }
            else if constexpr (std::is_base_of_v<std::input_iterator_tag, std_category>) {
                return NumericIteratorCategory::Input;
            }
            else if constexpr (std::is_base_of_v<std::output_iterator_tag, std_category>) {
                return NumericIteratorCategory::Output;
            }
            else {
                return NumericIteratorCategory::Unknown;
            }
        }();
    };

    template<typename Iterator>
    inline constexpr NumericIteratorCategory numeric_iterator_category_v =
    numeric_iterator_category<Iterator>::value;

    /**
     * @brief Check if iterator is random access or better
     */
    template<typename Iterator>
    struct is_random_access_iterator {
        static constexpr bool value =
                numeric_iterator_category_v<Iterator> >= NumericIteratorCategory::RandomAccess;
    };

    template<typename Iterator>
    inline constexpr bool is_random_access_iterator_v = is_random_access_iterator<Iterator>::value;



    /**
     * @brief Check if iterator points to contiguous memory
     */
    template<typename Iterator>
    struct is_contiguous_iterator {
    private:
        // Helper to check if iterator behaves like it points to contiguous memory
        // This is a heuristic for pre-C++20 code
        template<typename It>
        static constexpr bool has_contiguous_behavior() {
            if constexpr (std::is_pointer_v<It>) {
                return true;
            }
            // Random access iterators from vector/array are typically contiguous
            // This is a reasonable assumption for standard library implementations
            else if constexpr (std::is_same_v<typename std::iterator_traits<It>::iterator_category,
                                            std::random_access_iterator_tag> ||
                             std::is_base_of_v<std::random_access_iterator_tag,
                                             typename std::iterator_traits<It>::iterator_category>) {
                // Additional check: if we can get a pointer and the value type is trivial,
                // it's very likely contiguous (covers vector, array, string iterators)
                if constexpr (requires(It it) {
                    { &(*it) } -> std::convertible_to<typename std::iterator_traits<It>::pointer>;
                }) {
                    // For trivially copyable types with random access, assume contiguous
                    // This is true for std::vector and std::array in practice
                    if constexpr (std::is_trivially_copyable_v<typename std::iterator_traits<It>::value_type>) {
                        return true;
                    }
                }
            }
            return false;
        }

    public:
        static constexpr bool value = [] {
            // Check if it's a pointer (always contiguous)
            if constexpr (std::is_pointer_v<Iterator>) {
                return true;
            }
            // Check numeric_iterator_traits from base first
            if constexpr (numeric_iterator_traits<Iterator>::is_contiguous) {
                return true;
            }
            // Check our custom numeric category directly
            if constexpr (
                numeric_iterator_category_v<Iterator> ==
                NumericIteratorCategory::Contiguous) {
                return true;
            }
            // For C++20, use the contiguous_iterator concept if available
            #ifdef __cpp_lib_concepts
            #if __cpp_lib_concepts >= 202002L
            if constexpr (requires { std::contiguous_iterator<Iterator>; }) {
                if constexpr (std::contiguous_iterator<Iterator>) {
                    return true;
                }
            }
            #endif
            #endif
            // Check for C++20 contiguous_iterator_tag if available
            if constexpr (requires { typename std::contiguous_iterator_tag; }) {
                using cat = typename std::iterator_traits<Iterator>::iterator_category;
                if constexpr (std::is_base_of_v<std::contiguous_iterator_tag, cat>) {
                    return true;
                }
            }
            // Heuristic for pre-C++20: assume standard random access iterators are contiguous
            if constexpr (has_contiguous_behavior<Iterator>()) {
                return true;
            }

            return false;
        }();
    };

    template<typename Iterator>
    inline constexpr bool is_contiguous_iterator_v = is_contiguous_iterator<Iterator>::value;

    /**
     * @brief Check if iterator is strided
     */
    template<typename Iterator>
    struct is_strided_iterator {
        static constexpr bool value =
                numeric_iterator_traits<Iterator>::is_strided ||
                numeric_iterator_category_v<Iterator> == NumericIteratorCategory::Strided ||
                requires(Iterator it) {
            { it.stride() } -> std::convertible_to<std::ptrdiff_t>;
        };
    };

    template<typename Iterator>
    inline constexpr bool is_strided_iterator_v = is_strided_iterator<Iterator>::value;

    /**
     * @brief Get stride of iterator (if applicable)
     */
    template<typename Iterator>
    struct iterator_stride {
        static constexpr std::ptrdiff_t value = [] {
            if constexpr (is_contiguous_iterator_v<Iterator>) {
                return 1;  // Contiguous iterators always have stride 1
            } else if constexpr (is_strided_iterator_v<Iterator>) {
                return 1;  // Would need instance to get actual stride
            } else {
                return 1;  // Default stride is 1 for most iterators
            }
        }();

        static std::ptrdiff_t get(const Iterator& it) {
            if constexpr (requires { it.stride(); }) {
                return it.stride();
            } else {
                return 1;
            }
        }
    };

    /**
     * @brief Check if iterator supports parallel operations
     */
    template<typename Iterator>
    struct is_parallel_safe_iterator {
        static constexpr bool value =
                is_random_access_iterator_v<Iterator> &&
                !is_checked_iterator_v<Iterator>;  // Checked iterators may have state
    };

    template<typename Iterator>
    inline constexpr bool is_parallel_safe_iterator_v = is_parallel_safe_iterator<Iterator>::value;

    /**
     * @brief Iterator properties aggregator
     * Enhanced to work with base iterator types
     */
    template<typename Iterator>
    struct iterator_properties {
        using basic_traits = basic_iterator_traits<Iterator>;
        using base_numeric_traits = numeric_iterator_traits<Iterator>;
        using value_type = typename basic_traits::value_type;
        using difference_type = typename basic_traits::difference_type;
        using pointer = typename basic_traits::pointer;
        using reference = typename basic_traits::reference;

        static constexpr NumericIteratorCategory category = numeric_iterator_category_v<Iterator>;
        static constexpr bool is_random_access = is_random_access_iterator_v<Iterator>;
        static constexpr bool is_contiguous = is_contiguous_iterator_v<Iterator>;
        static constexpr bool is_strided = is_strided_iterator_v<Iterator>;
        static constexpr bool is_checked = is_checked_iterator_v<Iterator>;
        static constexpr bool is_multidim = is_multidim_iterator_v<Iterator>;
        static constexpr bool is_parallel_safe = is_parallel_safe_iterator_v<Iterator>;

        // From base traits
        static constexpr bool is_numeric = base_numeric_traits::is_numeric;
        static constexpr bool is_ieee_compliant = base_numeric_traits::is_ieee_compliant;

        // Performance characteristics
        static constexpr bool supports_fast_advance = is_random_access;
        static constexpr bool supports_fast_distance = is_random_access;
        static constexpr bool supports_simd = is_contiguous &&
                                              std::is_arithmetic_v<value_type> &&
                                              is_ieee_compliant;

        // Memory access pattern - using immediate invocation
        static constexpr IteratorAccessPattern access_pattern = ([]() constexpr {
            if constexpr (is_contiguous) {
                return IteratorAccessPattern::Sequential;
            } else if constexpr (is_strided) {
                return IteratorAccessPattern::Strided;
            } else if constexpr (is_multidim) {
                return IteratorAccessPattern::Blocked;
            } else if constexpr (is_random_access) {
                return IteratorAccessPattern::Random;
            } else {
                return IteratorAccessPattern::Unknown;
            }
        })();
    };

    /**
     * @brief Distance computation optimization
     */
    template<typename Iterator>
    struct distance_traits {
        using difference_type = typename std::iterator_traits<Iterator>::difference_type;

        static difference_type compute(Iterator first, Iterator last) {
            if constexpr (is_random_access_iterator_v<Iterator>) {
                // O(1) for random access
                return last - first;
            } else {
                // O(n) for other iterators
                return std::distance(first, last);
            }
        }

        static constexpr bool is_constant_time = is_random_access_iterator_v<Iterator>;
    };

    /**
     * @brief Advance operation optimization
     */
    template<typename Iterator>
    struct advance_traits {
        using difference_type = typename std::iterator_traits<Iterator>::difference_type;

        static void advance(Iterator& it, difference_type n) {
            if constexpr (is_random_access_iterator_v<Iterator>) {
                // O(1) for random access
                it += n;
            } else {
                // O(n) for other iterators
                std::advance(it, n);
            }
        }

        static constexpr bool is_constant_time = is_random_access_iterator_v<Iterator>;
    };

    /**
     * @brief Check if iterator range is valid
     */
    template<typename Iterator>
    struct range_validation {
        static bool is_valid_range(Iterator first, Iterator last) {
            if constexpr (is_random_access_iterator_v<Iterator>) {
                return first <= last;
            } else {
                // Can't efficiently validate for non-random-access
                return true;  // Assume valid
            }
        }

        static bool is_empty(Iterator first, Iterator last) {
            return first == last;
        }

        static size_t size(Iterator first, Iterator last) {
            if constexpr (is_random_access_iterator_v<Iterator>) {
                return first <= last ? static_cast<size_t>(last - first) : 0;
            } else {
                size_t count = 0;
                for (auto it = first; it != last; ++it) {
                    ++count;
                }
                return count;
            }
        }
    };

    /**
     * @brief Iterator pair traits
     */
    template<typename Iterator1, typename Iterator2>
    struct iterator_pair_traits {
        static constexpr bool are_compatible =
                std::is_same_v<typename std::iterator_traits<Iterator1>::value_type,
                        typename std::iterator_traits<Iterator2>::value_type>;

        static constexpr bool both_random_access =
                is_random_access_iterator_v<Iterator1> &&
                is_random_access_iterator_v<Iterator2>;

        static constexpr bool both_contiguous =
                is_contiguous_iterator_v<Iterator1> &&
                is_contiguous_iterator_v<Iterator2>;

        static constexpr bool can_parallel_process =
                both_random_access &&
                is_parallel_safe_iterator_v<Iterator1> &&
                is_parallel_safe_iterator_v<Iterator2>;
    };

    /**
     * @brief SIMD iteration support
     */
    template<typename Iterator>
    struct simd_iteration_traits {
        using value_type = typename std::iterator_traits<Iterator>::value_type;

        static constexpr bool can_vectorize =
                is_contiguous_iterator_v<Iterator> &&
                std::is_arithmetic_v<value_type> &&
                !std::is_same_v<value_type, bool>;

        static constexpr size_t vector_width = [] {
            if constexpr (!can_vectorize) {
                return 1;
            } else if constexpr (sizeof(value_type) == 4) {
                return 8;  // 256-bit AVX for 32-bit types
            } else if constexpr (sizeof(value_type) == 8) {
                return 4;  // 256-bit AVX for 64-bit types
            } else {
                return 16 / sizeof(value_type);  // 128-bit SSE fallback
            }
        }();

        // Check if range is aligned for SIMD
        static bool is_aligned(const Iterator& it) {
            if constexpr (std::is_pointer_v<Iterator>) {
                return reinterpret_cast<uintptr_t>(&(*it)) % (vector_width * sizeof(value_type)) == 0;
            } else {
                return false;  // Can't check alignment for non-pointers
            }
        }
    };

    /**
    * @brief Multi-dimensional iteration traits
    * Works with MultiDimIterator from base
    */
    template<typename Iterator>
    struct multidim_iterator_traits {
        static constexpr bool is_multidim = is_multidim_iterator_v<Iterator>;

        static constexpr size_t dimensionality = ([]() constexpr {
            if constexpr (is_multidim) {
                // MultiDimIterator has index_type = std::array<size_t, Rank>
                if constexpr (requires {
                    typename Iterator::index_type;
                    std::tuple_size<typename Iterator::index_type>::value;
                }) {
                    return std::tuple_size<typename Iterator::index_type>::value;
                }
            }
            return size_t(1);  // Default to 1D
        })();

        // Get current indices from MultiDimIterator
        template<typename It>
        static auto get_indices(const It& it) {
            if constexpr (requires { it.indices(); }) {
                return it.indices();
            } else {
                return std::array<size_t, 1>{0};
            }
        }

        static constexpr bool supports_broadcasting = false;  // Base MultiDimIterator doesn't support broadcasting
    };

    /**
     * @brief Iteration optimization hints
     */
    template<typename Iterator>
    struct iteration_optimization {
        using props = iterator_properties<Iterator>;

        enum Strategy {
            Sequential,         // Simple sequential loop
            Vectorized,         // SIMD vectorization
            Parallel,           // Multithreaded
            ParallelVectorized, // Both parallel and SIMD
            Blocked,            // Cache-blocked iteration
            Prefetched          // With prefetching
        };

        static constexpr Strategy recommended_strategy(size_t range_size) {
            if (range_size < 100) {
                return Sequential;  // Too small for optimization
            }

            if constexpr (props::supports_simd) {
                if constexpr (props::is_parallel_safe) {
                    if (range_size > 10000) {
                        return ParallelVectorized;
                    } else {
                        return Vectorized;
                    }
                } else {
                    return Vectorized;
                }
            } else if constexpr (props::is_parallel_safe) {
                if (range_size > 1000) {
                    return Parallel;
                }
            }

            return Sequential;
        }
        //TODO: should the cache_line and l1_cache sizes be fetched from the machine hardware info instead of
        //      being hard coded?
        static constexpr size_t optimal_block_size = [] {
            constexpr size_t cache_line = 64;
            constexpr size_t l1_cache = 32768;

            if constexpr (props::is_contiguous) {
                return l1_cache / sizeof(typename props::value_type);
            } else {
                return cache_line / sizeof(typename props::value_type);
            }
        }();
    };

    /**
     * @brief Iterator algorithm selector
     */
    template<typename Iterator>
    struct algorithm_selector {
        using props = iterator_properties<Iterator>;

        // Select copy algorithm
        template<typename OutputIterator>
        static void copy(Iterator first, Iterator last, OutputIterator out) {
            if constexpr (is_contiguous_iterator_v<Iterator> &&
                          is_contiguous_iterator_v<OutputIterator>) {
                // Use memcpy for POD types
                if constexpr (std::is_trivially_copyable_v<typename props::value_type>) {
                    size_t n = distance_traits<Iterator>::compute(first, last);
                    std::memcpy(&(*out), &(*first), n * sizeof(typename props::value_type));
                    return;
                }
            }

            // Fall back to std::copy
            std::copy(first, last, out);
        }

        // Select find algorithm
        template<typename T>
        static Iterator find(Iterator first, Iterator last, const T& value) {
            if constexpr (simd_iteration_traits<Iterator>::can_vectorize) {
                // Could use SIMD find for arithmetic types
                return simd_find(first, last, value);
            } else {
                return std::find(first, last, value);
            }
        }

    private:
        template<typename T>
        static Iterator simd_find(Iterator first, Iterator last, const T& value) {
            // Placeholder for SIMD implementation
            return std::find(first, last, value);
        }
    };

    /**
     * @brief Checked iterator wrapper traits
     * Works with CheckedIterator from base
     */
    template<typename Iterator>
    struct checked_iterator_traits {
        static constexpr bool is_checked = is_checked_iterator_v<Iterator>;

        static constexpr bool has_debug_info = is_checked;

        // Extract the underlying iterator type from CheckedIterator
        template<typename It>
        using base_iterator_t = std::conditional_t<
            is_checked_iterator_v<It>,
            typename It::iterator,  // CheckedIterator has 'iterator' typedef
            It
        >;

        // Cost of bounds checking (CheckedIterator adds runtime checks)
        static constexpr double overhead_factor = is_checked ? 1.2 : 1.0;

        // Check if IEEE compliance checking is enabled
        static constexpr bool checks_ieee_compliance =
                is_checked && numeric_iterator_traits<Iterator>::is_ieee_compliant;
    };

    /**
     * @brief Zip iterator detection
     */
    template<typename Iterator>
    struct is_zip_iterator {
        static constexpr bool value = requires(Iterator it) {
            typename Iterator::iterator_tuple;  // Tuple of iterators
            { it.template get<0>() };  // Access individual iterators - fixed template keyword
        };
    };

    template<typename Iterator>
    inline constexpr bool is_zip_iterator_v = is_zip_iterator<Iterator>::value;

    /**
     * @brief Transform iterator detection
     */
    template<typename Iterator>
    struct is_transform_iterator {
        static constexpr bool value = requires(Iterator it) {
            typename Iterator::function_type;  // Has transformation function
            typename Iterator::base_iterator;  // Has underlying iterator
        };
    };

    template<typename Iterator>
    inline constexpr bool is_transform_iterator_v = is_transform_iterator<Iterator>::value;

} // namespace fem::numeric::traits

#endif //NUMERIC_ITERATOR_TRAITS_H