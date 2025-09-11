#pragma once

#ifndef NUMERIC_CONTAINER_UTILS_H
#define NUMERIC_CONTAINER_UTILS_H

#include "container_traits.h"

namespace fem::numeric::traits {

    /**
     * @brief Determine optimal evaluation strategy for an expression or container.
     */
    template<typename Expr>
    struct evaluation_strategy {
        enum Strategy {
            Immediate,   ///< Evaluate immediately
            Lazy,        ///< Keep as expression
            Parallel,    ///< Use parallel evaluation
            Vectorized   ///< Use SIMD instructions
        };

        static constexpr Strategy value = []() constexpr {
            if constexpr (is_expression_base_v<Expr>) {
                // Already an expression, keep lazy
                return Lazy;
            } else if constexpr (extended_container_traits<Expr>::supports_simd) {
                return Vectorized;
            } else if constexpr (requires { Expr::use_parallel; }) {
                return Parallel;
            } else {
                return Immediate;
            }
        }();
    };

    /**
     * @brief Helper for selecting optimal container type based on requirements.
     */
    template<typename T, size_t Size = 0, bool NeedsSIMD = false>
    struct optimal_container_selector {
        using chosen_storage = std::conditional_t<
            Size == 0,
            // Dynamic size storage
            DynamicStorage<T>,
            // Static size
            std::conditional_t<
                NeedsSIMD,
                // Need SIMD alignment
                AlignedStorage<T, 32>,
                // Small size optimization
                std::conditional_t<
                    Size * sizeof(T) <= 256,
                    StaticStorage<T, Size>,
                    DynamicStorage<T>
                >
            >
        >;

        template<typename U, typename Storage>
        struct selected_container : ContainerBase<selected_container<U, Storage>, U, Storage> {};

        using type = selected_container<T, chosen_storage>;
    };

    template<typename T, size_t Size = 0, bool NeedsSIMD = false>
    using optimal_container_t = typename optimal_container_selector<T, Size, NeedsSIMD>::type;

} // namespace fem::numeric::traits

#endif // NUMERIC_CONTAINER_UTILS_H
