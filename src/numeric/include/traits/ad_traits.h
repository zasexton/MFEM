#pragma once

#ifndef NUMERIC_AD_TRAITS_H
#define NUMERIC_AD_TRAITS_H

#include "../base/dual_base.hpp"
#include "type_traits.hpp"
#include "numeric_traits.hpp"

namespace fem::numeric::traits {

        // Type traits for dual numbers
        template<typename T>
        struct is_dual : std::false_type {};

        template<typename T, std::size_t N>
        struct is_dual<autodiff::DualBase<T, N>> : std::true_type {};

        template<typename T>
        inline constexpr bool is_dual_v = is_dual<T>::value;

        // Extract underlying scalar type
        template<typename T>
        struct scalar_type {
            using type = T;
        };

        template<typename T, std::size_t N>
        struct scalar_type<autodiff::DualBase<T, N>> {
            using type = T;
        };

        template<typename T>
        using scalar_type_t = typename scalar_type<T>::type;

        // Check if type supports dual arithmetic
        template<typename T>
        struct supports_dual_arithmetic {
            static constexpr bool value = std::is_arithmetic_v<T> || is_dual_v<T>;
        };

        template<typename T>
        inline constexpr bool supports_dual_arithmetic_v = supports_dual_arithmetic<T>::value;

} // namespace fem::numeric::traits


#endif //NUMERIC_AD_TRAITS_H