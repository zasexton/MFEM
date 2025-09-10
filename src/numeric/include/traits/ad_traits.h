#pragma once

#ifndef NUMERIC_AD_TRAITS_H
#define NUMERIC_AD_TRAITS_H

#include "../base/dual_base.h"
#include "../base/traits_base.h"
#include "type_traits.h"
#include "numeric_traits.h"

namespace fem::numeric::traits {

        using fem::numeric::is_dual_number;
        using fem::numeric::is_dual_number_v;
        using fem::numeric::scalar_type;
        using fem::numeric::scalar_type_t;

        // Check if type supports dual arithmetic
        template<typename T>
        struct supports_dual_arithmetic {
            static constexpr bool value = std::is_arithmetic_v<T> || is_dual_number_v<T>;
        };

        template<typename T>
        inline constexpr bool supports_dual_arithmetic_v = supports_dual_arithmetic<T>::value;

} // namespace fem::numeric::traits


#endif //NUMERIC_AD_TRAITS_H
