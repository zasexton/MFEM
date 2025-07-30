#pragma once

#ifndef CONCEPTS_H
#define CONCEPTS_H

#include <concepts>

#include "numeric/scalar/traits.h"

namespace numeric::scalar {
    //=============================================================================
    // Basic scalar concepts
    // =============================================================================
    /**
     * @brief Any IEEE‑754 floating‑point type (`float`, `double`, `long double`, …).
     */
    template <typename T>
    concept RealLike = std::floating_point<T>;

    /**
     * @brief A `std::complex<FP>` where `FP` is itself a `RealLike`.
     */
    template <typename T>
    concept ComplexLike = is_complex_v<T> && RealLike<real_t<T>>;

    /**
     * @brief Union of `RealLike` and `ComplexLike`.
     */
    template <typename T>
    concept ScalarLike = RealLike<T> || ComplexLike<T>;

    /**
     * @brief Signed integral types (`int`, `long`, …), excludes enums.
     */
    template <typename T>
    concept SignedIntegral = std::signed_integral<T>;

    /**
     * @brief Unsigned integral types (`unsigned`, `std::size_t`, …), excludes enums.
     */
    template <typename T>
    concept UnsignedIntegral = std::unsigned_integral<T>;

    /**
     * @brief All integral types but not including true/false booleans.
     */
     template <typename T>
     concept IntegerLike = std::integral<T> && !std::same_as<std::remove_cv_t<T>, bool>;;

     /**
      * @brief Any IEEE-754 real OR integral scalar
      *        (`float`, `double`, `int`, `std::size_t`, ...).
      *  *Complex numbers are deliberately excluded* – use `ScalarLike`
      *  when you need those as well.
      */
      template <typename T>
      concept ArithmeticLike = RealLike<T> || IntegerLike<T>;

      template <typename T>
      concept NumberLike = ScalarLike<T> || IntegerLike<T>;

} // namespace numeric::scalar

#endif //CONCEPTS_H
