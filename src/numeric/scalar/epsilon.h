#pragma once

#ifndef EPSILON_H
#define EPSILON_H

/**
 * @file epsilon.hpp
 * @brief Robust floating‑point comparison helpers.
 *
 * This header is part of **`src/numeric/scalar/`** and is *header‑only*.  All
 * routines are `constexpr` so they can be evaluated at compile time (e.g. for
 * generating lookup tables in the quadrature module).
 *
 * ### Organization
 * | Function / Alias | Purpose |
 * |---------------------|---------|
 * | `almost_equal`      | Relative + absolute floating‑point comparison (Python‑style `math.isclose`) |
 * | `is_zero`           | Quick zero test using absolute tolerance |
 * | `relative_error`    | Returns *|a‑b| / max(|a|,|b|)* |
 * | `ulp_distance`      | Number of ULPs between two floats (bit‑cast method) |
 * | `almost_equal_ulps` | Division guard that substitutes a fallback when the denominator is ~0 |
 * | `clamp`             | Standard clamp (pre‑C++17 polyfill w/ `constexpr`) |
 * | `sign`              | Returns −1, 0, +1 as an int for the sign of a scalar |
 * | `sqr`, `cube`    | Compile‑time inline power helpers |
 */

#include <bit>
#include <limits>

#include "numeric/scalar/concepts.h"

namespace numeric::scalar::epsilon {
    //====================================================================
    //  Basic comparison helpers
    //====================================================================

    /**
     * @brief Combined relative **and** absolute comparison for two floating‑point numbers.
     *
     * Two values *a* and *b* are considered equal if either of the following holds:
     *   1. \f$|a-b| \le \text{abs\_tol}\f$  *(absolute criterion)*
     *   2. \f$|a-b| \le \text{rel\_tol}\,\max(|a|,|b|)\f$ *(relative criterion)*
     *
     * The default `rel_tol` is set to \f$10\,\epsilon\f$ of the type `T`, which is a
     * safe threshold for most iterative linear solvers.  `abs_tol` defaults to
     * zero because absolute tolerance is problem‑dependent (e.g. 1e‑12 for lengths
     * in metres, 1e‑6 for stresses in Pa).
     *
     * @tparam T       A `std::floating_point` (usually `float`, `double`, or `long double`).
     * @param  a       First value to compare.
     * @param  b       Second value to compare.
     * @param  rel_tol Relative tolerance, expressed as a multiple of machine epsilon.
     * @param  abs_tol Absolute tolerance below which numbers are regarded as equal.
     * @return `true` if the numbers are sufficiently close, `false` otherwise.
     */
    template<RealLike T>
    [[nodiscard]]
    constexpr bool almost_equal(
            T a, T b,
            T rel_tol = std::numeric_limits<T>::epsilon() * static_cast<T>(10),
            T abs_tol = static_cast<T>(0))
    {
        using std::abs;
        const T diff = abs(a - b);
        if (diff <= abs_tol) {
            return true;                             // absolute criterion
        } return diff <= rel_tol *std::max(abs(a), abs(b)); // relative criterion
    }

    /**
     * @brief Exact equality overload for integral types.
     *
     * Included to keep template code generic; integral values are compared with the
     * `==` operator, which is exact and cheap.
     *
     * @tparam Int A type satisfying `std::integral`.
     * @param  a   First integer value.
     * @param  b   Second integer value.
     * @return `true` if `a` equals `b`, `false` otherwise.
     */
     template<SignedIntegral Int>
     [[nodiscard]]
     constexpr bool almost_equal(Int a, Int b)
     {
        return a == b;
     }

     /**
      * @brief Returns `true` if a floating-point value is effectively zero.
      *
      * @tparam T       A `std::floating_point`.
      * @param  x       The value to test.
      * @param  abs_tol Absolute tolerance against zeros.
      */
      template<RealLike T>
      [[nodiscard]]
      constexpr bool is_zero(
              T x,
              T abs_tol = std::numeric_limits<T>::epsilon() * static_cast<T>(10))
      {
        using std::abs;
        return abs(x) <= abs_tol;
      }

      /**
       * @brief Compute the relative error |a-b| / max(|a|, |b|).
       *
       * Intended for unit tests and convergence diagnostics. It does **not**
       * perform division-by-zero checks; call `is_zero()` when either value
       * may be exactly zero.
       *
       * @tparam T A `std::floating_point`.
       * @param  a First value.
       * @param  b Second value.
       * @return Relative error as a non-negative number.
       */
       template<RealLike T>
       [[nodiscard]]
       constexpr T relative_error(T a, T b)
       {
           using std::abs;
           return abs(a - b) / std::max(abs(a), abs(b));
       }
       //====================================================================
       // Bit‑level ULP tools
       //====================================================================
       /**
        * @brief Returns the unsigned distance in ULPs (Units in the Last Place)
        * between @p a and @p b.
        *
        * For IEEE‑754 binary32/64 this is implemented by bit‑casting to the
        * corresponding signed integer, then taking the absolute difference.  Note
        * that comparison across opposite signs is undefined (returns large value).
        *
        * @tparam T `float` or `double`.
        * @param  a First number.
        * @param  b Second number.
        * @return Number of representable values that lie strictly between the two
        *         given bit patterns.
        */
        template<RealLike T>
        [[nodiscard]]
        constexpr auto ulp_distance(T a, T b)
        {
            using UIntT = std::conditional_t<sizeof(T)==4,std::uint32_t,std::uint64_t>;
            constexpr UIntT sign_mask = UIntT{1} << (sizeof(T)*8 - 1);

            UIntT ua = std::bit_cast<UIntT>(a);
            UIntT ub = std::bit_cast<UIntT>(b);

            ua = (ua & sign_mask) ? (sign_mask - ua) : (ua + sign_mask);
            ub = (ub & sign_mask) ? (sign_mask - ub) : (ub + sign_mask);
            return ua > ub ? ua - ub : ub - ua;
        }

        /**
         * @brief Boolean convenience wrapper around `ulp_distance()`.
         *
         * @tparam T        std::floating_point`.
         * @param  a,b      Numbers to compare.
         * @param  max_ulps Maximum allowed ULP gap.
         * @return `true` if `ulp_distance(a,b) ≤ max_ulps`.
         */
         template<RealLike T>
         [[nodiscard]]
         constexpr bool almost_equal_ulps(T a, T b, std::uint64_t max_ulps = 8)
         {
             return ulp_distance(a, b) <= max_ulps;
         }

         //====================================================================
         // Tolerance‑aware relational predicates
         //====================================================================
         /**
          * @brief `true` when `a` is *strictly* less than `b`, beyond tolerance `eps`.
          *
          * @tparam T   `std::floating_point`.
          * @param  a   Candidate smaller value.
          * @param  b   Candidate larger value.
          * @param  eps Symmetric tolerance applied to both numbers.
          * @return `true` if `a < b` and `almost_equal(a,b)` is `false`.
          */
         template<RealLike T>
         [[nodiscard]]
         constexpr bool tolerant_less(T a, T b,
                                      T eps = std::numeric_limits<T>::epsilon()*static_cast<T>(10))
         {
             return (a < b) && !almost_equal(a, b, eps, eps);
         }

         /**
          * @brief `true` when `a` is *strictly* greater than `b`, beyond tolerance.
          *
          * @tparam T   `std::floating_point`.
          * @param  a   Candidate larger value.
          * @param  b   Candidate smaller value.
          * @param  eps Symmetric tolerance.
          * @return `true` if `a > b` and the numbers are not nearly equal.
          */
         template<RealLike T>
         [[nodiscard]]
         constexpr bool tolerant_greater(T a, T b,
                                         T eps = std::numeric_limits<T>::epsilon()*static_cast<T>(10))
         {
             return (a > b) && !almost_equal(a, b, eps, eps);
         }

         /**
          * @brief Inclusive range test with tolerance on both bounds.
          *
          * @tparam T  `std::floating_point`.
          * @param  x  Value to evaluate.
          * @param  lo Lower bound (inclusive).
          * @param  hi Upper bound (inclusive).
          * @param  eps Symmetric tolerance.
          * @return `true` if `x` lies within `[lo, hi]` up to `eps`.
          */
         template<RealLike T>
         [[nodiscard]]
         constexpr bool in_range(T x, T lo, T hi,
                                 T eps = std::numeric_limits<T>::epsilon()*static_cast<T>(10))
         {
             return (tolerant_greater(x, lo, eps) || almost_equal(x, lo, eps, eps)) &&
                    (tolerant_less   (x, hi, eps) || almost_equal(x, hi, eps, eps));
         }

         /**
           * @brief Detect if a float is nearly an integer.
           *
           * @tparam T  `std::floating_point`.
           * @param  x  Value to evaluate.
           * @param  eps Tolerance.
           * @return `true` if |x − round(x)| ≤ eps.
           */
          template<RealLike T>
          [[nodiscard]]
          constexpr bool almost_integer(T x,
                                        T eps = std::numeric_limits<T>::epsilon()*static_cast<T>(10))
          {
              using std::round;
              return almost_equal(x, round(x), eps, eps);
          }

          /**
           * @brief Detect if two floats are of the same sign.
           *
           * @tparam T   `std::floating_point`
           * @param  a    First value
           * @param  b    Second value
           * @return `true` if the two floats are of the same sign, otherwise `false`
           */
          template<RealLike T>
          [[nodiscard]]
          constexpr bool same_sign(T a, T b,
                                   T eps = std::numeric_limits<T>::epsilon()*static_cast<T>(10))
          {
              if (is_zero(a, eps) || is_zero(b, eps)) return false;
              return (a > 0) == (b > 0);
          }

         //====================================================================
         // Finite‑ness
         //====================================================================

         /**
          * @brief Determine if a float is a finite number.
          *
          * @tparam T  `std::floating_point`.
          * @param  x  Value to evaluate.
          * @return `true` if the value is finite, otherwise `false` (e.g. NaN, Inf)
          */
         template<RealLike T>
         [[nodiscard]] constexpr bool is_finite(T x) { return std::isfinite(x); }

} // namespace numeric::scalar
#endif //EPSILON_H
