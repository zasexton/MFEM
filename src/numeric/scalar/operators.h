#pragma once

#ifndef SCALAR_OPERATORS_H
#define SCALAR_OPERATORS_H

#include <optional>
#include <limits>
#include <type_traits>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <typeinfo>
#include <string>

#include "numeric/scalar/concepts.h"
#include "numeric/scalar/epsilon.h"

namespace numeric::scalar::operations {

    //============================================
    // Primative operations
    //============================================

    /**
    * @brief Safely divide two possibly–heterogeneous scalars.
    *
    * The function computes \f$ a / b \f$ and returns the result wrapped in
    * `std::optional`.  Division is **rejected** (`std::nullopt`) in three
    * pathological situations that would otherwise lead to *undefined behaviour*
    * or a **signalled exception**:
    *
    * | Case | Guard                                                                     |
    * |------|---------------------------------------------------------------------------|
    * | Real / integer **divide-by-zero**            | `b == B{0}`                       |
    * | **Complex** divide-by-zero                    | `b == B{Re{0},Re{0}}`             |
    * | **Signed-integer overflow** *min / -1*        | `b == -1 && a == min()`           |
    *
    * ### Type promotion
    * The return type is `numeric::scalar::promote_t<A,B>` which chooses an
    * IEEE-754 or `std::complex<>` type large enough to hold the exact quotient of
    * *any* values of `A` and `B`:
    *
    * *   `int / float`        → `float`
    * *   `unsigned / double`  → `double`
    * *   `float / complex<T>` → `std::complex<T>`
    * *   `int / complex<float>` → `std::complex<float>`
    *
    * See `scalar/traits.h` for full deduction rules.
    *
    * ### `constexpr`
    * All checks are `constexpr`-friendly; you may use this function in constant
    * expressions on C++20 compilers.
    *
    * @tparam A  Any `NumberLike` (integer, real, or complex) — CV-qualified OK.
    * @tparam B  Any `NumberLike` – may differ from *A* in signedness or value-cat.
    *
    * @param a Dividend (numerator)
    * @param b Divisor (denominator)
    *
    * @returns `std::optional<promote_t<A,B>>`
    *          *   **engaged**  – quotient `a / b` when the operation is safe
    *          *   **empty**    – if the divisor is zero or the operation would
    *                             overflow (signed integers).
    *
    * @note No attempt is made to detect floating-point overflow/underflow;
    *       IEEE-754 rules apply (Inf, sub-normals, etc.).
    */
     template <NumberLike A, NumberLike B>
     [[nodiscard]] constexpr std::optional<promote_t<A,B>>
     checked_divide(const A& a, const B& b) noexcept
    {
        using R = promote_t<A,B>;
        using BR = std::remove_cvref_t<B>;

         if constexpr (ComplexLike<R>) {
             using Re = real_t<R>;
             if (b == B{Re{0}, Re{0}} ) {
                 return std::nullopt;
             }
             return static_cast<R>(a) / static_cast<R>(b);
         } else {
             bool zero_divisor = false;
             if constexpr (RealLike<BR>) {
                 zero_divisor = std::fpclassify(b) == FP_ZERO;
             } else {
                 zero_divisor = (b == B{0});
             }
             if (zero_divisor) {
                 return std::nullopt;
             }
             if constexpr (std::signed_integral<R>) {
                 if (static_cast<R>(b) == R(-1) &&
                 static_cast<R>(a) == std::numeric_limits<R>::min()) {
                     return std::nullopt;
                 }
             }
             return static_cast<R>(a) / static_cast<R>(b);
         }
    }


    /**
    * @brief Safely multiply two scalar values of (possibly) different types.
    *
    * The function computes \f$ a \times b \f$ and returns the result wrapped in
    * `std::optional`.
    *
    * | Category                     | Guard-condition – returns `std::nullopt` |
    * |------------------------------|-------------------------------------------|
    * | **Unsigned** integer         | `b != 0` **and** `a > max / b`           |
    * | **Signed** integer           | Overflow test derived from two’s-complement arithmetic<sup>†</sup> |
    * | **Real / Complex** floating  | **No guard** – IEEE-754 already yields `Inf`/`NaN` |
    *
    * <sup>† The signed-integer check prevents UB when the mathematical product
    * lies outside the representable range; e.g. `int32_t` overflow.</sup>
    *
    * ### Type promotion
    * The return type is `promote_t<A,B>` and therefore large enough to hold the
    * exact product *unless* the operands are both integral **and** the product
    * overflows the promoted type’s range – in which case the operation is
    * rejected (nullopt).
    *
    * @tparam A,B Any `NumberLike` (integral, floating, or complex) – they may
    *             differ in CV qualifiers, signedness, etc.
    * @return `std::optional<promote_t<A,B>>`
    *         * **engaged** — `a * b` when safe
    *         * **empty**   — overflow detected **for integral results**.
    *
    * @note For floating-point or complex results, IEEE-754 overflow **is not**
    *       treated as an error; the function simply returns `Inf`/`NaN` exactly
    *       like the built-in operator (`*`).
    */
     template <NumberLike A, NumberLike B>
     [[nodiscard]] constexpr std::optional<promote_t<A,B>>
     checked_multiply(const A& a, const B& b) noexcept
    {
         using R = promote_t<A,B>;
         //------ Integer overflow detection-------------
         if constexpr ( std::integral<R> ) {
             const R ar = static_cast<R>(a);
             const R br = static_cast<R>(b);

             if constexpr ( std::unsigned_integral<R>) {
                 if ( br != 0 && ar > (std::numeric_limits<R>::max() / br)) {
                     return std::nullopt;
                 }
                 return ar * br;
             } else {
                 if ( ar == 0 || br == 0) {
                     return R{0};
                 }
                 const R max = std::numeric_limits<R>::max();
                 const R min = std::numeric_limits<R>::min();
                 if ((br > 0 && (ar>max/br || ar < min/br)) ||
                         (br < 0 && (ar == min || -ar > max / (-br)))) {
                     return std::nullopt;
                 }
                 return ar * br;
             }
         } else {
             //------Floating-point & complex: IEEE-754 rules-----------
             return static_cast<R>(a) * static_cast<R>(b);
         }
    }
    /**
    * @brief Safely compute `a + b` for two (possibly different) scalar types.
    *
    * The routine returns the promoted sum wrapped in `std::optional`.
    * For **integral** results an overflow check is performed; if overflow would
    * occur the function returns `std::nullopt`.
    * For **floating-point** and **complex** results the function simply
    * forwards to the built-in `+` operator because IEEE-754 already yields
    * `Inf`/`NaN` on overflow.
    *
    * ### Guard conditions
    *
    * | Result category           | Overflow test (UB-free) – returns `nullopt` |
    * |---------------------------|---------------------------------------------|
    * | Unsigned integral         | `ar > max − br`                             |
    * | Signed integral           | <code>(br &gt; 0 && ar &gt; max - br)<br> &#124;&#124; (br &lt; 0 && ar &lt; min - br)</code> |
    * | Floating / Complex        | *none* (IEEE handling)                      |
    *
    * @tparam A,B  Any `NumberLike` (integral, floating, or complex) – the two
    *              operands *may differ* in type.
    * @return `std::optional<promote_t<A,B>>`
    *   * engaged — `a + b` when it is representable in the promoted type
    *   * empty   — overflow detected for integral arithmetic
    *
    * @note  Compile-time evaluation (`constexpr`) works as expected.
    */
    template <NumberLike A, NumberLike B>
    [[nodiscard]] constexpr std::optional<promote_t<A,B>>
    checked_addition(const A& a, const B& b) noexcept
    {
        using R = promote_t<A,B>;
        //------ Integer checking---------------------
        if constexpr ( std::integral<R>) {
            const R ar = static_cast<R>(a);
            const R br = static_cast<R>(b);
            const R max = std::numeric_limits<R>::max();
            const R min = std::numeric_limits<R>::min();

            if constexpr ( std::unsigned_integral<R> ){
                if ( ar > max - br) {
                    return std::nullopt;
                } else {
                    return ar + br;
                }
            } else {
                if ((br > 0 && ar > max - br) ||
                        (br < 0 && ar < min - br)) {
                    return std::nullopt;
                } else {
                    return ar + br;
                }
            }
        } else {
            //------RealLike or ComplexLike IEEE-754 overflow rules------
            return static_cast<R>(a) + static_cast<R>(b);
        }
    }

    /**
    * @brief Safely compute `a − b` for two (possibly different) scalar types.
    *
    * The function returns the promoted difference wrapped in `std::optional`.
    * For **integral** results it checks for overflow/underflow; if either would
    * occur the function yields `std::nullopt`.
    * For **floating-point** and **complex** results the function simply forwards
    * to the built-in operator (`Inf`/`NaN` signal overflow per IEEE-754).
    *
    * ### Guard conditions
    *
    * | Result category           | Overflow / underflow test – returns `nullopt` |
    * |---------------------------|-----------------------------------------------|
    * | Unsigned integral         | `ar < br`                                     |
    * | Signed integral           | <code>(br &gt; 0 && ar &lt; min + br)<br> &#124;&#124; (br &lt; 0 && ar &gt; max + br)</code> |
    * | Floating / Complex        | *none*                                        |
    *
    * @tparam A,B  Any `NumberLike` (integral, floating, or complex) – the two
    *              operands *may differ* in type.
    * @return `std::optional<promote_t<A,B>>`
    *   * engaged — `a − b` when representable in the promoted type
    *   * empty   — overflow/underflow detected for integral arithmetic
    *
    * @note Works in `constexpr` contexts.
    */
     template <NumberLike A, NumberLike B>
     [[nodiscard]] constexpr std::optional<promote_t<A,B>>
     checked_sub(const A& a, const B& b) noexcept
    {
         using R = promote_t<A,B>;

         if constexpr (std::integral<R>) {
             const R ar = static_cast<R>(a);
             const R br = static_cast<R>(b);
             const R max = std::numeric_limits<R>::max();
             const R min = std::numeric_limits<R>::min();

             if constexpr (std::unsigned_integral<R>) {
                 if (ar < br) {
                     return std::nullopt;
                 } else {
                     return ar - br;
                 }
             } else {
                 if ((br>0 && ar<min+br)||
                         (br<0 && ar>max+br)) {
                     return std::nullopt;
                 } else {
                     return ar - br;
                 }
             }
         } else {
             return static_cast<R>(a) - static_cast<R>(b);
         }
    }

    /**
     * @brief Safely perform a remainder `a % b` for two (possibly different) scalar types.
     *
     *  Performs the *Euclidean* remainder **a % b** (integers) or
     *  **fmod(a,b)** (floating point) **with full run-time guards** against
     *  the undefined-behaviour cases mandated by the C++ Standard.
     *
     * ### Guard conditions
     *  | category          | guard performed | Overflow / underflow test – returns `nullopt`    |             |
     *  |-------------------|-----------------|--------------------------------------------------|
     *  | **Integral**      | • divisor-zero  | `b == 0`                                         |
     *  |                   | • *INT_MIN % -1* overflow (2-compl) | `a == min` **and** `b == -1` |
     *  | **Floating-point**| • divisor-zero  | `b == 0` (exact zero – matches `checked_divide`) |
     *  | **Complex**       | • divisor is exactly `0 + 0i` |
     * @tparam A,B  Any `NumberLike` (integral, floating, or complex) – the two
     *              operands *may differ* in type.
     *
     * @param a  first value
     * @param b  second value
     * @return `std::optional<promote_t<A,B>>`
     *   * engaged — `a − b` when representable in the promoted type
     *   * empty   — overflow/underflow detected for integral arithmetic
     *
     * @note  The function is `constexpr`-friendly:* all branch conditions are
     *  `if constexpr` or value-dependent and thus evaluable during constant
     *  expression evaluation.
     */
    template <NumberLike A, NumberLike B>
    [[nodiscard]] constexpr std::optional<promote_t<A,B>>
    checked_mod(const A& a, const B& b) noexcept
    {
        using R = promote_t<A,B>;

        if constexpr (std::integral<R>) {
            if (b == B{0}) {
                return std::nullopt;
            }
            if constexpr (std::signed_integral<R>) {
                if (a == std::numeric_limits<R>::min() && b == B{-1}) {
                    return std::nullopt;
                }
            }
            return static_cast<R>(a % b);
        } else {
            return (b == B{0}) ? std::nullopt : std::optional<R>{std::fmod(static_cast<R>(a),
                                                                           static_cast<R>(b))};
        }
    }

    /**
    * ---------------------------------------------------------------------------
    *  @brief Safe unary **negation** (`-x`) with run-time overflow checks
    *
    *  ### Template parameter
    *  | name | requirement  |
    *  |------|--------------|
    *  | `T`  | `NumberLike` (integral, IEEE floating-point, or `std::complex`) |
    *
    *  ### Behaviour & failure modes
    *  | category                | guarded condition      | returns `std::nullopt` when …            |
    *  |-------------------------|------------------------|-------------------------------------------|
    *  | **signed integral**     | *two’s-complement* wrap| `x == std::numeric_limits<T>::min()`      |
    *  | **unsigned integral**   | N/A — cannot negate    | **always** (operation is ill-defined)     |
    *  | **floating-point**      | none (IEEE-754 defined)| never – result is `-x`                    |
    *  | **complex**             | none                   | never – result is `-x`                    |
    *
    *  ### Complexity & constexpr
    *  *O(1)*, `constexpr`-friendly – all checks are simple comparisons that
    *  fold at compile time whenever `x` is a constant expression.
    *
    *  @see checked_abs, checked_add, checked_sub
    * ---------------------------------------------------------------------------
    */
    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_negate(const T& x) noexcept
    {
        if constexpr (std::is_signed_v<T> && std::is_integral_v<T>) {
            if (x == std::numeric_limits<T>::min()) {
                return std::nullopt;
            } else {
                return static_cast<T>(-x);
            }
        } else if constexpr (std::is_unsigned_v<T> && std::is_integral_v<T>) {
            return std::nullopt;
        } else {
            return static_cast<T>(-x);
        }
    }

    /**
    *  @brief Safe **absolute value**  magnitude with overflow guards
    *
    *
    *  ### Template parameter
    *  | name | requirement  |
    *  |------|--------------|
    *  | `T`  | `NumberLike` |
    *
    *  ### Behaviour & failure modes
    *  | category            | guard performed                           | `std::nullopt` when…                      |
    *  |---------------------|-------------------------------------------|-------------------------------------------|
    *  | **signed integral** | overflow on `abs(INT_MIN)`                | `x == std::numeric_limits<T>::min()`      |
    *  | **unsigned integral**| no overflow possible                     | never                                     |
    *  | **floating-point**  | divisor-zero **not relevant**             | never – `std::fabs` used                  |
    *  | **complex**         | magnitude computed via `std::abs`         | never (overflow would yield `inf`)        |
    *
    *  ### Notes
    *  * For *unsigned* inputs the function is a cheap pass-through (the absolute
    *    value is the number itself).
    *  * For *complex* inputs the magnitude is always a **real** scalar.
    */
    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<std::conditional_t<ComplexLike<T>, real_t<T>, T>>
    checked_abs(const T& x) noexcept
    {
        using R = std::conditional_t<ComplexLike<T>, real_t<T>, T>;

        if constexpr (std::is_signed_v<T> && std::is_integral_v<T>) {
            if (x == std::numeric_limits<T>::min()) {
                return std::nullopt;
            } else {
                return static_cast<R>(x < 0 ? -x : x);
            }
        } else if constexpr (std::is_unsigned_v<T> && std::is_integral_v<T>) {
            return static_cast<R>(x);
        } else if constexpr (ComplexLike<T>) {
            return std::abs(x);
        } else {
            return std::fabs(x);
        }
    }

    /**
     * ---------------------------------------------------------------------------
     *  @brief  **Safe bitwise shift** with full run-time validation
     *
     *  ```cpp
     *  auto y = checked_shift_left ( value, 3  );   // std::optional<T>
     *  auto z = checked_shift_right( value, 10 );   //      ″
     *  ```
     *
     *  ### Template parameters
     *  | name | requirement |
     *  |------|-------------|
     *  | `T`  | `IntegerLike` (signed *or* unsigned) |
     *
     *  ### Function arguments
     *  | arg | semantics                                    |
     *  |-----|----------------------------------------------|
     *  | `value` | the integer to be shifted                |
     *  | `count` | number of bit-positions to shift (`size_t`) |
     *
     *  ### Guarantees & failure modes
     *  Both helpers return `std::nullopt` when the operation would be
     *  **ill-defined or lossy** on the target platform:
     *
     *  | guard                                     | reason / UB avoided             |
     *  |-------------------------------------------|---------------------------------|
     *  | `count ≥ bit_width<T>`                    | C++ spec: shift ≥ width is UB   |
     *  | `count < 0` (theoretically)               | negative shift is UB            |
     *  | **left-shift**: result would overflow `T` | bits would be lost / sign flip  |
     *  | **left-shift, signed T** & `value < 0`    | shifting a negative is UB       |
     *
     *  When all checks pass, the helpers return the shifted value wrapped in
     *  `std::optional<T>`.
     *
     *  ### Complexity & compile-time
     *  *O(1)*, **`constexpr`-friendly** – all tests fold at compile-time for
     *  constant expressions.  On most compilers the fast path is identical to the
     *  raw shift instruction.
     *
     *  @see checked_add, checked_sub, checked_multiply, checked_divide
     * ---------------------------------------------------------------------------
     */
    template <IntegerLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_shift_left(T value, std::size_t count) noexcept
    {
        constexpr std::size_t W = std::numeric_limits<std::make_unsigned_t<T>>::digits;

        if (count >= W) {
            return std::nullopt;
        }

        if constexpr ( std::is_signed_v<T> ) {
            if (value < 0) {
                return std::nullopt;
            }
        }

        using U = std::make_unsigned_t<T>;
        const U uval = static_cast<U>(value);
        const U limit = std::numeric_limits<U>::max() >> count;

        if (uval > limit) {
            return std::nullopt;
        }

        U shifted = uval << count;
        return static_cast<T>(shifted);
    }


    template <IntegerLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_shift_right(T value, std::size_t count) noexcept
    {
        constexpr std::size_t W = std::numeric_limits<std::make_unsigned_t<T>>::digits;
        if (count >= W) {
            return std::nullopt;
        }
        /* For right shifts the C++ rules are well-defined:
           – unsigned -> logical
           – signed   -> arithmetic (implementation-defined but ubiquitous)
           Overflow cannot occur, we only guard the count. */
        if constexpr (std::is_signed_v<T>) {
            return static_cast<T>(value >> count); // arithmetic shift
        } else {
            return static_cast<T>(value >> count); // logical shift
        }
    }

    /**
     * @brief Saturating addition (clamps to max/min instead of nullopt)
     */
     template <IntegerLike T>
     [[nodiscard]] constexpr T saturating_int_addition(T a, T b) noexcept
    {
         auto res = checked_int_addition(a, b);
         if (res) {
             return *res;
         } else {
             return (b > 0) ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
         }
    }

    /**
     * @brief Saturating subtraction (clamps to the max/min for the nullopt)
     */
     template <IntegerLike T>
     [[nodiscard]] constexpr T saturating_int_subtraction(T a, T b) noexcept
    {
         auto res = checked_int_subtraction(a, b);
         if (res) {
             return *res;
         } else {
             return (b > 0) ? std::numeric_limits<T>::min() : std::numeric_limits<T>::max();
         }
    }

     template <IntegerLike T, RealLike U>
     [[nodiscard]] constexpr U saturating_int_division(T a, T b) noexcept
     {
         if (b == T{0}) {
             if (a > T{0}) return std::numeric_limits<U>::infinity();
             if (a < T{0}) return -std::numeric_limits<U>::infinity();
             return std::numeric_limits<U>::quiet_NaN();
         }

         // Cast to the real type *before* dividing-avoids INT_MIN/-1 overflow.
         return static_cast<U>(a) / static_cast<U>(b);
     }

    //============================================
    // Basic Functions
    //============================================

    // Power Functions

    /**
     * @brief (x)^(1/2) with run-time domain checks.
     *
     * | Category                  | Behaviour                                                |
     * |---------------------------|----------------------------------------------------------|
     * | **RealLike** (`float`, `double`, ...) |  **x < 0 ⇒** `std::nullopt`  <br>• otherwise `√x` |
     * | **IntegerLike**           | **x < 0 ⇒** `std::nullopt`  <br>• `x ≥ 0` -> returns `⌊√x⌋` **iff** that square root is *exact* (perfect square); otherwise `std::nullopt`.  <br>  This avoids silent truncation. |
     * | **ComplexLike**           | Always succeeds – returns the principal branch of the complex square root (‖ℑ√x‖ ≥ 0). |
     *
     * The function is `constexpr`-friendly (works in constant-evaluation contexts)
     * and never throws.
     *
     * @tparam T  Any type that fulfils the `NumberLike` concept
     *            (`RealLike ∪ IntegerLike ∪ ComplexLike`).
     * @param  x  Input value
     * @return    `std::optional<T>` holding the result *or* `std::nullopt`
     *            if the operation is undefined for @p x.
     *
     * @note For integer inputs we promote to `long double` internally when
     *       calling `std::sqrt` to avoid precision loss for 64-bit values.
     *       The result is then cast back to `T` only if it squares back
     *       exactly to the original operand.
     */
    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_sqrt(const T& x) noexcept
    {
        if constexpr (RealLike<T>) {
            if (x < T{0}) {
                return std::nullopt;
            }
            return std::sqrt(x);
        } else if constexpr (IntegerLike<T>) {
            if (x < 0) {
                return std::nullopt;
            }
            long double tmp = std::sqrt(static_cast<long double>(x));

            //Round-trip check for perfect squares
            T root = static_cast<T>(tmp + 0.5L);
            return (root * root == x) ? std::optional<T>{root} : std::nullopt;
        } else {
            return std::sqrt(x);
        }
    }

    /**
     * @brief ∛x (cube-root) with run-time domain & overflow checks.
     *
     * | Category          | Behaviour                                                                                     |
     * |-------------------|------------------------------------------------------------------------------------------------|
     * | **RealLike**      | Always succeeds – returns `cbrt(x)` (negative inputs are fine).                               |
     * | **IntegerLike**   | Returns the **exact** integer cube-root *iff* `x` is a perfect cube; otherwise `std::nullopt`.|
     * | **ComplexLike**   | Always succeeds – returns the principal complex cube-root.                                    |
     *
     * The function is `constexpr` capable and never throws.
     *
     * @tparam T  A type satisfying the **NumberLike** concept
     *           (`RealLike ∪ IntegerLike ∪ ComplexLike`).
     * @param  x  Operand
     * @return    `std::optional<T>` holding the result or `std::nullopt`
     *            where the operation is undefined (see table above).
     *
     * ### Rationale
     * *  For floating-point and complex numbers the cube root is defined for
     *    all finite values, so we simply forward to `std::cbrt` / `std::pow`.
     * *  For integers an *inexact* cube-root is almost never what numerical
     *    code wants – we therefore succeed **only** on perfect cubes, using a
     *    round-trip test to verify the result.
     */
    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_cbrt(const T& x) noexcept
    {
        if constexpr (RealLike<T>) {
            return std::cbrt(x);
        } else if constexpr (IntegerLike<T>) {
            long double tmp = std::cbrt(static_cast<long double>(x));

            // Round to nearest integer and verify perfect-cube property
            T root = static_cast<T>(tmp + (tmp < 0 ? -0.5L : 0.5L));
            return (root * root * root == x) ? std::optional<T>{root} : std::nullopt;
        } else {
            using Re = real_t<T>;
            constexpr Re one_third = Re(1)/Re(3);
            return std::pow(x, one_third);
        }
    }

    /**
    *  @brief (base ^ exponent)  with full run‑time guards
    *
    *
    *  • Result type = promote_t<T,U>         (same rule as checked_divide, etc.)
    *  • Returns std::nullopt for any case that would:
    *
    *       – overflow a (signed/unsigned) integral result
    *       – produce NaN / ±Inf for floating results
    *       – be mathematically undefined in ℝ  (negative base & non‑integer exp)
    *
    *  • Complex results always succeed (principal branch of std::pow)
    */
    template <NumberLike T, RealLike U>
    [[nodiscard]] constexpr std::optional<T>
    checked_pow(const T& x, const U& y) noexcept
    {
        if constexpr (ComplexLike<T>) {
            using R = real_t<T>;
            return std::pow(x,static_cast<R>(y));
        } else if constexpr (RealLike<T>) {
            return std::pow(x,static_cast<T>(y));
        } else if constexpr (IntegerLike<T>) {
            if (std::trunc(y) != y || y < U{0}) {
                std::nullopt;
            }
            using UInt = std::make_unsigned_t<T>;
            UInt e = static_cast<UInt>(exp);
            T acc{1};
            T b = static_cast<T>(x);
            while (e) {
                if (e & 1u) {
                    auto p = checked_multiply(acc,b);
                    if (!p) { return std::nullopt; }
                    acc = *p;
                }
                e >>= 1u;
                if (e) {
                    auto p = checked_multiply(b,b);
                    if (!p) { return std::nullopt; }
                    b = *p;
                }
            }
            return acc;
        } else {
            throw std::logic_error("checked_pow: unsupported types: "
                                    + std::string(typeid(T).name()) + ", "
                                    + std::string(typeid(U).name()) + '>');
        }
    }
    /**
    template <NumberLike T, NumberLike>
    [[nodiscard]] constexpr std::optional<T>
    checked_hypot(const T& x) noexcept
    {
    }

    // Exponential Functions
    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_exp(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_exp2(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_expml(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_log(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_log2(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_log10(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_lp(const T& x) noexcept
    {
    }

    // Trigonometric Functions
    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_sin(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_cos(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_tan(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_asin(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_acos(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_atan(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_atan2(const T& x) noexcept
    {
    }

    // Hyperbolic Functions
    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_sinh(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_cosh(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_tanh(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_asinh(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_acosh(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_atanh(const T& x) noexcept
    {
    }

    // Error and Gamma functions
    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_erf(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_erfc(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_erfinv(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_erfcinv(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_tgamma(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_lgamma(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_digamma(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_trigamma(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_polygamma(const T& x) noexcept
    {
    }

    // Nearest integer and floating-point operations
    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_ceil(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_floor(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_trunc(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_round(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_nearbyint(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_rint(const T& x) noexcept
    {
    }

    // Floating-point manipulation functions
    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_frexp(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_ldexp(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_scalbn(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_ilogb(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_nextafter(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_copysign(const T& x) noexcept
    {
    }

    // Classification and comparison
    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_fpclassify(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_isfinite(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_isinf(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_isnan(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_isnormal(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_signbit(const T& x) noexcept
    {
    }

     */

    /**
     * @brief Safely test whether *a* &gt; *b*.
     *
     * Same semantics as `checked_isless`, but for the *greater-than* relation.
     * Unordered or undefined comparisons yield `std::nullopt`.
     */
    template <NumberLike A, NumberLike B>
    [[nodiscard]] constexpr std::optional<bool>
    checked_isgreater(const A& a, const B& b) noexcept
    {
        using R = promote_t<A,B>;

        if constexpr (ComplexLike<R>) {
            return std::nullopt;
        } else {
            const R lhs = static_cast<R>(a);
            const R rhs = static_cast<R>(b);

            if constexpr (RealLike<R>) {
                if (std::isnan(lhs) || std::isnan(rhs)) {
                    return std::nullopt;
                }
            }
            return lhs > rhs;
        }
    }

    //template <NumberLike T>
    //[[nodiscard]] constexpr std::optional<T>
    //checked_isgreaterequal(const T& x) noexcept
    //{
    //}

    /**
     * @brief Safely test whether *a* &lt; *b*.
     *
     * | Case                                 | Result                                          |
     * |--------------------------------------|-------------------------------------------------|
     * | **Integral operands**                | `std::optional<bool>{ a < b }` (never empty).   |
     * | **Floating-point operands**          | *If* either input is `NaN` → `std::nullopt`,<br>
     * |                                      | otherwise the boolean result of `a < b`.        |
     * | **At least one complex type**        | `std::nullopt` — ordering of complex numbers    |
     * |                                      | is undefined, we refuse to guess.               |
     *
     * The operands may have *different* types.  They are first promoted with
     * `promote_t<A, B>` (exactly the same rule used by `checked_add`, etc.),
     * then the comparison is performed on the promoted value.
     *
     * @tparam A,B  Any `NumberLike` (real, integer, or complex).
     * @return      `std::optional<bool>` – engaged when the comparison is defined.
     *
     * @note The function is `constexpr` and `noexcept`; it can be used in both
     *       compile-time and run-time contexts without throwing.
     */
    template <NumberLike A, NumberLike B>
    [[nodiscard]] constexpr std::optional<bool>
    checked_isless(const A& a, const B& b) noexcept
    {
        using R = promote_t<A, B>;

        if constexpr (ComplexLike<R>) {
            return std::nullopt;
        } else {
            const R lhs = static_cast<R>(a);
            const R rhs = static_cast<R>(b);

            if constexpr (RealLike<R>) {
                if (std::isnan(lhs) || std::isnan(rhs)) {
                    return std::nullopt;
                }
            }
            return lhs < rhs;
        }
    }

    /**
    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_islessequal(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_islessgreater(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_isunordered(const T& x) noexcept
    {
    }

    // Special Functions
    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_assoc_laguerre(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_assoc_legendre(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_beta(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_comp_ellint_1(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_comp_ellint_2(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_comp_ellint_3(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_cyl_bessel_i(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_cyl_bessel_j(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_cyl_bessel_k(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_hankel(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_cyl_neumann(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_ellint_1(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_ellint_2(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_ellint_3(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_expint(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_hermite(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_legendre(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_laguerre(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_riemann_zeta(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_sph_bessel(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_sph_legendre(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_sph_neumann(const T& x) noexcept
    {
    }

    // Airy & Airy-prime
    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_Ai(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_Bi(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_lambert_w(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_zeta(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_polylog(const T& x) noexcept
    {
    }

    // Incomplete trig integrals
    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_sinint(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_cosint(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_Ei(const T& x) noexcept
    {
    }

    // Activation Functions
    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_sigmoid(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_softplus(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_gelu(const T& x) noexcept
    {
    }

    template <NumberLike T>
    [[nodiscard]] constexpr std::optional<T>
    checked_swish(const T& x) noexcept
    {
    }
     */
}

#endif //SCALAR_OPERATORS_H
