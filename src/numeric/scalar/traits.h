#pragma once

#ifndef SCALAR_TRAITS_H
#define SCALAR_TRAITS_H

#include <complex>
#include <type_traits>

namespace numeric::scalar {

    template <typename T>
    inline constexpr bool is_complex_v = false;

    template <typename U>
    inline constexpr bool is_complex_v<std::complex<U>> = true;

    template <typename T>
    struct real
    {
        using type = T;
    };

    template <typename U>
    struct real<std::complex<U>>
    {
        using type = U;
    };

    template <typename T>
    using real_t = typename real<T>::type;

    template <typename T>
    struct imag
    {
        using type = void;
    };

    template <typename U>
    struct imag<std::complex<U>>
    {
        using type = U;
    };

    //-----------------------------
    // promote
    //-----------------------------

    template <typename A, typename B, typename = void>
    struct promote {};


    template <std::floating_point A, typename B>
    requires is_complex_v<B>
    struct promote<A, B>
    {
        using type = std::complex<std::common_type_t<A, real_t<B>> >;
    };

    template <typename A, std::floating_point B>
    requires is_complex_v<A>
    struct promote<A, B> : promote<B, A> { };

    template <typename A, typename B>
    requires (is_complex_v<A> && is_complex_v<B>)
    struct promote<A, B>
    {
        using type = std::complex<std::common_type_t<real_t<A>, real_t<B> >>;
    };

    template <typename A, typename B>
    requires (!is_complex_v<A> && !is_complex_v<B>)
    struct promote<A, B>
    {
        using type = std::common_type_t<A, B>;
    };

    template<class A, class B>
    using promote_t = typename promote<A,B>::type;
}
#endif //SCALAR_TRAITS_H
