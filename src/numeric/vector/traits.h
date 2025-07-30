#pragma once

#ifndef VECTOR_TRAITS_H
#define VECTOR_TRAITS_H

#include <cstddef>
#include <type_traits>

#include "numeric/scalar/concepts.h"
#include "numeric/vector/concepts.h"

namespace numeric::vector {

    template<class> struct is_vector : std::false_type {};

    template<class S, std::size_t N, class Layout>
    struct is_vector<Vector<S,N,Layout>> : std::true_type {};

    template<class T>
    inline constexpr bool is_vector_v =
            is_vector< std::remove_cv_t<std::remove_reference_t<T>>>::value;

#if defined(__cpp_concepts)
    template<class T>
    concept VectorLike = is_vector_v<T>;
#else
    template<class T>
    struct VectorLike : std::bool_constant<is_vector_v<std::remove_cv_t<T>>> {};
#endif

    template<class> struct scalar_t;

    template<class S, std::size_t N, class Layout>
    struct scalar_t< Vector<S,N,Layout> > {using type = S; };

    template<class V>
    using scalar_t_t = typename scalar_t<std::remove_cv_t<V>>::type;

    template<class> struct static_size;

    template<class S, std::size_t N, class Layout>
    struct static_size< Vector<S,N,Layout> >
            : std::integral_constant<std::size_t, N> {};

    template<class V>
    inline constexpr std::size_t static_size_v =
            static_size<std::remove_cv_t<V>>::value;

    template<VectorLike V>
    constexpr std::size_t extent(const V& v) noexcept
    {
        if constexpr (static_size_v<V> != Dynamic)
        {
            return static_size_v<V>;
        } else {
            return v.size();
        }
    }

    template<class> struct rank : std::integral_constant<int, 1>{};

    template<class V>
    inline constexpr int rank_v = rank<std::remove_cv_t<V>>::value;

    template<class> struct layout_t;

    template<class S, std::size_t N, class Layout>
    struct layout_t<Vector<S,N,Layout>> {using type = Layout; };

    template<class V>
    using layout_t_t = typename layout_t<std::remove_cv_t<V>>::type;

#if defined(__cpp_concepts)
    template<class V>
    concept DenseVectorLike  = VectorLike<V> &&
                               std::same_as<layout_t_t<V>, Dense>;

    template<class V>
    concept SparseVectorLike = VectorLike<V> &&
                               std::same_as<layout_t_t<V>, Sparse>;

    template<class V>
    concept StaticVectorLike  = VectorLike<V> &&
                                (static_size_v<V> != Dynamic);

    template<class V>
    concept DynamicVectorLike = VectorLike<V> &&
                                (static_size_v<V> == Dynamic);
#else
    template<class V>
    using DenseVectorLike  = std::bool_constant<
    VectorLike<V> && std::is_same_v<layout_t_t<V>, Dense>>;

    template<class V>
    using SparseVectorLike = std::bool_constant<
    VectorLike<V> && std::is_same_v<layout_t_t<V>, Sparse>>;

    template<class V>
    using StaticVectorLike = std::bool_constant<
    VectorLike<V> && static_size_v<V> != Dynamic>;

    template<class V>
    using DynamicVectorLike = std::bool_constant<
    VectorLike<V> && static_size_v<V> == Dynamic>;
#endif

} // numeric::vector

namespace numeric::scalar {

    using numeric::vector::Vector;

    template<class S1, class S2, std::size_t N1, std::size_t N2>
    struct promote< Vector<S1,N1>, Vector<S2,N2>>
    {
        using type = Vector<
                typename promote<S1,S2>::type,
                (N1 == N2 ? N1 : numeric::vector::Dynamic)>;
    };

    template<class S, std::size_t N, class U>
    struct promote<Vector<S,N>,U>
    {
        using type = Vector< typename promote<S,U>::type, N>;
    };

    template<class U, class S, std::size_t N>
    struct promote<U, Vector<S,N>>
    {
        using type = Vector< typename promote<U,S>::type, N>;
    };
}
#endif //VECTOR_TRAITS_H
