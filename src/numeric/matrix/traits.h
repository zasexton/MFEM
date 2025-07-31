#pragma once

#ifndef MATRIX_TRAITS_H
#define MATRIX_TRAITS_H
#include <cstddef>
#include <type_traits>

#include "numeric/scalar/concepts.h"
#include "numeric/vector/concepts.h"     // tags / Dynamic
#include "numeric/matrix/concepts.h"

namespace numeric::matrix
{
    // ── detect Matrix ─────────────────────────────────────────
    template<class> struct is_matrix : std::false_type {};

    template<class S, std::size_t R, std::size_t C, class Layout, class Order>
    struct is_matrix< Matrix<S,R,C,Layout,Order> > : std::true_type {};

    template<class T>
    inline constexpr bool is_matrix_v =
            is_matrix< std::remove_cv_t<std::remove_reference_t<T>> >::value;

#if defined(__cpp_concepts)
    template<class T>
    concept MatrixLike = is_matrix_v<T>;
#else
    template<class T>
    using MatrixLike = std::bool_constant< is_matrix_v<T> >;
#endif

    // ── scalar type ───────────────────────────────────────────
    template<class> struct scalar_t;
    template<class S, std::size_t R, std::size_t C, class Layout, class Order>
    struct scalar_t< Matrix<S,R,C,Layout,Order> > { using type = S; };

    template<class M> using scalar_t_t =
            typename scalar_t<std::remove_cv_t<M>>::type;

    // ── static extents ────────────────────────────────────────
    template<class> struct static_rows;
    template<class> struct static_cols;

    template<class S, std::size_t R, std::size_t C, class Layout, class Order>
    struct static_rows< Matrix<S,R,C,Layout> >
            : std::integral_constant<std::size_t, R> {};

    template<class S, std::size_t R, std::size_t C, class Layout, class Order>
    struct static_cols< Matrix<S,R,C,Layout> >
            : std::integral_constant<std::size_t, C> {};

    template<class M>
    inline constexpr std::size_t static_rows_v =
    static_rows<std::remove_cv_t<M>>::value;

    template<class M>
    inline constexpr std::size_t static_cols_v =
    static_cols<std::remove_cv_t<M>>::value;

    // run-time helpers
    template<MatrixLike M>
    constexpr std::size_t rows(const M& m) noexcept
    {
        if constexpr (static_rows_v<M> != Dynamic) {
            return static_rows_v<M>;
        } else{
            return m.rows();
        }
    }

    template<MatrixLike M>
    constexpr std::size_t cols(const M& m) noexcept
    {
        if constexpr (static_cols_v<M> != Dynamic) {
            return static_cols_v<M>;
        } else{
            return m.cols();
        }
    }

    // ── rank (always 2) ───────────────────────────────────────
    template<class> struct rank : std::integral_constant<int, 2> {};
    template<class M> inline constexpr int rank_v = rank<std::remove_cv_t<M>>::value;

    // ── layout tag ────────────────────────────────────────────
    template<class> struct layout_t;
    template<class S, std::size_t R, std::size_t C, class Layout, class Order>
    struct layout_t< Matrix<S,R,C,Layout> > { using type = Layout; };

    template<class M>
    using layout_t_t = typename layout_t<std::remove_cv_t<M>>::type;

    //── storage order ───────────────────────────────────────────────
    template<class> struct order_t;

    template<class S, std::size_t R, std::size_t C, class Layout, class Order>
    struct order_t< Matrix<S,R,C,Layout,Order> >
    {
        using type = Order;
    };

    template<class M>
    using order_t_t = typename order_t<std::remove_cv_t<M>>::type;

    template<class M>
    inline constexpr bool is_row_major_v = std::is_same_v<order_t_t<M>,RowMajor>;

    template<class M>
    inline constexpr bool is_col_major_v = std::is_same_v<order_t_t<M>, ColMajor>;

#if defined(__cpp_concepts)
    template<class M> concept RowMajorMatrixLike = MatrixLike<M> && is_row_major_v<M>;
    template<class M> concept ColMajorMatrixLike = MatrixLike<M> && is_col_major_v<M>;
#endif

    template<MatrixLike M>
    constexpr std::size_t row_stride(const M& m) noexcept
    {
        return m.row_stride();   // compile-time in static cases is okay
    }

    template<MatrixLike M>
    constexpr std::size_t col_stride(const M& m) noexcept
    {
        return m.col_stride();
    }

    template<MatrixLike M>
    inline constexpr bool is_contiguous_v(const M& m) noexcept
    {
        return is_row_major_v<M> ? col_stride(m)==1 : row_stride(m)==1;
    }

#if defined(__cpp_concepts)
    template<class M>
    concept ContiguousMatrixLike = MatrixLike<M> && is_contiguous_v<M>;
#endif

    template<class> struct data_ptr;
    template<class S, std::size_t R, std::size_t C, class Layout, class Order>
    struct data_ptr< Matrix<S,R,C,Layout,Order> >
    {
        using type = S*;
    };

    template<class M> using data_ptr_t = typename data_ptr<std::remove_cv_t<M>>::type;

    template<ContiguousMatrixLike M>
    constexpr data_ptr_t<M> data(M& m) noexcept
    {                                   // const-overload if you want too
        return m.data();                // assumes .data() member
    }

    template<ContiguousMatrixLike M>
    constexpr std::add_pointer_t<const scalar_t_t<M>>
    data(const M& m) noexcept { return m.data(); }

    template<ContiguousMatrixLike M>
    constexpr std::size_t lda(const M& m) noexcept
    {
        if constexpr (is_col_major_v<M>) {
            return row_stride(m);
        } else{
            return col_stride(m);
        }
    }

#if defined(__cpp_concepts)
    template<class M>
    concept DenseMatrixLike  = MatrixLike<M> && std::same_as<layout_t_t<M>, Dense>;

    template<class M>
    concept SparseMatrixLike = MatrixLike<M> && std::same_as<layout_t_t<M>, Sparse>;

    template<class M>
    concept StaticMatrixLike = MatrixLike<M> &&
                               (static_rows_v<M> != Dynamic && static_cols_v<M> != Dynamic);

    template<class M>
    concept DynamicMatrixLike = MatrixLike<M> &&
                                (static_rows_v<M> == Dynamic || static_cols_v<M> == Dynamic);

    template<class M>
    concept ContiguousMatrixLike = MatrixLike<M> && requires(const M& m) {
        { is_continuous(m) } -> std::same_as<bool>;
    };

    template<class M>
    concept BlasCompatibleMatrixLike = ContiguousMatrixLike<M> && scalar::is_blas_scalar<scalar_t_t<M>>;
#else
//  non-concept fall-backs (optional)
#endif
} // namespace numeric::matrix

// ──────────────────────────────────────────────────────────────
//  Matrix-aware promotions (slot into scalar::promote machinery)
// ──────────────────────────────────────────────────────────────
namespace numeric::scalar
{
    using numeric::vector::Vector;
    using numeric::matrix::Matrix;

    // Matrix ⊕ Matrix
    template<class S1, class S2,
            std::size_t R1, std::size_t C1,
            std::size_t R2, std::size_t C2, class O1, class O2>
    struct promote< Matrix<S1,R1,C1,O1>, Matrix<S2,R2,C2,O2> >
    {
        using type = Matrix<
                typename promote<S1,S2>::type,
                (R1 == R2 ? R1 : numeric::matrix::Dynamic),
                (C1 == C2 ? C1 : numeric::matrix::Dynamic),
                O1>;
    };

    // 1. Matrix  ⊕  Vector  (row-broadcast)
    template<class S1, class S2,
            std::size_t R, std::size_t C, class LMat, class OMat,
            std::size_t N, class LVec>
    struct promote< Matrix<S1,R,C,LMat,OMat>,
            Vector<S2,N,LVec> >
    {
        static_assert( (C == N) || (C == Dynamic) || (N == numeric::vector::Dynamic),
        "Vector length must match Matrix columns for row-broadcast." );

        using type = Matrix< typename promote<S1,S2>::type,
                R,
                (C == N ? C : Dynamic),
                LMat, OMat>;
    };

    // 2. Vector  ⊕  Matrix  (commutative)
    template<class S1, class S2,
            std::size_t N, class LVec,
            std::size_t R, std::size_t C, class LMat, class OMat>
    struct promote< Vector<S1,N,LVec>,
            Matrix<S2,R,C,LMat,OMat> >
    {
        static_assert( (C == N) || (C == Dynamic) || (N == numeric::vector::Dynamic),
        "Vector length must match Matrix columns for row-broadcast." );

        using type = Matrix< typename promote<S1,S2>::type,
                R,
                (C == N ? C : Dynamic),
                LMat, OMat>;
    };

    // Matrix ⊕ scalar
    template<class S , std::size_t R, std::size_t C, class L, class O, class U>
    struct promote< Matrix<S,R,C,L,O>, U >
    {
        using type = Matrix< typename promote<S,U>::type, R, C, L, O>;
    };

    // scalar ⊕ Matrix
    template<class U, class S , std::size_t R, std::size_t C, class L, class O>
    struct promote< U, Matrix<S,R,C,L,O> >
    {
        using type = Matrix< typename promote<U,S>::type, R, C, L, O>;
    };

    //  Every BLAS routine only handles these four element types.
    template<class T> struct is_blas_scalar : std::false_type {};

    template<> struct is_blas_scalar<float>                 : std::true_type {};
    template<> struct is_blas_scalar<double>                : std::true_type {};
    template<> struct is_blas_scalar<std::complex<float>>   : std::true_type {};
    template<> struct is_blas_scalar<std::complex<double>>  : std::true_type {};

    template<class T>
    inline constexpr bool is_blas_scalar_v = is_blas_scalar<T>::value;
}

#endif //MATRIX_TRAITS_H
