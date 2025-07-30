#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  numeric/vector/operators.h
//
//  Free-function operator overloads and helpers for
//      numeric::vector::Vector<T,N>
//
//  *   vector  ±  vector      (same extent, mixed scalar types OK)
//  *   scalar  ±  vector      (any arithmetic scalar on either side)
//  *   vector  *  scalar      (scale)
//  *   scalar  *  vector
//  *   vector  /  scalar
//  *   dot( v, w )            – or v * w if you like that style
//  *   cross( a, b )          – only when static_size == 3
//  *   unary  +v, -v
//
//  All type deduction goes through numeric::scalar::promote<T,U> so
//  `Vector<float,3> + Vector<double,3>` → `Vector<double,3>`,
//  `2.0 * Vector<int,Dynamic>`          → `Vector<double,Dynamic>` etc.
//
//  This header does NOT implement expression templates; each operator
//  produces a concrete result vector immediately.
// ─────────────────────────────────────────────────────────────────────────────
#ifndef VECTOR_OPERATORS_H
#define VECTOR_OPERATORS_H

#include <array>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <execution>
#include <iostream> //replace later
#include <type_traits>
#include <utility>

#include "numeric/scalar/concepts.h"
#include "numeric/scalar/traits.h"
#include "numeric/scalar/operators.h"
#include "numeric/vector/vector.h"
#include "numeric/vector/traits.h"

namespace numeric::vector {
    namespace detail {
        template<std::size_t N1, std::size_t N2>
        struct static_extent_match
                : std::bool_constant<
        (N1 == Dynamic || N2 == Dynamic || N1 == N2 )> {};

        template<VectorLike V1, VectorLike V2>
        inline void assert_same_length(const V1& a, const V2& b)
        {
            if constexpr (static_size_v<V1> == Dynamic ||
                          static_size_v<V2> == Dynamic)
            {
                if (a.size() != b.size()) {
                    throw std::runtime_error("Vector length mismatch in operator");
                }
            }
        }

        template<class LeftVec, class Scalar>
        inline auto make_result_vector(const LeftVec& v, Scalar)
        {
            using ResScalar = typename numeric::scalar::promote<typename LeftVec::value_type, Scalar>::type;
            constexpr std::size_t N = static_size_v<LeftVec>;
            using Layout            = layout_t_t<LeftVec>;
            if constexpr (N == Dynamic){
                return Vector<ResScalar, Dynamic, Layout>(v.size());
            } else {
                return Vector<ResScalar, N, Layout>();
            }
        }

        template<class V>
        inline V make_empty_like(const V& v)
        {
            if constexpr (static_size_v<V> == Dynamic)
            return V(v.size());       // dynamic – pass runtime length
            else
            return V{};               // static – default ctor
        }

        template<std::size_t N, DenseVectorLike... Vecs, std::size_t... I>
        static constexpr auto cross_n_impl(std::index_sequence<I...>,
                                           const Vecs&... vs)
        {
            using T = typename numeric::scalar::promote<typename Vecs::value_type...>::type;
            Vector<T, N> r{};
            for (std::size_t i=0; i<N; ++i) {
                T sum{};
                // iterate over permutations via constexpr recursion
                auto accumulate = [&](auto self,
                                      std::array<std::size_t,N-1> idx,
                                      std::size_t depth,
                                      T sign)->void
                {
                    if (depth == N - 1) {
                        sum += sign * ( (... * vs[idx[Vecs{}]] ) );
                        return;
                    }
                    for (std::size_t j = 0;j<N;++j) {
                        if (j != i && std::find(idx.begin(), idx.begin()+depth,j)==idx.begin() + depth) {
                            idx[depth]=j;
                            self(self,idx,depth+1,sign*((depth%2)?-1:1));
                        }
                    }
                };
                std::array<std::size_t,N-1> idx{};
                accumulate(accumulate,idx,0,T{1});
                r[i]=sum;
            }
            return r;
        }
    }

    //===================================
    //unary operations
    //===================================

    template<VectorLike V>
    [[nodiscard]] inline auto operator+(const V& v) {return v;}

    template<DenseVectorLike V>
    [[nodiscard]] inline auto operator-(const V& v)
    {
        auto r = detail::make_result_vector(v, typename V::value_type{});
        for (std::size_t i = 0; i < v.size(); ++i){
            auto tmp = numeric::scalar::operations::checked_negate(v[i]);
            if (tmp) {
                r[i] = *tmp;
            } else {
                std::cerr << "Undefined negate behavior\n";
            }
        }
        return r;
    }

    template<SparseVectorLike V>
    [[nodiscard]] inline auto operator-(const V& v)
    {
        using numeric::scalar::operations::checked_negate;
        using value_type = typename V::value_type;

        auto r = detail::make_result_vector(v, value_type{});   // same shape/layout

        for (std::size_t i = 0; i < v.size(); ++i)
            if (auto tmp = checked_negate(v[i]); tmp) {
                r[i] = *tmp;
            } else {
                std::cerr << "Undefined negate behaviour at index " << i << '\n';
            }
        return r;
    }
    //==================================
    // Basic arithmitic
    //==================================

    template<DenseVectorLike V1, DenseVectorLike V2>
    requires detail::static_extent_match<static_size_v<V1>,
                                         static_size_v<V2>>::value
    inline auto operator+(const V1& a, const V2& b)
    {
        detail::assert_same_length(a,b);
        auto r = detail::make_result_vector(a, typename V2::value_type{});
        for (std::size_t i = 0; i < a.size(); ++i) {
            auto tmp = numeric::scalar::operations::checked_addition(a[i], b[i]);
            if (tmp) {
                r[i] = *tmp;
            } else {
                throw std::overflow_error("Checked addition overflow in operation+");
            }
        }
        return r;
    }

    template<DenseVectorLike VD, SparseVectorLike VS>
    requires detail::static_extent_match<static_size_v<VD>, static_size_v<VS>>::value
    inline auto operator+(const VD& dense, const VS& sparse)
    {
        detail::assert_same_length(dense,sparse);

        auto r = detail::make_result_vector(dense, typename VS::value_type{});
        // start with the dense values
        for (std::size_t i = 0; i < dense.size(); ++i) r[i] = dense[i];

        // add only the non‑zeros from sparse
        for (auto [i,x] : sparse)
            if (auto tmp = numeric::scalar::operations::checked_addition(r[i], x); tmp)
                r[i] = *tmp;
            else
                throw std::overflow_error("Checked addition overflow in operation+.");
        return r;
    }

    template<SparseVectorLike VS, DenseVectorLike  VD>
    inline auto operator+(const VS& sparse, const VD& dense) { return dense + sparse; }

    template<SparseVectorLike V1, SparseVectorLike V2>
    requires detail::static_extent_match<static_size_v<V1>, static_size_v<V2>>::value
    inline auto operator+(const V1& a, const V2& b)
    {
        detail::assert_same_length(a,b);

        using value_type = typename V1::value_type;
        auto r = detail::make_result_vector(a, value_type{});

        auto ia = a.begin(), ea = a.end();
        auto ib = b.begin(), eb = b.end();

        while (ia != ea || ib != eb)
        {
            if (ib == eb || (ia!=ea && ia->first < ib->first)) {
                r.set(ia->first, ia->second); ++ia;
            } else if (ia == ea || ib->first < ia->first) {
                r.set(ib->first, ib->second); ++ib;
            } else {                                       // same index
                auto tmp = numeric::scalar::operations::checked_addition(
                        ia->second, ib->second );
                if (tmp && *tmp != value_type{}) {
                    r.set(ia->first, *tmp);
                } else {
                    throw std::overflow_error("Checked addition overflow in operator+.");
                }
                ++ia; ++ib;
            }
        }
        return r;
    }


    template<DenseVectorLike V1, DenseVectorLike V2>
    requires detail::static_extent_match<static_size_v<V1>,static_size_v<V2>>::value
    inline auto operator-(const V1& a, const V2& b)
    {
        detail::assert_same_length(a,b);
        auto r = detail::make_result_vector(a, typename V2::value_type{});
        for (std::size_t i = 0; i<a.size(); ++i) {
            auto tmp = numeric::scalar::operations::checked_sub(a[i], b[i]);
            if (tmp) {
                r[i] = *tmp;
            } else {
                std::overflow_error("Checked subtraction overflow in operator-");
            }
        }
        return r;
    }


    template<DenseVectorLike  VD, SparseVectorLike VS>
    requires detail::static_extent_match<static_size_v<VD>, static_size_v<VS>>::value
    inline auto operator-(const VD& dense, const VS& sparse)
    {
        detail::assert_same_length(dense,sparse);

        auto r = detail::make_result_vector(dense, typename VS::value_type{});
        for (std::size_t i = 0; i < dense.size(); ++i) {
            r[i] = dense[i];
        }
        for (auto [i,x] : sparse) {
            if (auto tmp = numeric::scalar::operations::checked_sub(r[i], x); tmp) {
                r[i] = *tmp;
            } else {
                throw std::overflow_error("Checked subtraction overflow in operator-.");
            }
        }
        return r;
    }

    template<SparseVectorLike VS, DenseVectorLike  VD>
    inline auto operator-(const VS& sparse, const VD& dense)
    {
        // (sparse - dense)  =  -(dense - sparse)
        return -(dense - sparse);
    }

    template<SparseVectorLike V1, SparseVectorLike V2>
    requires detail::static_extent_match<static_size_v<V1>,static_size_v<V2>>::value
    inline auto operator-(const V1& a, const V2& b)
    {
        detail::assert_same_length(a,b);

        using value_type = typename V1::value_type;
        auto r = detail::make_result_vector(a, value_type{});

        auto ia = a.begin(), ea = a.end();
        auto ib = b.begin(), eb = b.end();

        while (ia != ea || ib != eb)
        {
            if (ib == eb || (ia!=ea && ia->first < ib->first)) {
                r.set(ia->first, ia->second); ++ia;
            } else if (ia == ea || ib->first < ia->first) {
                r.set(ib->first, -ib->second); ++ib;
            } else {                                       // same index
                auto tmp = numeric::scalar::operations::checked_sub(
                        ia->second, ib->second );
                if (tmp && *tmp != value_type{}) {
                    r.set(ia->first, *tmp);
                } else {
                    throw std::overflow_error("Checked subtraction overflow in operator-.");
                }
                ++ia; ++ib;
            }
        }
        return r;
    }

    template<DenseVectorLike V, numeric::scalar::NumberLike S>
    inline auto operator+(const V& v, S s)
    {
        auto r = detail::make_result_vector(v, s);
        for (std::size_t i=0; i<v.size(); ++i) {
            auto tmp = numeric::scalar::operations::checked_addition(v[i], s);
            if (tmp) {
                r[i] = *tmp;
            } else {
                throw std::overflow_error("Checked addition overflow in operator+.");
            }
        }
        return r;
    }

    template<numeric::scalar::NumberLike S, VectorLike V>
    inline auto operator+(S s, const V& v) { return v + s;}

    template<SparseVectorLike V, numeric::scalar::NumberLike S>
    inline auto operator+(const V& v, S s)
    {
        //using value_type = typename V::value_type;

        // If s==0 keep sparsity, else promote to dense result
        if (s == S{}) return v;   // nothing changes

        auto r = detail::make_result_vector(v, s);

        // start at s, then add stored deviations
        for (std::size_t i=0; i<r.size(); ++i) {
            r[i] = s;
        }
        for (auto [i,x] : v) {
            auto tmp = numeric::scalar::operations::checked_addition(x, s);
            if (!tmp) {
                throw std::overflow_error("Checked addition overflow in operator+\n");
            }
            r[i] = *tmp;
        }
        return r;
    }


    template<DenseVectorLike V, numeric::scalar::NumberLike S>
    inline auto operator-(const V& v, S s)
    {
        auto r = detail::make_result_vector(v, s);
        for (std::size_t i=0; i<v.size(); ++i) {
            auto tmp = numeric::scalar::operations::checked_sub(v[i],s);
            if (tmp) {
                r[i] = *tmp;
            } else {
                throw std::overflow_error("Checked subtraction overflow in operator-.\n");
            }
        }
        return r;
    }

    template<SparseVectorLike V, numeric::scalar::NumberLike S>
    inline auto operator-(const V& v, S s)
    {
        //using value_type = typename V::value_type;

        if (s == S{}) return v;           // no change -> keep sparsity

        auto r = detail::make_result_vector(v, s);

        for (std::size_t i=0; i<r.size(); ++i) {
            r[i] = -s;   // start at -s
        }
        for (auto [i,x] : v) {
            r[i] = x - s;
        }
        return r;
    }

    template<numeric::scalar::NumberLike S, VectorLike V>
    inline auto operator-(S s, const V& v)
    {
        // s - v = -(v - s)
        return -(v - s);
    }

    //=====================================
    // Basic multiplication and division
    //=====================================

    template<DenseVectorLike V, numeric::scalar::NumberLike S>
    inline auto operator*(const V& v, S s)
    {
        auto r = detail::make_result_vector(v,s);
        for (std::size_t i=0; i<v.size(); ++i) {
            auto tmp = numeric::scalar::operations::checked_multiply(v[i], s);
            if (tmp) {
                r[i] = *tmp;
            } else {
                throw std::overflow_error("Undefined multiplication behavior (likely overflow)\n");
            }
        }
        return r;
    }


    template<SparseVectorLike V, numeric::scalar::NumberLike S>
    inline auto operator*(const V& v, S s)
    {
        using value_type = typename V::value_type;
        auto r = detail::make_result_vector(v, s);   // empty sparse, same layout

        if (s == S{}) return r;                      // product is identically zero

        for (auto [i,x] : v)
        {
            auto y = numeric::scalar::operations::checked_multiply(x, s);
            if (!y) {
                throw std::overflow_error("Checked multiply overflow in operator*\n.");
            }
            if (y != value_type{}) {
                r.set(i, y);      // retain sparsity
            }
        }
        return r;
    }


    template<numeric::scalar::NumberLike S, VectorLike V>
    inline auto operator*(S s, const V& v) { return v * s;}

    template<DenseVectorLike V, numeric::scalar::NumberLike S>
    inline auto operator/(const V& v, S s)
    {
        auto r = detail::make_result_vector(v,s);
        for (std::size_t i=0; i<v.size(); ++i) {
            auto tmp = numeric::scalar::operations::checked_divide(v[i],s);
            if (tmp) {
                r[i] = *tmp;
            } else {
                throw std::overflow_error("Checked divide overflow in operator/\n.");
            }
        }
        return r;
    }

    template<SparseVectorLike V, numeric::scalar::NumberLike S>
    inline auto operator/(const V& v, S s)
    {
        //TODO this should check for 0/0 error in the checked division
        if (s == S{}) { throw std::domain_error("division by zero (v / s)"); };
        using value_type = typename V::value_type;

        auto r = detail::make_result_vector(v, s);

        for (auto [i,x] : v)
        {
            auto y = numeric::scalar::operations::checked_divide(x, s);
            if (!y) {
                throw std::overflow_error("Checked division overflow in operator/\n.");
            }
            if (y != value_type{}) {
                r.set(i, y);
            }
        }
        return r;
    }

    template<numeric::scalar::NumberLike S, DenseVectorLike V>
    inline auto operator/(S s, const V& v)
    {
        auto r = detail::make_result_vector(v,s);
        for (std::size_t i=0; i<v.size(); ++i) {
            auto tmp = numeric::scalar::operations::checked_divide(s, v[i]);
            if (tmp) {
                r[i] = *tmp;
            } else {
                throw std::overflow_error("Checked division overflow error in operator/.\n");
            }
        }
        return r;
    }

    template<numeric::scalar::NumberLike S, SparseVectorLike V>
    inline auto operator/(S s, const V& v)
    {
        //TODO check for scalar =0 such that 0/0 is attempted
        using ResScalar = numeric::scalar::promote_t<S,typename V::value_type>::type;
        constexpr std::size_t N = static_size_v<V>;
        if constexpr (N == Dynamic)
        {
            Vector<ResScalar, Dynamic, Dense> r(v.size());

            for (std::size_t i = 0; i < v.size(); ++i) {
                r[i] = s / static_cast<ResScalar>(v[i]);
            }
            return r;
        } else {
            Vector<ResScalar, N, Dense> r;
            for (std::size_t i = 0; i < N; ++i) {
                r[i] = s / static_cast<ResScalar>(v[i]);
            }
            return r;
        }
    }

    //===============================
    // Vector operations
    //===============================

    template<DenseVectorLike V1, DenseVectorLike V2>
    requires detail::static_extent_match<static_size_v<V1>,
                                         static_size_v<V2>>::value
    inline auto dot(const V1& a, const V2& b)
    {
        detail::assert_same_length(a,b);
        using R = typename numeric::scalar::promote<typename V1::value_type,
                                                    typename V2::value_type>::type;
        R acc = R{0};
        for (std::size_t i=0; i<a.size();++i) {
            auto prod = numeric::scalar::operations::checked_multiply(a[i], b[i]);
            if (prod) {
                auto tmp = numeric::scalar::operations::checked_addition(acc, *prod);
                if (tmp) {
                    acc = *tmp;
                } else {
                    throw std::overflow_error("Checked addition overflow in dot.\n");
                }
            } else {
                throw std::overflow_error("Checked multiplication overflow in dot.\n");
            }
        }
        return acc;
    }

    template<DenseVectorLike  VD, SparseVectorLike VS>
    requires detail::static_extent_match<static_size_v<VD>,
                                         static_size_v<VS>>::value
    inline auto dot(const VD& dense, const VS& sparse)
    {
        detail::assert_same_length(dense,sparse);

        using R = numeric::scalar::promote_t<typename VD::value_type,
                                             typename VS::value_type>;

        R acc{0};

        for (auto [i,x] : sparse) {
            auto prod = numeric::scalar::operations::checked_multiply(dense[i], x);
            if (prod) {
                auto tmp = numeric::scalar::operations::checked_addition(acc, *prod);
                if (tmp) {
                    acc = *tmp;
                } else {
                    throw std::overflow_error("Checked addition overflow in dot()\n");
                }
            } else {
                throw std::overflow_error("Checked multiplication overflow in dot()\n");
            }
        }
        return acc;
    }

    template<SparseVectorLike VS, DenseVectorLike  VD>
    inline auto dot(const VS& s, const VD& d) { return dot(d, s); }

    template<SparseVectorLike V1, SparseVectorLike V2>
    requires detail::static_extent_match<static_size_v<V1>, static_size_v<V2>>::value
    inline auto dot(const V1& a, const V2& b)
    {
        detail::assert_same_length(a,b);

        using R = numeric::scalar::promote_t<typename V1::value_type,
                                             typename V2::value_type>;

        R acc{0};

        auto ia = a.begin(), ea = a.end();
        auto ib = b.begin(), eb = b.end();

        while (ia != ea && ib != eb)
        {
            if (ia->first < ib->first) { ++ia; }
            else if (ib->first < ia->first) { ++ib; }
            else {                                       // matching index
                auto prod = numeric::scalar::operations::checked_multiply(
                        ia->second, ib->second);
                if (prod) {
                    auto tmp = numeric::scalar::operations::checked_addition(acc, *prod);
                    if (tmp) {
                        acc = *tmp;
                    } else {
                        throw std::overflow_error("Checked addition overflow in dot()\n");
                    }
                } else {
                    throw std::overflow_error("Checked multiplication overflow in dot()\n");
                }
                ++ia; ++ib;
            }
        }
        return acc;
    }

    // Dense x Dense static vectors
    template<DenseVectorLike V1, DenseVectorLike V2>
    requires (static_size_v<V1> == 3 && static_size_v<V2> == 3)
    inline auto cross(const V1& a, const V2& b)
    {
        using R = numeric::scalar::promote_t<typename V1::value_type,typename V2::value_type>;
        using Res = Vector<R, 3, Dense>;
        Res r;
        auto a1b2 = numeric::scalar::operations::checked_multiply(a[1],b[2]);
        auto a2b1 = numeric::scalar::operations::checked_multiply(a[2],b[1]);
        auto a2b0 = numeric::scalar::operations::checked_multiply(a[2],b[0]);
        auto a0b2 = numeric::scalar::operations::checked_multiply(a[0],b[2]);
        auto a0b1 = numeric::scalar::operations::checked_multiply(a[0],b[1]);
        auto a1b0 = numeric::scalar::operations::checked_multiply(a[1],b[0]);

        if (!a1b2 || !a2b1 || !a2b0 || !a0b2 || !a0b1 || !a1b0) {
            throw std::overflow_error("overflow in multiplication");
        }

        auto tmp1 = numeric::scalar::operations::checked_sub(*a1b2, *a2b1);
        if (tmp1) {
            r[0] = *tmp1;
        } else {
            throw std::overflow_error("overflow in subtraction in cross()\n");
        }

        auto tmp2 = numeric::scalar::operations::checked_sub(*a2b0, *a0b2);
        if (tmp2) {
            r[1] = *tmp2;
        } else {
            throw std::overflow_error("overflow in subtraction in cross()\n");
        }

        auto tmp3 = numeric::scalar::operations::checked_sub(*a0b1, *a1b0);
        if (tmp3) {
            r[2] = *tmp3;
        } else {
            throw std::overflow_error("overflow in subtraction in cross()\n");
        }

        return r;
    }

    // Dense x Sparse static vectors (returns Dense)
    template<DenseVectorLike  VD, SparseVectorLike VS>
    requires (static_size_v<VD> == 3 && static_size_v<VS> == 3)
    inline auto cross(const VD& d, const VS& s)
    {
        using R   = numeric::scalar::promote_t<typename VD::value_type,
                                               typename VS::value_type>;
        Vector<R, 3, Dense> Res{};

        // materialise sparse into three scalars
        R sv[3]{};
        for (auto [i,x] : s) {
            sv[i] = x;                            // implicit zeros filled already
        }

        Res[0] = sv[0];
        Res[1] = sv[1];
        Res[2] = sv[2];                         // cheap stack object
        return cross(d, Res);                  // reuse dense×dense kernel
    }

    // Sparse x Dense static vectors (returns Dense)
    template<SparseVectorLike VS, DenseVectorLike  VD>
    requires (static_size_v<VD> == 3 && static_size_v<VS> == 3)
    inline auto cross(const VS& s, const VD& d)
    {
        using R   = numeric::scalar::promote_t<typename VS::value_type,
                                               typename VD::value_type>;
        Vector<R, 3, Dense> Res{};

        R sv[3]{};
        for (auto [i,x] : s) {
            sv[i] = x;
        }

        Res[0] = sv[0];
        Res[1] = sv[1];
        Res[2] = sv[2];
        return cross(Res, d);                  // now order preserved
    }

    // Sparse x Sparse static vectors (returns Sparse)
    template<SparseVectorLike V1, SparseVectorLike V2>
    requires(static_size_v<V1> == 3 && static_size_v<V2> == 3)
    inline auto cross(const V1& a, const V2& b)
    {
        using R = numeric::scalar::promote_t<typename V1::value_type,
                                             typename V2::value_type>;
        Vector<R, 3, Sparse> Res{};
        R c0{}, c1{}, c2{};

        for (auto [i, ai] : a) {
            if (ai == R{}) {
                continue;
            }
            switch (i) {
                case 0: {
                    R bz = b[2];
                    if (bz != R{}) {
                        auto prod = numeric::scalar::operations::checked_multiply(ai, bz);
                        if (prod) {
                            auto acc = numeric::scalar::operations::checked_sub(c1, *prod);
                            if (acc) {
                                c1 = *acc;
                            } else {
                                throw std::overflow_error("Checked subtraction overflow in cross()\n");
                            }
                        } else {
                            throw std::overflow_error("Checked multiplication overflow in cross()\n");
                        }
                    }

                    R by = b[1];
                    if (by != R{}) {
                        auto prod = numeric::scalar::operations::checked_multiply(ai, by);
                        if (prod) {
                            auto acc = numeric::scalar::operations::checked_addition(c2,*prod);
                            if (acc) {
                                c2 = *acc;
                            } else {
                                throw std::overflow_error("Checked addition overflow in cross()\n");
                            }
                        } else {
                            throw std::overflow_error("Checked multiplication overflow in cross()\n");
                        }
                    }
                    break;
                }
                case 1: {
                    R bz = b[2];
                    if (bz != R{}) {
                        auto prod = numeric::scalar::operations::checked_multiply(ai, bz);
                        if (prod) {
                            auto acc = numeric::scalar::operations::checked_addition(c0,*prod);
                            if (acc) {
                                c0 = *acc;
                            } else {
                                throw std::overflow_error("Checked addition overflow in cross()\n");
                            }
                        } else {
                            throw std::overflow_error("Checked multiplication overflow in cross()\n");
                        }
                    }

                    R bx = b[0];
                    if (bx != R{}) {
                        auto prod = numeric::scalar::operations::checked_multiply(ai, bx);
                        if (prod) {
                            auto acc = numeric::scalar::operations::checked_sub(c2,*prod);
                            if (acc) {
                                c2 = *acc;
                            } else {
                                throw std::overflow_error("Checked addition overflow in cross()\n");
                            }
                        } else {
                            throw std::overflow_error("Checked multiplication overflow in cross()\n");
                        }
                    }
                    break;
                }
                case 2: {
                    R by = b[1];
                    if (by != R{}) {
                        auto prod = numeric::scalar::operations::checked_multiply(ai, by);
                        if (prod) {
                            auto acc = numeric::scalar::operations::checked_sub(c0,*prod);
                            if (acc) {
                                c0 = *acc;
                            } else {
                                throw std::overflow_error("Checked addition overflow in cross()\n");
                            }
                        } else {
                            throw std::overflow_error("Checked multiplication overflow in cross()\n");
                        }
                    }

                    R bx = b[0];
                    if (bx != R{}) {
                        auto prod = numeric::scalar::operations::checked_multiply(ai, bx);
                        if (prod) {
                            auto acc = numeric::scalar::operations::checked_addition(c1,*prod);
                            if (acc) {
                                c1 = *acc;
                            } else {
                                throw std::overflow_error("Checked addition overflow in cross()\n");
                            }
                        } else {
                            throw std::overflow_error("Checked multiplication overflow in cross()\n");
                        }
                    }
                    break;
                }
                default:
                    throw std::out_of_range("cross(): 3x3, index must be 0,1,2");
            }
        }
        if (c0 != R{}) {
            Res.set(0,c0);
        }
        if (c1 != R{}) {
            Res.set(1,c1);
        }
        if (c2 != R{}) {
            Res.set(2,c2);
        }
        return Res;
    }

    //TODO extend this to all vector types
    template<VectorLike V1, VectorLike... Vrest>
    requires (sizeof...(Vrest)+1 == static_size_v<V1>)
    inline auto cross(const V1& v1, const Vrest&... vs)
    {
        constexpr std::size_t N = static_size_v<V1>;
        static_assert(N != Dynamic, "General cross requires static extent");
        static_assert((std::is_same_v<std::decay_t<V1>,std::decay_t<Vrest>> && ...),
                      "All operands must have the same Vector type/extent");
        detail::assert_same_length(v1,vs...);
        return detail::cross_n_impl<N>(std::make_index_sequence<N>{}, vs...);
    }
    //====================================
    // Compund assignment
    //====================================

    template<DenseVectorLike V1, DenseVectorLike V2>
    requires detail::static_extent_match<static_size_v<V1>,
                                         static_size_v<V2>>::value
    inline V1& operator+=(V1& a, const V2& b)
    {
        detail::assert_same_length(a,b);
        using T = typename V1::value_type;
        for (std::size_t i=0; i<a.size(); ++i) {
            auto tmp = numeric::scalar::operations::checked_addition(a[i], b[i]);
            if (tmp) {
                a[i] = static_cast<T>(*tmp);
            } else {
                throw std::overflow_error("Checked addition overflow in operator+=\n");
            }
        }
        return a;
    }

    template<DenseVectorLike VD, SparseVectorLike VS>
    requires detail::static_extent_match<static_size_v<VD>,static_size_v<VS>>::value
    inline VD& operator+=(VD& a, const VS& b)
    {
        detail::assert_same_length(a,b);
        using T = typename VD::value_type;
        for (auto [i,x] : b) {
            auto tmp = numeric::scalar::operations::checked_addition(a[i], x);
            if (tmp) {
                a[i] = static_cast<T>(*tmp);
            } else {
                throw std::overflow_error("Checked addition overflow in operator+=\n");
            }
        }
        return a;
    }

    template<SparseVectorLike VS, DenseVectorLike VD>
    requires detail::static_extent_match<static_size_v<VD>,static_size_v<VS>>::value
    inline VS& operator+=(VS& a, const VD& b)
    {
        detail::assert_same_length(a,b);
        using T = typename VS::value_type;
        for (std::size_t i=0;i<b.size();++i){
            auto tmp = numeric::scalar::operations::checked_addition(a[i],b[i]);
            if (tmp) {
                T val = static_cast<T>(*tmp);
                if (val != T{}) {
                    a.set(i, val);
                }
            } else {
                throw std::overflow_error("Checked addition overflow in operator+=\n");
            }
        }
        return a;
    }

    // ---------------- sparse += sparse ----------------
    template<SparseVectorLike V1, SparseVectorLike V2>
    requires detail::static_extent_match<static_size_v<V1>, static_size_v<V2>>::value
    inline V1& operator+=(V1& a, const V2& b)
    {
        detail::assert_same_length(a,b);
        using T = typename V1::value_type;
        auto tmp = detail::make_empty_like(a);

        auto ia=a.begin(), ea=a.end();
        auto ib=b.begin(), eb=b.end();
        while (ia!=ea || ib!=eb) {
            if (ib==eb || (ia!=ea && ia->first < ib->first)) {
                tmp.set(ia->first, ia->second);
                ++ia;
            } else if (ia==ea || ib->first < ia->first) {
                tmp.set(ib->first, static_cast<T>(ib->second));
                ++ib;
            } else {
                auto val = numeric::scalar::operations::checked_addition(ia->second, ib->second);
                if (val) {
                    auto acc = static_cast<T>(*val);
                    if (acc != T{}) {
                        tmp.set(ia->first, acc);
                    }
                } else {
                    throw std::overflow_error("Checked addition overflow in operator+=\n");
                }
                ++ia; ++ib;
            }
        }
        swap(a,tmp);
        return a;
    }

    //TODO continue to add sparse aware operations for vectors
    template<DenseVectorLike V1, DenseVectorLike V2>
    requires detail::static_extent_match<static_size_v<V1>, static_size_v<V2>>::value
    inline V1& operator-=(V1& a, const V2& b)
    {
        detail::assert_same_length(a,b);
        using T = typename V1::value_type;
        for (std::size_t i=0; i<a.size(); ++i) {
            auto tmp = numeric::scalar::operations::checked_sub(a[i], b[i]);
            if (tmp) {
                a[i] = static_cast<T>(*tmp);
            } else {
                throw std::overflow_error("Checked subtraction overflow in operator-=");
            }
        }
        return a;
    }

    // dense -= sparse compound subtraction (return dense)
    template<DenseVectorLike VD, SparseVectorLike VS>
    requires detail::static_extent_match<static_size_v<VD>,static_size_v<VS>>::value
    inline VD& operator-=(VD& a, const VS& b)
    {
        detail::assert_same_length(a,b);
        using T = typename VD::value_type;

        for (auto [i,x] : b) {
            auto tmp = numeric::scalar::operations::checked_sub(a[i], x);
            if (tmp) {
                a[i] = static_cast<T>(*tmp);
            } else {
                throw std::overflow_error("Checked subtraction overflow in operator-=");
            }
        }
        return a;
    }

    // sparse -= dense (returns sparse)
    template<SparseVectorLike VS, DenseVectorLike VD>
    requires detail::static_extent_match<static_size_v<VS>,static_size_v<VD>>::value
    inline VS& operator-=(VS& a, const VD& b)
    {
        detail::assert_same_length(a,b);
        using T = typename VS::value_type;

        for (std::size_t i=0; i<b.size(); ++i) {
            auto val = numeric::scalar::operations::checked_sub(a[i],b[i]);
            if (val) {
                T tmp = static_cast<T>(*val);
                if (tmp != T{}) {
                    a.set(i, tmp);
                }
            } else {
                throw std::overflow_error("Checked subtraction overflow in operator-=");
            }
        }
        return a;
    }

    // sparse -= sparse (returns sparse)
    template<SparseVectorLike V1, SparseVectorLike V2>
    requires detail::static_extent_match<static_size_v<V1>,static_size_v<V2>>::value
    inline V1& operator-=(V1& a, const V2& b)
    {
        detail::assert_same_length(a,b);
        using T = typename V1::value_type;
        auto tmp = detail::make_empty_like(a);

        auto ia=a.begin(), ea=a.end();
        auto ib=b.begin(), eb=b.end();
        while (ia!=ea || ib!=eb) {
            if (ib==eb || (ia != ea && ia->first < ib->first)) {
                tmp.set(ia->first, ia->second);
                ++ia;
            } else if (ia==ea || ib->first < ia->first) {
                tmp.set(ib->first,static_cast<T>(ib->second));
                ++ib;
            } else {
                auto val = numeric::scalar::operations::checked_sub(ia->second,ib->second);
                if (val) {
                    T acc = static_cast<T>(*val);
                    if (acc != T{}) {
                        tmp.set(ia->first, acc);
                    }
                } else {
                    throw std::overflow_error("Checked subtraction overflow in operator-=");
                }
                ++ia; ++ib;
            }
        }
        swap(a,tmp);
        return a;
    }


    // dense *= dense (returns dense)
    template<DenseVectorLike V1, DenseVectorLike V2>
    requires detail::static_extent_match<static_size_v<V1>, static_size_v<V2>>::value
    inline V1& operator*=(V1& a, const V2& b)
    {
        detail::assert_same_length(a,b);
        for (std::size_t i=0; i<a.size(); ++i) {
            auto tmp = numeric::scalar::operations::checked_multiply(a[i], b[i]);
            if (tmp) {
                a[i] = *tmp;
            } else {
                throw std::overflow_error("Checked multiplication overflow in operator*=");
            }
        }
        return a;
    }

    // dense *= sparse (returns sparse)
    template<DenseVectorLike VD, SparseVectorLike VS>
    requires detail::static_extent_match<static_size_v<VD>, static_size_v<VS>>::value
    inline VD& operator*=(VD& a, const VS& b)
    {
        detail::assert_same_length(a,b);
        using T = typename VD::value_type;
        for (std::size_t i=0; i<a.size(); ++i) {
            auto tmp = numeric::scalar::operations::checked_multiply(a[i],b[i]);
            if (tmp) {
                T prod = static_cast<T>(*tmp);
                a[i] = prod;
            } else {
                throw std::overflow_error("Checked multiplication overflow in operator*=");
            }
        }
        return a;
    }

    // sparse *= dense (return sparse)
    template<SparseVectorLike VS, DenseVectorLike VD>
    requires detail::static_extent_match<static_size_v<VS>, static_size_v<VD>>::value
    inline VS& operator*=(VS& a, const VD& b)
    {
        detail::assert_same_length(a,b);
        using T = typename VS::value_type;
        auto tmp = detail::make_empty_like(a);

        for (auto [i,x] : a) {
            auto val = numeric::scalar::operations::checked_multiply(x,b[i]);
            if (val){
                T prod = static_cast<T>(*val);
                if (prod != T{}) {
                    tmp.set(i, prod);
                }
            } else {
                throw std::overflow_error("Checked multiplication overflow in operator*=");
            }
        }
        swap(a,tmp);
        return a;
    }

    // sparse *= sparse (return sparse)
    template<SparseVectorLike V1, SparseVectorLike V2>
    requires detail::static_extent_match<static_size_v<V1>, static_size_v<V2>>::value
    inline V1& operator*=(V1& a, const V2& b)
    {
        detail::assert_same_length(a,b);
        using T = typename V1::value_type;
        auto tmp = detail::make_empty_like(a);

        auto ia=a.begin(), ea=a.end();
        auto ib=b.begin(), eb=b.end();
        while (ia!=ea && ib!=eb) {
            if (ia->first < ib->first) {
                ++ia;
            } else if (ib->first < ia->first) {
                ++ib;
            } else {
                auto val = numeric::scalar::operations::checked_multiply(ia->second, ib->second);
                tmp.set(ia->first, static_cast<T>(val));
                ++ia; ++ib;
            }
        }
        swap(a,tmp);
        return a;
    }

    // dense /= dense (returns dense)
    template<DenseVectorLike V1, DenseVectorLike V2>
    requires detail::static_extent_match<static_size_v<V1>, static_size_v<V2>>::value
    inline V1& operator/=(V1& a, const V2& b)
    {
        using T = typename V1::value_type;
        detail::assert_same_length(a,b);
        for (std::size_t i=0; i<a.size(); ++i) {
            auto tmp = numeric::scalar::operations::checked_divide(a[i], b[i]);
            if (tmp) {
                a[i] = static_cast<T>(*tmp);
            } else {
                throw std::overflow_error("Checked division overflow in operator/=");
            }
        }
        return a;
    }

    // dense /= sparse (returns dense)
    template<DenseVectorLike VD, SparseVectorLike VS>
    requires detail::static_extent_match<static_size_v<VD>, static_size_v<VS>>::value
    inline VD& operator/=(VS& a, const VD& b)
    {
        detail::assert_same_length(a,b);
        using T = typename VS::value_type;
        for (std::size_t i=0; i<a.size(); ++i) {
            auto tmp = numeric::scalar::operations::checked_divide(a[i], b[i]);
            if (tmp) {
                if constexpr (numeric::scalar::IntegerLike<T>) {
                    if (std::isfinite(*tmp)) {
                        a[i] = static_cast<T>(*tmp);
                    } else {
                        throw std::overflow_error("Conversion of infinite to finite integral in operator/=");
                    }
                } else {
                    a[i] = static_cast<T>(*tmp);
                }
            } else {
                throw std::overflow_error("Checked division overflow in operator/=");
            }
        }
        return a;
    }


    // sparse /= dense (return sparse)
    template<SparseVectorLike VS, DenseVectorLike VD>
    requires detail::static_extent_match<static_size_v<VD>, static_size_v<VS>>::value
    inline VS& operator/=(VS& a, const VD& b)
    {
        detail::assert_same_length(a,b);
        using T = typename VS::value_type;
        auto tmp = detail::make_empty_like(a);

        for (std::size_t i=0; i<b.size(); ++i) {
            auto val = numeric::scalar::operations::checked_divide(a[i], b[i]);
            if (val) {
                if constexpr (numeric::scalar::IntegerLike<T>) {
                    if (std::isfinite(*val)) {
                        if (static_cast<T>(*val) != T{}) {
                            tmp.set(i,static_cast<T>(*val));
                        }
                    } else {
                        throw std::overflow_error("Conversion of infinite to finite integral in operator/=");
                    }
                } else {
                    if (static_cast<T>(*val) != T{}) {
                        tmp.set(i,static_cast<T>(*val));
                    }
                }
            } else {
                throw std::overflow_error("Checked division overflow in operator/=");
            }
        }
        swap(a,tmp);
        return a;
    }

    // sparse /= sparse (returns sparse) [ignores 0/0 division only ∞/∞ returns an error]
    template<SparseVectorLike V1, SparseVectorLike V2>
    requires detail::static_extent_match<static_size_v<V1>, static_size_v<V2>>::value
    inline V1& operator/=(V1& a, const V2& b)
    {
        detail::assert_same_length(a,b);
        using T = typename V1::value_type;
        auto tmp = detail::make_empty_like(a);

        auto ia=a.begin(), ea=a.end();
        auto ib=b.begin(), eb=b.end();
        while (ia != ea && ib != eb) {
            if (ia->first < ib->first) {
                ++ia;
            } else if (ib->first < ia->first) {
                ++ib;
            } else {
                if (!std::isfinite(ia->second) && !std::isfinite(ib->second)) {
                    throw std::overflow_error("inf/inf invalid operation encountered in operator/=");
                }
                auto val = numeric::scalar::operations::checked_divide(ia->second,ib->second);
                if (val) {
                    if (static_cast<T>(*val) != T{}){
                        tmp.set(ia->first, static_cast<T>(*val));
                    }
                } else {
                    throw std::overflow_error("Checked division overflow in operator/=");
                }
            }
        }
        swap(a,tmp);
        return a;
    }


    template<DenseVectorLike V, numeric::scalar::NumberLike S>
    inline V& operator*=(V& v, S s)
    {
        using T = typename V::value_type;
        for(std::size_t i=0; i<v.size(); ++i) {
            auto tmp = numeric::scalar::operations::checked_multiply(v[i],s);
            if (tmp) {
                v[i] = static_cast<T>(*tmp);
            } else {
                throw std::overflow_error("Checked multiplication overflow in operator*=");
            }
        }
        return v;
    }

    template<SparseVectorLike V, numeric::scalar::NumberLike S>
    inline V& operator*=(V& v, S s)
    {
        using T = typename V::value_type;
        auto tmp = detail::make_empty_like(v);
        if (s == S{}) {
            swap(v,tmp);
            return v;
        }
        for (auto [i,x] : v) {
            auto val = numeric::scalar::operations::checked_multiply(x, s);
            if (val) {
                tmp.set(i, static_cast<T>(*val));
            } else {
                throw std::overflow_error("Checked multiplication overflow in operator*=");
            }
        }
        swap(v,tmp);
        return v;
    }


    // dense /= scalar
    template<DenseVectorLike V, numeric::scalar::NumberLike S>
    inline V& operator/=(V& v, S s)
    {
        using T = typename V::value_type;
        for(std::size_t i=0; i<v.size(); ++i) {
            auto tmp = numeric::scalar::operations::checked_divide(v[i],s);
            if (tmp) {
                v[i] = static_cast<T>(*tmp);
            } else {
                throw std::overflow_error("Checked division overflow in operator/=");
            }
        }
        return v;
    }

    // sparse /= scalar
    template<SparseVectorLike V, numeric::scalar::NumberLike S>
    inline V& operator/=(V& v, S s)
    {
        using T = typename V::value_type;

        if (!std::isfinite(s)) {
            auto tmp = detail::make_empty_like(v);
            swap(v,tmp);
            return v;
        }
        for (auto [i,x] : v) {
            auto tmp = numeric::scalar::operations::checked_divide(x,s);
            if (tmp) {
                v.set(i, static_cast<T>(*tmp));
            }
        }
        return v;
    }
    //===================================
    // Element-wise multiply & divide
    //===================================

    // dense * dense (return dense)
    template<DenseVectorLike V1, DenseVectorLike V2>
    requires detail::static_extent_match<static_size_v<V1>,static_size_v<V2>>::value
    inline auto operator*(const V1& a, const V2& b)
    {
        detail::assert_same_length(a,b);
        auto r = detail::make_result_vector(a, typename V2::value_type{});
        for (std::size_t i=0; i<a.size(); ++i) {
            auto tmp = numeric::scalar::operations::checked_multiply(a[i], b[i]);
            if (tmp) {
                r[i] = *tmp;
            } else {
                throw std::overflow_error("Checked multiplication overflow in operator*");
            }
        }
        return r;
    }

    template<DenseVectorLike VD, SparseVectorLike VS>
    requires detail::static_extent_match<static_size_v<VD>,static_size_v<VS>>::value
    inline auto operator*(const VD& a, const VS& b)
    {
        detail::assert_same_length(a,b);
        auto r = detail::make_result_vector(b,typename VD::value_type{});

        for (auto [i,x] : b) {
            auto tmp = numeric::scalar::operations::checked_multiply(a[i], x);
            if (tmp) {
                r.set(i, *tmp);
            } else {
                throw std::overflow_error("Checked multiplication overflow in operator*");
            }
        }
        return r;
    }

    template<SparseVectorLike VS, DenseVectorLike VD>
    requires detail::static_extent_match<static_size_v<VS>,static_size_v<VD>>::value
    inline auto operator*(const VS& a, const VD& b)
    {
        detail::assert_same_length(a,b);
        return b * a;
    }

    template<SparseVectorLike V1, SparseVectorLike V2>
    requires detail::static_extent_match<static_size_v<V1>,static_size_v<V2>>::value
    inline auto operator*(const V1& a, const V2& b)
    {
        detail::assert_same_length(a,b);
        auto r = detail::make_result_vector(a, typename V2::value_type{});

        auto ia = a.begin(), ea = a.end();
        auto ib = b.begin(), eb = b.end();
        while (ia != ea && ib != eb) {
            if (ia->first < ib->first) { ++ia; }
            else if (ib->first < ia->first) { ++ib; }
            else {
                auto tmp = numeric::scalar::operations::checked_multiply(ia->second,ib->second);
                if (tmp) {
                    r.set(ia->first, *tmp);
                } else {
                    throw std::overflow_error("Checked multiplication overflow in operator*");
                }
            }
        }
        return r;
    }

    template<DenseVectorLike V1, DenseVectorLike V2>
    requires detail::static_extent_match<static_size_v<V1>,static_size_v<V2>>::value
    inline auto operator/(const V1& a, const V2& b)
    {
        detail::assert_same_length(a,b);
        auto r = detail::make_result_vector(a, typename V2::value_type{});
        for (std::size_t i=0;i<a.size(); ++i) {
            auto tmp = numeric::scalar::operations::checked_divide(a[i], b[i]);
            if (tmp) {
                r[i] = *tmp;
            } else {
                throw std::overflow_error("Checked division overflow in operator/");
            }
        }
        return r;
    }

    template<DenseVectorLike VD, SparseVectorLike VS>
    requires detail::static_extent_match<static_size_v<VD>,static_size_v<VS>>::value
    inline auto operator/(const VD& a, const VS& b)
    {
        detail::assert_same_length(a,b);
        auto r = detail::make_result_vector(a, typename VS::value_type{});
        for (std::size_t i=0; i<a.size(); ++i) {
            auto tmp = numeric::scalar::operations::checked_divide(a[i], b[i]);
            if (tmp) {
                r[i] = *tmp;
            } else {
                throw std::overflow_error("Checked division overflow in operator/");
            }
        }
        return r;
    }

    // tentatively ignoring 0/0 division in sparse vectors
    template<SparseVectorLike V1, SparseVectorLike V2>
    requires detail::static_extent_match<static_size_v<V1>,static_size_v<V2>>::value
    inline auto operator/(const V1& a, const V2& b)
    {
        detail::assert_same_length(a,b);
        auto r = detail::make_result_vector(a, typename V2::value_type{});
        using T = numeric::scalar::promote_t<typename V1::value_type, typename V2::value_type>;

        auto ia = a.begin(), ea = a.end();
        auto ib = b.begin(), eb = b.end();
        while (ia != ea && ib != eb) {
            if (ia->first < ib->first){
                auto tmp = numeric::scalar::operations::checked_divide(ia->second, T{});
                if (tmp) {
                    r.set(ia->first, *tmp);
                } else {
                    throw std::overflow_error("Checked division overflow in operator/");
                }
                ++ia;
            } else if (ib->first < ia->first) {
                ++ib;
            } else {
                auto tmp = numeric::scalar::operations::checked_divide(ia->second, ib->second);
                if (tmp) {
                    r.set(ia->first, *tmp);
                } else {
                    throw std::overflow_error("Checked division overflow in operator/");
                }
            }
        }
        return r;
    }

    //================================
    // Reductions
    //================================
    template<DenseVectorLike V>
    inline auto sum(const V& v)
    {
        using T = typename V::value_type;
        if (v.empty()){
            throw std::out_of_range("Empty dense vector; no finite sum.");
        }
        T acc{};

        for (const auto& x : v){
            auto tmp = numeric::scalar::operations::checked_addition(acc, x);
            if (tmp) {
                acc = *tmp;
            } else {
                throw std::overflow_error("Checked addition overflow in dense sum");
            }
        }
        return acc;
    }

    template<SparseVectorLike V>
    inline auto sum(const V& v)
    {
        using T = typename V::value_type;
        if (v.empty()) {
            throw std::out_of_range("Empty sparse vector; no finite sum()");
        }
        T acc{};

        for (const auto& [i,x] : v){
            auto tmp = numeric::scalar::operations::checked_addition(acc,x);
            if (tmp) {
                acc = *tmp;
            } else {
                throw std::overflow_error("Checked addition overflow in sparse sum()");
            }
        }
    }

    template<DenseVectorLike V>
    inline auto abs_sum(const V& v)
    {
        using T = typename V::value_type;
        if (v.empty()){
            throw std::out_of_range("Empty dense vector; no finite sum()");
        }
        T acc{};

        for (const auto& x : v){
            auto abs_x = numeric::scalar::operations::checked_abs(x);
            if (!abs_x) {
                throw std::overflow_error("Checked abs overflow in dense abs_sum().");
            }
            auto tmp = numeric::scalar::operations::checked_addition(acc, *abs_x);
            if (tmp) {
                acc = *tmp;
            } else {
                throw std::overflow_error("Checked addition overflow in dense abs_sum()");
            }
        }
        return acc;
    }

    template<SparseVectorLike V>
    inline auto abs_sum(const V& v)
    {
        using T = typename V::value_type;
        if (v.empty()) {
            throw std::out_of_range("Empty sparse vector; no finite abs_sum()");
        }
        T acc{};

        for (const auto& [i, x]: v) {
            auto abs_x = numeric::scalar::operations::checked_abs(x);
            if (!abs_x) {
                throw std::overflow_error("Checked abs overflow in sparse abs_sum().");
            }
            auto tmp = numeric::scalar::operations::checked_addition(acc, *abs_x);
            if (tmp) {
                acc = *tmp;
            } else {
                throw std::overflow_error("Checked addition overflow in dense abs_sum()");
            }
        }
        return acc;
    }


    template<DenseVectorLike V>
    inline auto min(const V& v)
    {
        using T = typename V::value_type;

        if (v.empty()){
            throw std::out_of_range("Empty dense vector; no finite min()");
        }

        T current = v[0];
        for (std::size_t i=1;i<v.size(); ++i){
            auto tmp = numeric::scalar::operations::checked_isless(v[i], current);
            if (tmp) {
                if (*tmp) {
                    current = v[i];
                }
            } else {
                throw std::overflow_error("Comparison overflow < in min()");
            }
        }
        return current;
    }

    template<SparseVectorLike V>
    inline auto min(const V& v)
    {
        using T = typename V::value_type;
        if (v.empty()){
            throw std::out_of_range("Empty sparse vector; no finite min()");
        }
        auto iv = v.begin(), ev = v.end();
        T current{}; //assume that the minimum value is zero by default
        while (iv != ev) {
            auto tmp = numeric::scalar::operations::checked_isless(iv->second, current);
            if (tmp) {
                if (*tmp) {
                    current = iv->second;
                }
            } else {
                throw std::overflow_error("Comparison overflow < in min()");
            }
            ++iv;
        }
        return current;
    }

    template<DenseVectorLike V>
    inline auto abs_min(const V& v)
    {
        using T = typename V::value_type;

        if (v.empty()){
            throw std::out_of_range("Empty dense vector; no finite abs_min()");
        }

        T current{};
        auto abs_val = numeric::scalar::operations::checked_abs(v[0]);
        if (abs_val) {
            current = *abs_val;
        } else {
            throw std::overflow_error("Checked abs overflow in operation dense abs_min()");
        }
        for (std::size_t i=1;i<v.size(); ++i){
            auto abs_x = numeric::scalar::operations::checked_abs(v[i]);
            if (!abs_x) {
                throw std::overflow_error("Checked abs overflow in operation dense abs_min()");
            }
            auto tmp = numeric::scalar::operations::checked_isless(*abs_x, current);
            if (tmp) {
                if (*tmp) {
                    current = v[i];
                }
            } else {
                throw std::overflow_error("Comparison overflow < in abs_min()");
            }
        }
        return current;
    }

    template<SparseVectorLike V>
    inline auto abs_min(const V& v)
    {
        using T = typename V::value_type;
        if (v.empty()){
            throw std::out_of_range("Empty dense vector; no finite abs_min()");
        }

        auto iv = v.begin(), ev = v.end();
        T current{};
        while (iv != ev) {
            auto abs_x = numeric::scalar::operations::checked_abs(iv->second);
            if (!abs_x) {
                throw std::overflow_error("Checked abs overflow in operation sparse abs_min()");
            }
            auto tmp = numeric::scalar::operations::checked_isless(*abs_x, current);
            if (tmp) {
                if (*tmp) {
                    current = iv->second;
                }
            } else {
                throw std::overflow_error("Comparison overflow < in abs_min()");
            }
            ++iv;
        }
        return current;
    }

    template<DenseVectorLike V>
    inline auto max(const V& v)
    {
        using T = typename V::value_type;

        if (v.empty()) {
            throw std::out_of_range("Empty dense vector, no finite maximum.");
        }

        T current = v[0];
        for (std::size_t i=1;i<v.size();++i) {
            auto tmp = numeric::scalar::operations::checked_isgreater(v[i],current);
            if (tmp) {
                if (*tmp) {
                    current = v[i];
                }
            } else {
                throw std::overflow_error("Checked comparison > overflow in dense max()");
            }
        }
        return current;
    }

    template<SparseVectorLike V>
    inline auto max(const V& v)
    {
        using T = typename V::value_type;

        if (v.empty()) {
            throw std::out_of_range("Empty dense vector, no finite maximum.");
        }

        auto iv = v.begin(), ev = v.end();
        T current{};
        while (iv != ev) {
            auto tmp = numeric::scalar::operations::checked_isgreater(iv->second,current);
            if (tmp) {
                if (*tmp) {
                    current = iv->second;
                }
            } else {
                throw std::overflow_error("Checked comparison > overflow in sparse max()");
            }
            ++iv;
        }
        return current;
    }

    template<DenseVectorLike V>
    inline auto abs_max(const V& v)
    {
        using T = typename V::value_type;

        if (v.empty()){
            throw std::out_of_range("Empty dense vector; no finite absolute maximum");
        }

        T current{};
        auto abs_val = numeric::scalar::operations::checked_abs(v[0]);
        if (abs_val) {
            current = *abs_val;
        } else {
            throw std::overflow_error("Checked abs overflow in dense abs_max()");
        }
        for (std::size_t i=1;i<v.size(); ++i){
            auto abs_x = numeric::scalar::operations::checked_abs(v[i]);
            if (!abs_x) {
                throw std::overflow_error("Checked abs overflow in dense abs_max()");
            }
            auto tmp = numeric::scalar::operations::checked_isgreater(*abs_x, current);
            if (tmp) {
                if (*tmp) {
                    current = v[i];
                }
            } else {
                throw std::overflow_error("Checked comparison > overflow in dense abs_max()");
            }
        }
        return current;
    }

    template<SparseVectorLike V>
    inline auto abs_max(const V& v)
    {
        using T = typename V::value_type;

        if (v.empty()){
            throw std::out_of_range("Empty sparse vector; no finite absolute maximum");
        }

        auto iv=v.begin(), ev=v.end();
        T current{};
        while (iv != ev){
            auto abs_x = numeric::scalar::operations::checked_abs(iv->second);
            if (!abs_x) {
                throw std::overflow_error("Checked abs overflow in sparse abs_max()");
            }
            auto tmp = numeric::scalar::operations::checked_isgreater(*abs_x, current);
            if (tmp) {
                if (*tmp) {
                    current = iv->second;
                }
            } else {
                throw std::overflow_error("Checked comparison > overflow in sparse abs_max()");
            }
            ++iv;
        }
        return current;
    }

    template<VectorLike V>
    inline auto norm2(const V& v) { return dot(v,v);}

    template<VectorLike V>
    inline auto norm(const V& v)
    {
        auto tmp = numeric::scalar::operations::checked_sqrt(norm2(v));
        if (tmp) {
            return *tmp;
        } else {
            throw std::overflow_error("Checked sqrt overflow in norm()");
        }
    }

    template <DenseVectorLike V>
    inline auto mean(const V& v)
    {
        using T = typename V::value_type;
        T acc{0};

        if (v.empty()){
            throw std::out_of_range("Empty sparse vector; no finite absolute maximum");
        }

        for (std::size_t i=0;i<v.size();++i) {
            auto tmp = numeric::scalar::operations::checked_addition(acc,v[i]);
            if (tmp) {
                acc = *tmp;
            } else {
                throw std::overflow_error("Checked addition overflow in dense mean()");
            }
        }
        auto tmp = numeric::scalar::operations::checked_divide(acc, v.size());

        if (tmp) {
            return *tmp;
        } else {
            throw std::overflow_error("Checked division overflow in dense mean()");
        }
    }

    template<SparseVectorLike V>
    inline auto mean(const V& v)
    {
        using T = typename V::value_type;
        T acc{0};
        if (v.empty()){
            throw std::out_of_range("Empty sparse vector; no finite absolute maximum");
        }

        for (const auto& [i,x] : v) {
            auto tmp = numeric::scalar::operations::checked_addition(acc,x);
            if (tmp) {
                acc = *tmp;
            } else {
                throw std::overflow_error("Checked addition overflow for sparse mean()");
            }
        }
        auto tmp = numeric::scalar::operations::checked_divide(acc,v.size());
        if (tmp) {
            return *tmp;
        } else {
            throw std::overflow_error("Checked division overflow in sparse mean()");
        }
    }

    template <DenseVectorLike V>
    inline auto prod(const V& v)
    {
        using T = typename V::value_type;

        if (v.empty()) {
            throw std::out_of_range("Empty dense vector, no finite product.");
        }

        T acc = v[0];

        for (std::size_t i=1; i<v.size(); ++i){
            auto tmp = numeric::scalar::operations::checked_multiply(acc, v[i]);
            if (tmp) {
                acc = *tmp;
            } else {
                throw std::overflow_error("Checked multiplication overflow in prod().");
            }
        }
        return acc;
    }

    // Note that the prod for a sparse vector is only considering the non-zero entries (may need to change)
    template<SparseVectorLike V>
    inline auto prod(const V& v)
    {
        using T = typename V::value_type;
        if (v.empty()) {
            throw std::out_of_range("Empty sparse vector, no finite product.");
        }

        T acc{1};
        for (const auto& [i,x] : v) {
            auto tmp = numeric::scalar::operations::checked_multiply(acc,x);
            if (tmp) {
                acc = *tmp;
            } else {
                throw std::overflow_error("Checked multiplication overflow in sparse prod()");
            }
        }
        return acc;
    }


    template<DenseVectorLike V>
    inline auto sum_of_squares(const V& v)
    {
        using T = typename V::value_type;

        if (v.empty()) {
            throw std::out_of_range("Empty dense vector, no maximum.");
        }

        T acc{};
        T square{};

        for (std::size_t i=0; i<v.size(); ++i){
            auto val = numeric::scalar::operations::checked_multiply(v[i], v[i]);
            if (val) {
                square = *val;
            } else {
                throw std::overflow_error("Checked multiplication overflow in sum_of_squares()");
            }
            auto tmp = numeric::scalar::operations::checked_addition(acc, square);
            if (tmp) {
                acc = *tmp;
            } else {
                throw std::overflow_error("Checked addition overflow in sum_of_squares()");
            }
        }
        return acc;
    }

    // TODO: add sparse sum_of_squares

    template <VectorLike V, numeric::scalar::NumberLike U>
    inline auto sum_of_powers(const V& v, const U& u)
    {
        using T = typename V::value_type;

        if (v.empty()) {
            throw std::out_of_range("Empty vector, no maximum.");
        }

        T acc{};
        T x{};

        for (std::size_t i=0; i<v.size(); ++i){
            auto val = numeric::scalar::operations::checked_pow(v[i], u);
            if (val) {
                x = *val;
            } else {
                throw std::overflow_error("Checked pow overflow in sum_of_powers()");
            }
            auto tmp = numeric::scalar::operations::checked_addition(acc, x);
            if (tmp) {
                acc = *tmp;
            } else {
                throw std::overflow_error("Checked addition in sum_of_powers()");
            }
        }
        return acc;
    }

    // TODO: add sparse sum_of_powers
    /*
    template <VectorLike V>
    inline auto gcd(const V& v)
    {

    }

    template <VectorLike V>
    inline auto lcm(const V& v)
    {

    }

    template <VectorLike V>
    inline auto log_sum_exp(const V& v)
    {

    }

    template <VectorLike V>
    inline auto squared_l2_norm(const V& v)
    {

    }

    template <VectorLike V>
    inline auto linf_norm(const V& v)
    {

    }

    template <VectorLike V>
    inline auto lp_norm(const V& v)
    {

    }

    template <VectorLike V>
    inline auto rms(const V& v)
    {

    }

    template <VectorLike V>
    inline auto variance(const V& v)
    {

    }

    template <VectorLike V>
    inline auto std_dev(const V& v)
    {

    }

    template <VectorLike V>
    inline auto skewness(const V& v)
    {

    }

    template <VectorLike V>
    inline auto kurtosis(const V& v)
    {

    }
    */
    //======================
    // Normalize
    //======================

    template<VectorLike V>
    inline auto normalize(const V& v)
    {
        return v / norm(v);
    }

    //===============================
    // Element-wise uary math wrappers (abs, sqrt, exp, ...)
    //===============================

    /*
    template<VectorLike V>
    inline auto abs(const V& v)
    {
        auto r = detail::make_result_vector(v,typename V::value_type{});
        for (std::size_t i=0; i<v.size();++i) {
            auto tmp = numeric::scalar::operations::checked_abs(v[i]);
            if (tmp) {
                r[i] = *tmp;
            } else {
                std::cerr << "Undefined behavior for abs (overflow).\n";
            }
        }
        return r;
    }

    template<VectorLike V>
    inline auto sqrt(const V& v)
    {
        auto r = detail::make_result_vector(v,typename V::value_type{});
        std::transform(v.begin(), v.end(), r.begin(),
                       [](auto x){ return std::sqrt(x);});
        return r;
    }

    template<VectorLike V>
    inline auto exp(const V& v)
    {
        auto r = detail::make_result_vector(v,typename V::value_type{});
        std::transform(v.begin(), v.end(), r.begin(),
                       [](auto x){ return std::exp(x);});
        return r;
    }

    template<VectorLike V>
    inline auto log(const V& v)
    {
        auto r = detail::make_result_vector(v,typename V::value_type{});
        std::transform(v.begin(), v.end(), r.begin(),
                       [](auto x){ return std::log(x);});
        return r;
    }

    template<VectorLike V>
    inline auto sign(const V& v)
    {
        auto r = detail::make_result_vector(v,typename V::value_type{});
        std::transform(v.begin(), v.end(), r.begin(),
                       [](auto x){ return (x > 0) - (x < 0);});
        return r;
    }
    */
    //=================================
    // Fused BLAS-1 kernels
    //=================================
    /*
    template<VectorLike Y, VectorLike X, numeric::scalar::RealLike S>
    requires detail::static_extent_match<static_size_v<Y>,static_size_v<X>>::value
    inline void axpy(S alpha, const X& x, Y& y)
    {
        detail::assert_same_length(x,y);
        for (std::size_t i=0; i<y.size();++i) {
            y[i] += alpha * x[i];
        }
    }

    template<VectorLike Y, VectorLike X, numeric::scalar::RealLike S1, numeric::scalar::RealLike S2>
    requires detail::static_extent_match<static_size_v<Y>,static_size_v<X>>::value
    inline void axpby(S1 alpha, const X& x, S2 beta, Y& y)
    {
        detail::assert_same_length(x,y);
        for (std::size_t i=0;i<y.size();++i) {
            y[i] = alpha * x[i] + beta * y[i];
        }
    }

    template<VectorLike Y, VectorLike X, numeric::scalar::RealLike S>
    requires detail::static_extent_match<static_size_v<Y>,static_size_v<X>>::value
    inline void xpay(const X& x, S beta, Y& y)
    {
        detail::assert_same_length(x,y);
        for (std::size_t i=0; i<y.size();++i) {
            y[i] = x[i] + beta * y[i];
        }
    }
    */
    //===================================
    // Weighted dot product
    //===================================

    /*
    template<VectorLike X, VectorLike Y, VectorLike W>
    requires(detail::static_extent_match<static_size_v<X>,static_size_v<W>>::value &&
             detail::static_extent_match<static_size_v<X>,static_size_v<Y>>::value)
    inline auto dot_w(const X& x, const Y& y, const W& w)
    {
        detail::assert_same_length(x,w);
        detail::assert_same_length(x,y);
        using R12 = typename numeric::scalar::promote<
                                                    typename X::value_type,
                                                    typename Y::value_type>::type;
        using R = typename numeric::scalar::promote<R12,
                                                    typename W::value_type>::type;
        R sum{};
        for (std::size_t i=0; i<x.size(); ++i) {
            sum += w[i]*x[i]*y[i];
        }
        return sum;
    }
    */
    //==================================
    // Covariance
    //==================================
    /*
    template<VectorLike X, VectorLike Y>
    requires(detail::static_extent_match<static_size_v<X>,static_size_v<Y>>::value)
    inline auto covariance(const X& x, const Y& y)
    {

    }

    template<VectorLike X, VectorLike Y>
    requires(detail::static_extent_match<static_size_v<X>,static_size_v<Y>>::value)
    inline auto correlation(const X& x, const Y& y)
    {

    }
    */
    //===================================
    // Triple Product
    //===================================

    template<VectorLike V1, VectorLike V2, VectorLike V3>
    requires(static_size_v<V1> == 3 && static_size_v<V2> == 3 && static_size_v<V3> == 3)
    inline auto triple_product(const V1& a, const V2& b, const V3& c)
    {
        return dot(a, cross(b,c));
    }

    //===================================
    // Component-wise helpers (pow, clamp)
    //===================================

    /*
    template<VectorLike V, numeric::scalar::RealLike S>
    inline auto pow(const V& v, S p)
    {
        auto r = detail::make_result_vector(v,p);
        using std::pow;
        for (std::size_t i=0; i<v.size();++i) {
            r[i] = pow(v[i],p);
        }
        return r;
    }

    template<VectorLike V>
    inline auto clamp(const V& v, typename V::value_type lo,typename V::value_type hi)
    {
        auto r = detail::make_result_vector(v, lo);
        for (std::size_t i=0;i<v.size();++i) {
            r[i] = std::clamp(v[i],lo,hi);
        }
        return r;
    }

    //===================================
    // Inclusive scan / prefix-sum
    //===================================
    template<VectorLike Vin, VectorLike Vout>
    requires detail::static_extent_match<static_size_v<Vin>,static_size_v<Vout>>::value
    inline void prefix_sum(const Vin& in, Vout& out) {
        detail::assert_same_length(in, out);
#if __cpp_lib_execution >= 201603L
        std::inclusive_scan(std::execution::par_unseq, in.begin(), in.end(), out.begin());
#else
        std::inclusive_scan(in.begin(), in.end(), out.begin());
#endif
    }
    */
    //==================================
    // Projection
    //==================================

    template<VectorLike V>
    inline auto project_orthogonal(const V& v, const V& unit_n)
    {
        return v - dot(v,unit_n) * unit_n;
    }
}
#endif //VECTOR_OPERATORS_H
