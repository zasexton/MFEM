#pragma once

#ifndef NUMERIC_OPERATIONS_REDUCTIONS_H
#define NUMERIC_OPERATIONS_REDUCTIONS_H

#include <type_traits>
#include <limits>

#include "../base/expression_base.h"
#include "../base/ops_base.h"
#include "../traits/operation_traits.h"

namespace fem::numeric::operations {

// ------- Expression helpers (lazy reduction nodes) -------

template<typename Expr, typename Op>
auto reduce_expr(const Expr& expr, Op op, int axis = -1) {
    return fem::numeric::ReductionExpression<Expr, Op>(expr, op, axis);
}

template<typename Expr>
auto sum_expr(const Expr& expr, int axis = -1) {
    using T = typename std::decay_t<Expr>::value_type;
    return reduce_expr(expr, fem::numeric::ops::sum_op<T>{}, axis);
}

template<typename Expr>
auto product_expr(const Expr& expr, int axis = -1) {
    using T = typename std::decay_t<Expr>::value_type;
    return reduce_expr(expr, fem::numeric::ops::product_op<T>{}, axis);
}

template<typename Expr>
auto min_expr(const Expr& expr, int axis = -1) {
    using T = typename std::decay_t<Expr>::value_type;
    return reduce_expr(expr, fem::numeric::ops::min_op<T>{}, axis);
}

template<typename Expr>
auto max_expr(const Expr& expr, int axis = -1) {
    using T = typename std::decay_t<Expr>::value_type;
    return reduce_expr(expr, fem::numeric::ops::max_op<T>{}, axis);
}

template<typename Expr>
auto mean_expr(const Expr& expr, int axis = -1) {
    using T = typename std::decay_t<Expr>::value_type;
    return reduce_expr(expr, fem::numeric::ops::mean_op<T>{}, axis);
}

// ------- Eager reductions to scalar -------

template<typename E>
auto sum(const E& e) {
    using T = typename std::decay_t<E>::value_type;
    if constexpr (fem::numeric::is_expression_v<E>) {
        T acc{};
        const auto sz = e.shape().size();
        for (size_t i = 0; i < sz; ++i) acc = acc + e.template eval<T>(i);
        return acc;
    } else {
        return fem::numeric::ops::sum_op<T>{}(e);
    }
}

template<typename E>
auto product(const E& e) {
    using T = typename std::decay_t<E>::value_type;
    if constexpr (fem::numeric::is_expression_v<E>) {
        T acc{1};
        const auto sz = e.shape().size();
        for (size_t i = 0; i < sz; ++i) acc = acc * e.template eval<T>(i);
        return acc;
    } else {
        return fem::numeric::ops::product_op<T>{}(e);
    }
}

template<typename E>
auto min(const E& e) {
    using T = typename std::decay_t<E>::value_type;
    if constexpr (fem::numeric::is_expression_v<E>) {
        const auto sz = e.shape().size();
        if (sz == 0) throw std::runtime_error("min of empty expression");
        T m = e.template eval<T>(0);
        for (size_t i = 1; i < sz; ++i) {
            T v = e.template eval<T>(i);
            if (v < m) m = v;
        }
        return m;
    } else {
        using Op = fem::numeric::ops::min_op<T>;
        // Fold using iterator interface
        T m = std::numeric_limits<T>::max();
        for (const auto& v : e) if (v < m) m = v;
        return m;
    }
}

template<typename E>
auto max(const E& e) {
    using T = typename std::decay_t<E>::value_type;
    if constexpr (fem::numeric::is_expression_v<E>) {
        const auto sz = e.shape().size();
        if (sz == 0) throw std::runtime_error("max of empty expression");
        T m = e.template eval<T>(0);
        for (size_t i = 1; i < sz; ++i) {
            T v = e.template eval<T>(i);
            if (v > m) m = v;
        }
        return m;
    } else {
        using Op = fem::numeric::ops::max_op<T>;
        T m = std::numeric_limits<T>::lowest();
        for (const auto& v : e) if (v > m) m = v;
        return m;
    }
}

template<typename E>
auto mean(const E& e) {
    using T = typename std::decay_t<E>::value_type;
    if constexpr (fem::numeric::is_expression_v<E>) {
        const auto sz = e.shape().size();
        if (sz == 0) throw std::runtime_error("mean of empty expression");
        T acc{};
        for (size_t i = 0; i < sz; ++i) acc = acc + e.template eval<T>(i);
        return acc / static_cast<T>(sz);
    } else {
        return fem::numeric::ops::mean_op<T>{}(e);
    }
}

// ------- Dot product (1D) -------

template<typename X, typename Y>
auto dot(const X& x, const Y& y) {
    using TX = typename std::decay_t<X>::value_type;
    using TY = typename std::decay_t[Y>::value_type;
    using T = std::common_type_t<TX, TY>;

    if constexpr (fem::numeric::is_expression_v<X> || fem::numeric::is_expression_v<Y>) {
        // Require 1D shapes of same length
        auto sx = x.shape();
        auto sy = y.shape();
        if (sx.size() != sy.size()) {
            throw std::invalid_argument("dot: size mismatch");
        }
        T acc{};
        const auto n = sx.size();
        for (size_t i = 0; i < n; ++i) {
            acc += static_cast<T>(x.template eval<T>(i) * y.template eval<T>(i));
        }
        return acc;
    } else {
        if (x.size() != y.size()) {
            throw std::invalid_argument("dot: size mismatch");
        }
        T acc{};
        auto itx = std::begin(x);
        auto ity = std::begin(y);
        for (; itx != std::end(x); ++itx, ++ity) {
            acc += static_cast<T>((*itx) * (*ity));
        }
        return acc;
    }
}

} // namespace fem::numeric::operations

#endif // NUMERIC_OPERATIONS_REDUCTIONS_H

