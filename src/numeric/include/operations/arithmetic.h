#pragma once

#ifndef NUMERIC_OPERATIONS_ARITHMETIC_H
#define NUMERIC_OPERATIONS_ARITHMETIC_H

#include <type_traits>

#include "../base/expression_base.h"
#include "../base/ops_base.h"

namespace fem::numeric::operations {

// Helpers to normalize operands into expressions
template<typename T>
using is_expr = std::bool_constant<fem::numeric::is_expression_v<T>>;

template<typename T>
constexpr bool is_expr_v = is_expr<T>::value;

// Convert a container (or expression) to an expression node
template<typename X>
auto as_expr(X&& x) {
    if constexpr (is_expr_v<std::decay_t<X>>) {
        return std::forward<X>(x);
    } else {
        return fem::numeric::make_expression(std::forward<X>(x));
    }
}

// Create a scalar expression broadcast to a target shape
template<typename ScalarLike, typename ShapedExpr>
auto scalar_like(const ScalarLike& s, const ShapedExpr& shaped) {
    return fem::numeric::make_scalar_expression(s, shaped.shape());
}

// Element-wise arithmetic (named helpers over operator overloads)
template<typename LHS, typename RHS>
auto add(LHS&& lhs, RHS&& rhs) {
    return fem::numeric::make_binary_expression<fem::numeric::TypedOpWrapper<fem::numeric::ops::plus>>(
        as_expr(std::forward<LHS>(lhs)), as_expr(std::forward<RHS>(rhs))
    );
}

template<typename LHS, typename RHS>
auto sub(LHS&& lhs, RHS&& rhs) {
    return fem::numeric::make_binary_expression<fem::numeric::TypedOpWrapper<fem::numeric::ops::minus>>(
        as_expr(std::forward<LHS>(lhs)), as_expr(std::forward<RHS>(rhs))
    );
}

template<typename LHS, typename RHS>
auto mul(LHS&& lhs, RHS&& rhs) {
    return fem::numeric::make_binary_expression<fem::numeric::TypedOpWrapper<fem::numeric::ops::multiplies>>(
        as_expr(std::forward<LHS>(lhs)), as_expr(std::forward<RHS>(rhs))
    );
}

template<typename LHS, typename RHS>
auto div(LHS&& lhs, RHS&& rhs) {
    return fem::numeric::make_binary_expression<fem::numeric::TypedOpWrapper<fem::numeric::ops::divides>>(
        as_expr(std::forward<LHS>(lhs)), as_expr(std::forward<RHS>(rhs))
    );
}

template<typename LHS, typename RHS>
auto mod(LHS&& lhs, RHS&& rhs) {
    return fem::numeric::make_binary_expression<fem::numeric::TypedOpWrapper<fem::numeric::ops::mod_op>>(
        as_expr(std::forward<LHS>(lhs)), as_expr(std::forward<RHS>(rhs))
    );
}

// Scalar broadcasting flavors (expr op scalar or scalar op expr)
template<typename Expr, typename Scalar>
auto add_scalar(Expr&& e, const Scalar& s) {
    auto ee = as_expr(std::forward<Expr>(e));
    return add(ee, scalar_like(s, ee));
}

template<typename Expr, typename Scalar>
auto sub_scalar(Expr&& e, const Scalar& s) {
    auto ee = as_expr(std::forward<Expr>(e));
    return sub(ee, scalar_like(s, ee));
}

template<typename Scalar, typename Expr>
auto scalar_sub(const Scalar& s, Expr&& e) {
    auto ee = as_expr(std::forward<Expr>(e));
    return sub(scalar_like(s, ee), ee);
}

template<typename Expr, typename Scalar>
auto mul_scalar(Expr&& e, const Scalar& s) {
    auto ee = as_expr(std::forward<Expr>(e));
    return mul(ee, scalar_like(s, ee));
}

template<typename Expr, typename Scalar>
auto div_scalar(Expr&& e, const Scalar& s) {
    auto ee = as_expr(std::forward<Expr>(e));
    return div(ee, scalar_like(s, ee));
}

template<typename Scalar, typename Expr>
auto scalar_div(const Scalar& s, Expr&& e) {
    auto ee = as_expr(std::forward<Expr>(e));
    return div(scalar_like(s, ee), ee);
}

// Fused helpers (compose expressions to avoid temporaries)
// fma(a, b, c) := a*b + c
template<typename A, typename B, typename C>
auto fma(A&& a, B&& b, C&& c) {
    return add(mul(std::forward<A>(a), std::forward<B>(b)), std::forward<C>(c));
}

// axpy(alpha, x, y) := alpha*x + y (common BLAS pattern)
template<typename Alpha, typename X, typename Y>
auto axpy(const Alpha& alpha, X&& x, Y&& y) {
    return add(mul_scalar(std::forward<X>(x), alpha), std::forward<Y>(y));
}

// In-place evaluation helpers (evaluate expression into destination)
template<typename Dest, typename Expr>
void eval_into(Dest& dest, const Expr& expr) {
    as_expr(expr).eval_to(dest);
}

// In-place arithmetic on containers (single pass over dest)
template<typename Dest, typename Src>
void add_inplace(Dest& dest, const Src& src) {
    auto e = add(as_expr(dest), as_expr(src));
    e.eval_to(dest);
}

template<typename Dest, typename Src>
void sub_inplace(Dest& dest, const Src& src) {
    auto e = sub(as_expr(dest), as_expr(src));
    e.eval_to(dest);
}

template<typename Dest, typename Src>
void mul_inplace(Dest& dest, const Src& src) {
    auto e = mul(as_expr(dest), as_expr(src));
    e.eval_to(dest);
}

template<typename Dest, typename Src>
void div_inplace(Dest& dest, const Src& src) {
    auto e = div(as_expr(dest), as_expr(src));
    e.eval_to(dest);
}

template<typename Dest, typename Alpha, typename X>
void axpy_inplace(Dest& y, const Alpha& alpha, const X& x) {
    axpy(alpha, x, y).eval_to(y);
}

} // namespace fem::numeric::operations

#endif // NUMERIC_OPERATIONS_ARITHMETIC_H

