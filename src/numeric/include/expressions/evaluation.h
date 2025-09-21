#pragma once

#ifndef NUMERIC_EXPRESSIONS_EVALUATION_H
#define NUMERIC_EXPRESSIONS_EVALUATION_H

#include <type_traits>

#include "../base/expression_base.h"

namespace fem::numeric::expressions {

// Evaluate an expression (or container) into a destination container
template<typename Dest, typename Expr>
inline void eval_to(Dest& dest, const Expr& expr) {
    if constexpr (fem::numeric::is_expression_v<Expr>) {
        expr.eval_to(dest);
    } else {
        // Treat as terminal expression
        fem::numeric::make_expression(expr).eval_to(dest);
    }
}

// Materialize an expression/container into a new container type
// Usage: auto v = materialize<Vector>(expr);
template<template<typename> class Container, typename Expr>
inline auto materialize(const Expr& expr) {
    using T = typename std::decay_t<Expr>::value_type;
    if constexpr (fem::numeric::is_expression_v<Expr>) {
        return expr.template eval<Container, T>();
    } else {
        // Wrap and eval
        auto term = fem::numeric::make_expression(expr);
        return term.template eval<Container, T>();
    }
}

// Assign overload to support: assign(dest, expr)
template<typename Dest, typename Expr>
inline void assign(Dest& dest, const Expr& expr) {
    eval_to(dest, expr);
}

} // namespace fem::numeric::expressions

#endif // NUMERIC_EXPRESSIONS_EVALUATION_H

