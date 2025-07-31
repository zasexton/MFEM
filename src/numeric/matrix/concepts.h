#pragma once

#ifndef MATRIX_CONCEPTS_H
#define MATRIX_CONCEPTS_H

#include "numeric/scalar/concepts.h"
#include "numeric/vector/concepts.h"

namespace numeric::matrix {
    using numeric::vector::Dense;
    using numeric::vector::Sparse;

    struct RowMajor {};
    struct ColMajor {};

    inline constexpr std::size_t Dynamic = numeric::vector::Dynamic;

    template<numeric::scalar::NumberLike T,
             std::size_t Rows = Dynamic,
             std::size_t Cols = Dynamic,
             typename LayoutTag = Dense>
    class Matrix;
}
#endif //MATRIX_CONCEPTS_H
