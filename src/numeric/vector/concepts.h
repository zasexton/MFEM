#pragma once

#ifndef VECTOR_CONCEPTS_H
#define VECTOR_CONCEPTS_H

#include "numeric/scalar/concepts.h"

namespace numeric::vector {
    struct Dense {};
    struct Sparse {};

    inline constexpr std::size_t Dynamic = static_cast<std::size_t>(-1);

    template<typename T, std::size_t N = Dynamic, typename LayoutTag = Dense>
    requires (numeric::scalar::NumberLike<T>)
    class Vector;
}

#endif //VECTOR_CONCEPTS_H
