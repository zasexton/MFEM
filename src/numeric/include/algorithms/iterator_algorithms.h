#pragma once

#ifndef NUMERIC_ITERATOR_ALGORITHMS_H
#define NUMERIC_ITERATOR_ALGORITHMS_H

#include <algorithm>
#include <cstring>

#include "../traits/iterator_traits.h"

namespace fem::numeric::traits {

/**
 * @brief Iterator algorithm selector
 */
template <typename Iterator> struct algorithm_selector {
  using props = iterator_properties<Iterator>;

  // Select copy algorithm
  template <typename OutputIterator>
  static void copy(Iterator first, Iterator last, OutputIterator out) {
    if constexpr (is_contiguous_iterator_v<Iterator> &&
                  is_contiguous_iterator_v<OutputIterator>) {
      // Use memcpy for POD types
      if constexpr (std::is_trivially_copyable_v<typename props::value_type>) {
        size_t n = static_cast<size_t>(
            distance_traits<Iterator>::compute(first, last));
        std::memcpy(&(*out), &(*first), n * sizeof(typename props::value_type));
        return;
      }
    }

    // Fall back to std::copy
    std::copy(first, last, out);
  }

  // Select find algorithm
  template <typename T>
  static Iterator find(Iterator first, Iterator last, const T &value) {
    if constexpr (simd_iteration_traits<Iterator>::can_vectorize) {
      // Could use SIMD find for arithmetic types
      return simd_find(first, last, value);
    } else {
      return std::find(first, last, value);
    }
  }

private:
  template <typename T>
  static Iterator simd_find(Iterator first, Iterator last, const T &value) {
    // Placeholder for SIMD implementation
    return std::find(first, last, value);
  }
};

} // namespace fem::numeric::traits

#endif // NUMERIC_ITERATOR_ALGORITHMS_H
