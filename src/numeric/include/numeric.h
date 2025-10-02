// Aggregator header for the FEM Numeric library
// Provides a single include to access core containers, operations,
// linear algebra primitives, and decompositions.

#pragma once

#ifndef FEM_NUMERIC_NUMERIC_H
#define FEM_NUMERIC_NUMERIC_H

// ----------------------------------------------------------------------------
// Configuration & base traits/infrastructure
// ----------------------------------------------------------------------------

#include "config/compiler.h"
#include "config/config.h"
#include "config/platform.h"
#include "config/features.h"
#include "config/precision.h"

#include "base/numeric_base.h"
#include "base/traits_base.h"
#include "base/expression_base.h"
#include "base/ops_base.h"
#include "base/broadcast_base.h"
#include "base/iterator_base.h"
#include "base/view_base.h"
#include "base/slice_base.h"
#include "base/storage_base.h"

#include "traits/type_traits.h"
#include "traits/numeric_traits.h"
#include "traits/concepts.h"
#include "traits/operation_traits.h"
#include "traits/container_traits.h"
#include "traits/storage_traits.h"
#include "traits/iterator_traits.h"
#include "traits/iterator_algorithms.h"
#include "traits/SFINAE.h"

// ----------------------------------------------------------------------------
// Core containers
// ----------------------------------------------------------------------------

#include "core/vector.h"
#include "core/matrix.h"
#include "core/tensor.h"
#include "core/vector_view.h"
#include "core/matrix_view.h"
#include "core/tensor_view.h"
#include "core/block_vector.h"
#include "core/block_matrix.h"
#include "core/sparse_vector.h"
#include "core/sparse_matrix.h"
#include "core/sparse_tensor.h"
#include "core/small_matrix.h"

// ----------------------------------------------------------------------------
// Element-wise operations & reductions
// ----------------------------------------------------------------------------

#include "operations/arithmetic.h"
#include "operations/reductions.h"

// ----------------------------------------------------------------------------
// Linear algebra (BLAS-like) and helpers
// ----------------------------------------------------------------------------

#include "linear_algebra/blas_level1.h"
#include "linear_algebra/blas_level2.h"
#include "linear_algebra/blas_level3.h"
#include "linear_algebra/norms.h"
#include "linear_algebra/sparse_ops.h"
#include "linear_algebra/householder_wy.h"

// ----------------------------------------------------------------------------
// Decompositions (LU/QR/SVD/Cholesky/Eigen/LDLT)
// ----------------------------------------------------------------------------

#include "decompositions/lu.h"
#include "decompositions/cholesky.h"
#include "decompositions/qr.h"
#include "decompositions/qr_pivoted.h"
#include "decompositions/svd.h"
#include "decompositions/eigen.h"
#include "decompositions/ldlt.h"

// ----------------------------------------------------------------------------
// Optional backends (gated by CMake definitions)
// ----------------------------------------------------------------------------

#if defined(FEM_NUMERIC_ENABLE_LAPACK)
#include "backends/lapack_backend.h"
#endif

#endif // FEM_NUMERIC_NUMERIC_H

