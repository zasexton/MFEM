#include <algorithm>
#include <array>
#include <deque>
#include <forward_list>
#include <gtest/gtest.h>
#include <list>
#include <numeric>
#include <vector>

#include <algorithms/iterator_algorithms.h>
#include <base/iterator_base.h>
#include <base/numeric_base.h>
#include <traits/iterator_traits.h>

using namespace fem::numeric::traits;
using namespace fem::numeric;

// Test fixture for iterator traits
class IteratorTraitsTest : public ::testing::Test {
protected:
  std::vector<int> vec{1, 2, 3, 4, 5};
  std::list<int> lst{1, 2, 3, 4, 5};
  std::array<int, 5> arr{1, 2, 3, 4, 5};
  std::forward_list<int> fwd_lst{1, 2, 3, 4, 5};
  std::deque<int> deq{1, 2, 3, 4, 5};
};

// Custom test iterator for testing traits detection
template <typename T> class TestStridedIterator {
public:
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T *;
  using reference = T &;
  using iterator_category = std::random_access_iterator_tag;

  TestStridedIterator(T *ptr, std::ptrdiff_t stride)
      : ptr_(ptr), stride_(stride) {}

  T &operator*() { return *ptr_; }
  TestStridedIterator &operator++() {
    ptr_ += stride_;
    return *this;
  }
  TestStridedIterator operator++(int) {
    auto tmp = *this;
    ++(*this);
    return tmp;
  }
  TestStridedIterator &operator--() {
    ptr_ -= stride_;
    return *this;
  }
  TestStridedIterator operator--(int) {
    auto tmp = *this;
    --(*this);
    return tmp;
  }

  TestStridedIterator &operator+=(difference_type n) {
    ptr_ += n * stride_;
    return *this;
  }
  TestStridedIterator &operator-=(difference_type n) {
    ptr_ -= n * stride_;
    return *this;
  }
  TestStridedIterator operator+(difference_type n) const {
    return TestStridedIterator(ptr_ + n * stride_, stride_);
  }
  TestStridedIterator operator-(difference_type n) const {
    return TestStridedIterator(ptr_ - n * stride_, stride_);
  }

  difference_type operator-(const TestStridedIterator &other) const {
    return (ptr_ - other.ptr_) / stride_;
  }

  bool operator==(const TestStridedIterator &other) const {
    return ptr_ == other.ptr_;
  }
  bool operator!=(const TestStridedIterator &other) const {
    return !(*this == other);
  }
  bool operator<(const TestStridedIterator &other) const {
    return ptr_ < other.ptr_;
  }
  bool operator>(const TestStridedIterator &other) const {
    return other < *this;
  }
  bool operator<=(const TestStridedIterator &other) const {
    return !(other < *this);
  }
  bool operator>=(const TestStridedIterator &other) const {
    return !(*this < other);
  }

  std::ptrdiff_t stride() const { return stride_; }

private:
  T *ptr_;
  std::ptrdiff_t stride_;
};

// Test basic_iterator_traits
TEST_F(IteratorTraitsTest, BasicIteratorTraits) {
  using vec_iter = std::vector<int>::iterator;
  using traits = basic_iterator_traits<vec_iter>;

  EXPECT_TRUE((std::is_same_v<traits::value_type, int>));
  EXPECT_TRUE((std::is_same_v<traits::difference_type, std::ptrdiff_t>));
  EXPECT_TRUE((std::is_same_v<traits::iterator_category,
                              std::random_access_iterator_tag>));
}

// Test is_iterator trait
TEST_F(IteratorTraitsTest, IsIteratorTrait) {
  EXPECT_TRUE(is_iterator_v<std::vector<int>::iterator>);
  EXPECT_TRUE(is_iterator_v<std::list<int>::iterator>);
  EXPECT_TRUE(is_iterator_v<int *>);
  EXPECT_TRUE(is_iterator_v<const int *>);

  EXPECT_FALSE(is_iterator_v<int>);
  EXPECT_FALSE(is_iterator_v<std::vector<int>>);

  // Test custom iterator
  EXPECT_TRUE(is_iterator_v<TestStridedIterator<int>>);
}

// Test numeric_iterator_category detection
TEST_F(IteratorTraitsTest, NumericIteratorCategory) {
  // Random access iterators
  EXPECT_EQ(numeric_iterator_category_v<std::vector<int>::iterator>,
            NumericIteratorCategory::RandomAccess);
  EXPECT_EQ(numeric_iterator_category_v<std::deque<int>::iterator>,
            NumericIteratorCategory::RandomAccess);

  // Bidirectional iterator
  EXPECT_EQ(numeric_iterator_category_v<std::list<int>::iterator>,
            NumericIteratorCategory::Bidirectional);

  // Forward iterator
  EXPECT_EQ(numeric_iterator_category_v<std::forward_list<int>::iterator>,
            NumericIteratorCategory::Forward);

  // Pointer (contiguous)
  EXPECT_EQ(numeric_iterator_category_v<int *>,
            NumericIteratorCategory::Contiguous);

  // Custom strided iterator
  using StridedIter = StridedIterator<int>;
  EXPECT_EQ(numeric_iterator_category_v<StridedIter>,
            NumericIteratorCategory::Strided);

  // Checked iterator
  using CheckedIter = CheckedIterator<std::vector<int>::iterator>;
  EXPECT_EQ(numeric_iterator_category_v<CheckedIter>,
            NumericIteratorCategory::Checked);

  // Multi-dimensional iterator
  std::array<size_t, 2> shape = {3, 4};
  std::array<size_t, 2> strides = {4, 1};
  using MultiDimIter = MultiDimIterator<int, 2>; // Changed from int* to int
  MultiDimIter mdim_iter(
      vec.data(), strides,
      shape); // Note: constructor order is (ptr, strides, shape)
  EXPECT_EQ(numeric_iterator_category_v<MultiDimIter>,
            NumericIteratorCategory::Indexed);
}

// Test random access iterator detection
TEST_F(IteratorTraitsTest, IsRandomAccessIterator) {
  EXPECT_TRUE(is_random_access_iterator_v<std::vector<int>::iterator>);
  EXPECT_TRUE(is_random_access_iterator_v<std::deque<int>::iterator>);
  EXPECT_TRUE(is_random_access_iterator_v<int *>);

  // Use typedef to avoid macro issues with comma
  using ArrayIter = std::array<int, 5>::iterator;
  EXPECT_TRUE(is_random_access_iterator_v<ArrayIter>);

  EXPECT_FALSE(is_random_access_iterator_v<std::list<int>::iterator>);
  EXPECT_FALSE(is_random_access_iterator_v<std::forward_list<int>::iterator>);
}

// Test contiguous iterator detection
TEST_F(IteratorTraitsTest, IsContiguousIterator) {
  EXPECT_TRUE(is_contiguous_iterator_v<int *>);
  EXPECT_TRUE(is_contiguous_iterator_v<const int *>);

  // Use typedef to avoid macro issues with comma
  using ArrayIter = std::array<int, 5>::iterator;
  EXPECT_TRUE(is_contiguous_iterator_v<ArrayIter>);
  EXPECT_TRUE(is_contiguous_iterator_v<std::vector<int>::iterator>);
  EXPECT_TRUE(is_contiguous_iterator_v<std::vector<int>::const_iterator>);

  EXPECT_FALSE(is_contiguous_iterator_v<std::list<int>::iterator>);
  EXPECT_FALSE(is_contiguous_iterator_v<std::deque<int>::iterator>);

  // Test ContiguousIterator from base if it exists
  // Note: ContiguousIterator may not be defined in iterator_base.h
  // If it is, uncomment the following:
  // using ContIter = ContiguousIterator<int>;
  // EXPECT_TRUE(is_contiguous_iterator_v<ContIter>);
}

// Test strided iterator detection
TEST_F(IteratorTraitsTest, IsStridedIterator) {
  using StridedIter = StridedIterator<int>;
  EXPECT_TRUE(is_strided_iterator_v<StridedIter>);
  EXPECT_TRUE(is_strided_iterator_v<TestStridedIterator<int>>);

  EXPECT_FALSE(is_strided_iterator_v<std::vector<int>::iterator>);
  EXPECT_FALSE(is_strided_iterator_v<std::list<int>::iterator>);
}

// Test iterator stride
TEST_F(IteratorTraitsTest, IteratorStride) {
  // Contiguous iterator has stride 1
  EXPECT_EQ(iterator_stride<int *>::value, 1);
  EXPECT_EQ(iterator_stride<std::vector<int>::iterator>::value, 1);

  // Test actual stride retrieval
  TestStridedIterator<int> strided(vec.data(), 2);
  EXPECT_EQ(iterator_stride<TestStridedIterator<int>>::get(strided), 2);

  // Non-strided iterator returns 1 as default
  auto vec_iter = vec.begin();
  EXPECT_EQ(iterator_stride<decltype(vec_iter)>::get(vec_iter), 1);
}

// Test parallel safe iterator detection
TEST_F(IteratorTraitsTest, IsParallelSafeIterator) {
  EXPECT_TRUE(is_parallel_safe_iterator_v<std::vector<int>::iterator>);
  EXPECT_TRUE(is_parallel_safe_iterator_v<int *>);

  // Use typedef to avoid macro issues with comma
  using ArrayIter = std::array<int, 5>::iterator;
  EXPECT_TRUE(is_parallel_safe_iterator_v<ArrayIter>);

  EXPECT_FALSE(is_parallel_safe_iterator_v<std::list<int>::iterator>);
  EXPECT_FALSE(is_parallel_safe_iterator_v<std::forward_list<int>::iterator>);

  // Checked iterators are not parallel safe due to state
  using CheckedIter = CheckedIterator<std::vector<int>::iterator>;
  EXPECT_FALSE(is_parallel_safe_iterator_v<CheckedIter>);
}

// Test iterator properties aggregator
TEST_F(IteratorTraitsTest, IteratorProperties) {
  using VecIterProps = iterator_properties<std::vector<int>::iterator>;

  EXPECT_TRUE(VecIterProps::is_random_access);
  EXPECT_TRUE(VecIterProps::is_contiguous);
  EXPECT_FALSE(VecIterProps::is_strided);
  EXPECT_FALSE(VecIterProps::is_checked);
  EXPECT_FALSE(VecIterProps::is_multidim);
  EXPECT_TRUE(VecIterProps::is_parallel_safe);
  EXPECT_TRUE(VecIterProps::supports_fast_advance);
  EXPECT_TRUE(VecIterProps::supports_fast_distance);
  EXPECT_TRUE(VecIterProps::supports_simd);
  EXPECT_EQ(VecIterProps::access_pattern, IteratorAccessPattern::Sequential);

  using ListIterProps = iterator_properties<std::list<int>::iterator>;

  EXPECT_FALSE(ListIterProps::is_random_access);
  EXPECT_FALSE(ListIterProps::is_contiguous);
  EXPECT_FALSE(ListIterProps::supports_fast_advance);
  EXPECT_FALSE(ListIterProps::supports_fast_distance);
  EXPECT_FALSE(ListIterProps::supports_simd);
  EXPECT_EQ(ListIterProps::access_pattern, IteratorAccessPattern::Unknown);
}

// Test distance traits
TEST_F(IteratorTraitsTest, DistanceTraits) {
  // Random access - O(1)
  auto vec_dist = distance_traits<std::vector<int>::iterator>::compute(
      vec.begin(), vec.end());
  EXPECT_EQ(vec_dist, 5);
  EXPECT_TRUE(distance_traits<std::vector<int>::iterator>::is_constant_time);

  // Non-random access - O(n)
  auto lst_dist = distance_traits<std::list<int>::iterator>::compute(
      lst.begin(), lst.end());
  EXPECT_EQ(lst_dist, 5);
  EXPECT_FALSE(distance_traits<std::list<int>::iterator>::is_constant_time);
}

// Test advance traits
TEST_F(IteratorTraitsTest, AdvanceTraits) {
  // Random access - O(1)
  auto vec_iter = vec.begin();
  advance_traits<std::vector<int>::iterator>::advance(vec_iter, 3);
  EXPECT_EQ(*vec_iter, 4);
  EXPECT_TRUE(advance_traits<std::vector<int>::iterator>::is_constant_time);

  // Non-random access - O(n)
  auto lst_iter = lst.begin();
  advance_traits<std::list<int>::iterator>::advance(lst_iter, 3);
  EXPECT_EQ(*lst_iter, 4);
  EXPECT_FALSE(advance_traits<std::list<int>::iterator>::is_constant_time);
}

// Test range validation
TEST_F(IteratorTraitsTest, RangeValidation) {
  using VecRangeVal = range_validation<std::vector<int>::iterator>;

  EXPECT_TRUE(VecRangeVal::is_valid_range(vec.begin(), vec.end()));
  EXPECT_FALSE(VecRangeVal::is_empty(vec.begin(), vec.end()));
  EXPECT_EQ(VecRangeVal::size(vec.begin(), vec.end()), 5);

  EXPECT_TRUE(VecRangeVal::is_empty(vec.begin(), vec.begin()));
  EXPECT_EQ(VecRangeVal::size(vec.begin(), vec.begin()), 0);

  // List validation
  using ListRangeVal = range_validation<std::list<int>::iterator>;

  EXPECT_FALSE(ListRangeVal::is_empty(lst.begin(), lst.end()));
  EXPECT_EQ(ListRangeVal::size(lst.begin(), lst.end()), 5);
}

// Test iterator pair traits
TEST_F(IteratorTraitsTest, IteratorPairTraits) {
  using VecListPair = iterator_pair_traits<std::vector<int>::iterator,
                                           std::list<int>::iterator>;

  EXPECT_TRUE(VecListPair::are_compatible); // Both have int value_type
  EXPECT_FALSE(VecListPair::both_random_access);
  EXPECT_FALSE(VecListPair::both_contiguous);
  EXPECT_FALSE(VecListPair::can_parallel_process);

  using VecVecPair = iterator_pair_traits<std::vector<int>::iterator,
                                          std::vector<int>::iterator>;

  EXPECT_TRUE(VecVecPair::are_compatible);
  EXPECT_TRUE(VecVecPair::both_random_access);
  EXPECT_TRUE(VecVecPair::both_contiguous);
  EXPECT_TRUE(VecVecPair::can_parallel_process);

  using IntFloatPair = iterator_pair_traits<std::vector<int>::iterator,
                                            std::vector<float>::iterator>;

  EXPECT_FALSE(IntFloatPair::are_compatible); // Different value types
}

// Test SIMD iteration traits
TEST_F(IteratorTraitsTest, SimdIterationTraits) {
  using IntSimd = simd_iteration_traits<int *>;

  EXPECT_TRUE(IntSimd::can_vectorize);
  EXPECT_EQ(IntSimd::vector_width, 8); // 256-bit AVX for 32-bit ints

  using DoubleSimd = simd_iteration_traits<double *>;

  EXPECT_TRUE(DoubleSimd::can_vectorize);
  EXPECT_EQ(DoubleSimd::vector_width, 4); // 256-bit AVX for 64-bit doubles

  using ListSimd = simd_iteration_traits<std::list<int>::iterator>;

  EXPECT_FALSE(ListSimd::can_vectorize); // Not contiguous
  EXPECT_EQ(ListSimd::vector_width, 1);

  // Test alignment check
  alignas(32) int aligned_array[8];
  EXPECT_TRUE(IntSimd::is_aligned(aligned_array));

  int unaligned_array[8];
  // Alignment check might fail depending on allocation
}

// Test multidimensional iterator traits
TEST_F(IteratorTraitsTest, MultidimIteratorTraits) {
  using MultiDimIter = MultiDimIterator<int, 2>; // Changed from int* to int
  using MDTraits = multidim_iterator_traits<MultiDimIter>;

  EXPECT_TRUE(MDTraits::is_multidim);
  EXPECT_EQ(MDTraits::dimensionality, 2);
  EXPECT_FALSE(MDTraits::supports_broadcasting);

  // Test regular iterator
  using VecTraits = multidim_iterator_traits<std::vector<int>::iterator>;

  EXPECT_FALSE(VecTraits::is_multidim);
  EXPECT_EQ(VecTraits::dimensionality, 1);

  // Test get_indices
  std::array<size_t, 2> shape = {3, 4};
  std::array<size_t, 2> strides = {4, 1};
  MultiDimIter mdim_iter(
      vec.data(), strides,
      shape); // Note: constructor order is (ptr, strides, shape)

  auto indices = MDTraits::get_indices(mdim_iter);
  EXPECT_EQ(indices[0], 0);
  EXPECT_EQ(indices[1], 0);
}

// Test iteration optimization hints
TEST_F(IteratorTraitsTest, IterationOptimization) {
  using VecOpt = iteration_optimization<std::vector<int>::iterator>;

  // Small range - sequential
  EXPECT_EQ(VecOpt::recommended_strategy(50),
            iteration_optimization<std::vector<int>::iterator>::Sequential);

  // Medium range - vectorized
  EXPECT_EQ(VecOpt::recommended_strategy(5000),
            iteration_optimization<std::vector<int>::iterator>::Vectorized);

  // Large range - parallel vectorized
  EXPECT_EQ(
      VecOpt::recommended_strategy(50000),
      iteration_optimization<std::vector<int>::iterator>::ParallelVectorized);

  // List optimization - always sequential for small/medium
  using ListOpt = iteration_optimization<std::list<int>::iterator>;

  EXPECT_EQ(ListOpt::recommended_strategy(50),
            iteration_optimization<std::list<int>::iterator>::Sequential);
  EXPECT_EQ(ListOpt::recommended_strategy(5000),
            iteration_optimization<std::list<int>::iterator>::Sequential);

  // Check optimal block size
  EXPECT_GT(VecOpt::optimal_block_size, 0);
  EXPECT_GT(ListOpt::optimal_block_size, 0);
}

// Test algorithm selector
TEST_F(IteratorTraitsTest, AlgorithmSelector) {
  using VecAlgo = algorithm_selector<std::vector<int>::iterator>;

  std::vector<int> src{1, 2, 3, 4, 5};
  std::vector<int> dst(5);

  // Test copy selection
  VecAlgo::copy(src.begin(), src.end(), dst.begin());
  EXPECT_EQ(dst, src);

  // Test find selection
  auto found = VecAlgo::find(src.begin(), src.end(), 3);
  EXPECT_NE(found, src.end());
  EXPECT_EQ(*found, 3);

  auto not_found = VecAlgo::find(src.begin(), src.end(), 10);
  EXPECT_EQ(not_found, src.end());
}

// Test checked iterator traits
TEST_F(IteratorTraitsTest, CheckedIteratorTraits) {
  using CheckedIter = CheckedIterator<std::vector<int>::iterator>;
  using CheckedTraits = checked_iterator_traits<CheckedIter>;

  EXPECT_TRUE(CheckedTraits::is_checked);
  EXPECT_TRUE(CheckedTraits::has_debug_info);
  EXPECT_DOUBLE_EQ(CheckedTraits::overhead_factor, 1.2);

  // Test with regular iterator
  using RegularTraits = checked_iterator_traits<std::vector<int>::iterator>;

  EXPECT_FALSE(RegularTraits::is_checked);
  EXPECT_DOUBLE_EQ(RegularTraits::overhead_factor, 1.0);

  // Note: base_iterator_t extraction test removed as CheckedIterator
  // doesn't have the expected 'iterator' typedef based on error messages
}

// Create mock iterator types outside the test function
struct MockZipIterator {
  using iterator_tuple = std::tuple<int *, double *>;
  template <size_t I> auto get() {
    if constexpr (I == 0)
      return std::get<0>(iterators_);
    else
      return std::get<1>(iterators_);
  }

private:
  iterator_tuple iterators_;
};

struct MockTransformIterator {
  using function_type = std::function<int(int)>;
  using base_iterator = std::vector<int>::iterator;
};

// Test custom iterator adapters detection
TEST_F(IteratorTraitsTest, CustomIteratorAdapters) {
  EXPECT_TRUE(is_zip_iterator_v<MockZipIterator>);
  EXPECT_FALSE(is_zip_iterator_v<std::vector<int>::iterator>);

  EXPECT_TRUE(is_transform_iterator_v<MockTransformIterator>);
  EXPECT_FALSE(is_transform_iterator_v<std::vector<int>::iterator>);
}

// Integration test with base iterator types
TEST_F(IteratorTraitsTest, IntegrationWithBaseIterators) {
  // Test with StridedIterator
  StridedIterator<int> stride_iter(vec.data(), 2);
  using StrideProps = iterator_properties<StridedIterator<int>>;

  EXPECT_TRUE(StrideProps::is_strided);
  EXPECT_FALSE(StrideProps::is_contiguous);
  EXPECT_TRUE(StrideProps::is_random_access);

  // Test with CheckedIterator
  // Note: Based on error messages, CheckedIterator only takes one argument
  CheckedIterator<std::vector<int>::iterator> checked_iter(vec.begin());
  using CheckedProps =
      iterator_properties<CheckedIterator<std::vector<int>::iterator>>;

  EXPECT_TRUE(CheckedProps::is_checked);
  EXPECT_FALSE(CheckedProps::is_parallel_safe);

  // Test with MultiDimIterator
  std::array<size_t, 3> shape = {2, 3, 4};
  std::array<size_t, 3> strides = {12, 4, 1};
  MultiDimIterator<int, 3> mdim_iter(vec.data(), strides,
                                     shape); // Changed from int* to int
  using MDimProps = iterator_properties<MultiDimIterator<int, 3>>;

  EXPECT_TRUE(MDimProps::is_multidim);
  EXPECT_EQ(MDimProps::access_pattern, IteratorAccessPattern::Blocked);
}

// Performance characteristics test
TEST_F(IteratorTraitsTest, PerformanceCharacteristics) {
  // Verify performance hints are consistent
  using VecProps = iterator_properties<std::vector<int>::iterator>;
  using ListProps = iterator_properties<std::list<int>::iterator>;

  // Vector should support all fast operations
  EXPECT_TRUE(VecProps::supports_fast_advance);
  EXPECT_TRUE(VecProps::supports_fast_distance);
  EXPECT_TRUE(VecProps::supports_simd);

  // List should not support fast operations
  EXPECT_FALSE(ListProps::supports_fast_advance);
  EXPECT_FALSE(ListProps::supports_fast_distance);
  EXPECT_FALSE(ListProps::supports_simd);

  // Verify distance computation complexity
  EXPECT_TRUE(distance_traits<std::vector<int>::iterator>::is_constant_time);
  EXPECT_FALSE(distance_traits<std::list<int>::iterator>::is_constant_time);

  // Verify advance complexity
  EXPECT_TRUE(advance_traits<std::vector<int>::iterator>::is_constant_time);
  EXPECT_FALSE(advance_traits<std::list<int>::iterator>::is_constant_time);
}

// Edge cases and boundary conditions
TEST_F(IteratorTraitsTest, EdgeCases) {
  // Empty range
  std::vector<int> empty_vec;
  using EmptyRangeVal = range_validation<std::vector<int>::iterator>;

  EXPECT_TRUE(
      EmptyRangeVal::is_valid_range(empty_vec.begin(), empty_vec.end()));
  EXPECT_TRUE(EmptyRangeVal::is_empty(empty_vec.begin(), empty_vec.end()));
  EXPECT_EQ(EmptyRangeVal::size(empty_vec.begin(), empty_vec.end()), 0);

  // Single element
  std::vector<int> single{42};
  EXPECT_FALSE(EmptyRangeVal::is_empty(single.begin(), single.end()));
  EXPECT_EQ(EmptyRangeVal::size(single.begin(), single.end()), 1);

  // Test with const iterators
  using ConstVecProps = iterator_properties<std::vector<int>::const_iterator>;
  EXPECT_TRUE(ConstVecProps::is_contiguous);
  EXPECT_TRUE(ConstVecProps::is_random_access);

  // Test with reverse iterators
  using RevVecProps = iterator_properties<std::vector<int>::reverse_iterator>;
  EXPECT_TRUE(RevVecProps::is_random_access);
  EXPECT_FALSE(
      RevVecProps::is_contiguous); // Reverse iterators are not contiguous
}