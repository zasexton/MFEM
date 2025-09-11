#include <base/iterator_base.h>
#include <base/numeric_base.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <complex>
#include <limits>
#include <vector>

using namespace fem::numeric;

// ============================================================================
// Iterator category & trait sanity
// ============================================================================

TEST(IteratorTraits, CategoriesAndNumericFlags) {
  using CI_d = ContainerIterator<double>;
  using SI_d = StridedIterator<double>;
  using CI_i = ContainerIterator<int>;
  using CI_c = ContainerIterator<std::complex<double>>;
  using CHK_d = CheckedIterator<CI_d>;

  // Categories
  static_assert(std::is_same_v<std::iterator_traits<CI_d>::iterator_category,
                               std::random_access_iterator_tag>);
  static_assert(std::is_same_v<std::iterator_traits<SI_d>::iterator_category,
                               std::random_access_iterator_tag>);

  // CheckedIterator should inherit iterator_category of underlying
  static_assert(std::is_same_v<std::iterator_traits<CHK_d>::iterator_category,
                               std::random_access_iterator_tag>);

  // numeric_iterator_traits
  static_assert(numeric_iterator_traits<CI_d>::is_numeric);
  static_assert(
      numeric_iterator_traits<CI_c>::is_numeric); // complex is NumberLike
  static_assert(numeric_iterator_traits<CI_d>::is_ieee_compliant);  // double
  static_assert(!numeric_iterator_traits<CI_i>::is_ieee_compliant); // int

  static_assert(numeric_iterator_traits<CI_d>::is_contiguous);
  static_assert(!numeric_iterator_traits<CI_d>::is_strided);

  static_assert(!numeric_iterator_traits<SI_d>::is_contiguous);
  static_assert(numeric_iterator_traits<SI_d>::is_strided);

  // CheckedIterator is neither contiguous nor strided by this simple trait rule
  static_assert(!numeric_iterator_traits<CHK_d>::is_contiguous);
  static_assert(!numeric_iterator_traits<CHK_d>::is_strided);

  SUCCEED();
}

// ============================================================================
// ContainerIterator tests
// ============================================================================

TEST(ContainerIterator, BasicOperations) {
  int a[5] = {10, 20, 30, 40, 50};

  ContainerIterator<int> it(a);
  EXPECT_EQ(*it, 10);
  EXPECT_EQ(it[2], 30);

  // ++ / --
  EXPECT_EQ(*(++it), 20);
  it++;
  EXPECT_EQ(*it, 30);
  EXPECT_EQ(*(it--), 30);
  EXPECT_EQ(*it, 20);
  EXPECT_EQ(*(--it), 10);

  // + / - / += / -=
  auto it2 = it + 3;
  EXPECT_EQ(*it2, 40);
  it2 = 4 + it; // commutative overload
  EXPECT_EQ(*it2, 50);
  it2 -= 2; // now at 30
  EXPECT_EQ(*it2, 30);
  it2 += 1; // now at 40
  EXPECT_EQ(*it2, 40);

  // difference & comparisons
  EXPECT_EQ((it2 - it), 3);
  EXPECT_TRUE(it < it2);
  EXPECT_TRUE(it2 > it);
  EXPECT_TRUE(it <= it2);
  EXPECT_TRUE(it2 >= it);

  // base pointer
  EXPECT_EQ(it.base(), &a[0]);
  EXPECT_EQ((it + 4).base(), &a[4]);
}

TEST(ContainerIterator, ArrowOperatorWithStruct) {
  struct Foo {
    int x;
    double y;
  };
  Foo arr[2] = {{1, 2.5}, {3, 4.5}};

  ContainerIterator<Foo> it(arr);
  EXPECT_EQ(it->x, 1);
  EXPECT_DOUBLE_EQ(it->y, 2.5);
  ++it;
  EXPECT_EQ(it->x, 3);
  EXPECT_DOUBLE_EQ(it->y, 4.5);
}

// ============================================================================
// StridedIterator tests
// ============================================================================

TEST(StridedIterator, BasicOperations) {
  int a[10];
  for (int i = 0; i < 10; ++i)
    a[i] = i;

  StridedIterator<int> it(a, 2); // even elements: 0,2,4,6,8
  EXPECT_EQ(*it, 0);
  ++it;
  EXPECT_EQ(*it, 2);
  it++;
  EXPECT_EQ(*it, 4);

  // indexing in element units (not raw pointer units)
  EXPECT_EQ(it[1], 6);  // 4 + 1*stride -> a[6]
  EXPECT_EQ(it[-1], 2); // 4 - 1*stride -> a[2]

  // arithmetic
  auto it2 = it + 2; // -> a[8]
  EXPECT_EQ(*it2, 8);
  auto it3 = 1 + it; // -> a[6]
  EXPECT_EQ(*it3, 6);
  it2 -= 1; // -> a[6]
  EXPECT_EQ(*it2, 6);

  // difference counts elements, not bytes
  EXPECT_EQ((it2 - it), 1);

  // comparisons
  EXPECT_TRUE(it < it2);
  EXPECT_TRUE(it2 > it);
  EXPECT_TRUE(it <= it2);
  EXPECT_TRUE(it2 >= it);

  // accessors
  EXPECT_EQ(it2.base(), &a[6]);
  EXPECT_EQ(it2.stride(), 2);
}

// ============================================================================
// MultiDimIterator tests
// ============================================================================

TEST(MultiDimIterator, RowMajor2x3Traversal) {
  // Buffer laid out row-major [2 x 3]:
  // row 0: 0,1,2
  // row 1: 3,4,5
  std::vector<int> buf = {0, 1, 2, 3, 4, 5};
  using It = MultiDimIterator<int, 2>;
  std::array<size_t, 2> shape{2, 3};
  std::array<size_t, 2> strides{3, 1}; // linear = i*3 + j

  It it(buf.data(), strides, shape);

  std::vector<int> seen;
  seen.reserve(buf.size());

  // indices progression sanity:
  // start: {0,0} -> 0
  // ++ -> {0,1} -> 1
  // ++ -> {0,2} -> 2
  // ++ -> {1,0} -> 3  (carry)
  EXPECT_EQ(it.indices()[0], 0u);
  EXPECT_EQ(it.indices()[1], 0u);
  EXPECT_EQ(*it, 0);
  ++it; // {0,1}
  EXPECT_EQ(it.indices()[0], 0u);
  EXPECT_EQ(it.indices()[1], 1u);
  ++it; // {0,2}
  EXPECT_EQ(it.indices()[1], 2u);
  ++it; // {1,0}
  EXPECT_EQ(it.indices()[0], 1u);
  EXPECT_EQ(it.indices()[1], 0u);

  // Restart and traverse entire tensor
  It it2(buf.data(), strides, shape);
  size_t count = 0;
  for (; !it2.is_end(); ++it2) {
    seen.push_back(*it2);
    ++count;
  }
  EXPECT_EQ(count, buf.size());
  EXPECT_EQ(seen, buf);
}

TEST(MultiDimIterator, ArrowOperatorWithStruct3D) {
  struct P {
    int v;
  };
  // 2 x 2 x 2 tensor, row-major-like strides {4,2,1}
  std::vector<P> buf(8);
  for (size_t i = 0; i < 8; ++i)
    buf[i].v = static_cast<int>(i);

  using It = MultiDimIterator<P, 3>;
  std::array<size_t, 3> shape{2, 2, 2};
  std::array<size_t, 3> strides{4, 2, 1};

  It it(buf.data(), strides, shape);
  EXPECT_EQ(it->v, 0);
  ++it; // {0,0,1}
  EXPECT_EQ(it->v, 1);
  ++it; // {0,1,0}
  EXPECT_EQ(it->v, 2);
  ++it; // {0,1,1}
  EXPECT_EQ(it->v, 3);
}

// ============================================================================
// CheckedIterator tests
// ============================================================================

TEST(CheckedIterator, PassThroughWhenDisabled) {
  // With check_finite=false (default), NaN/Inf should not throw.
  std::vector<double> v = {1.0, std::numeric_limits<double>::quiet_NaN(), 3.0};
  ContainerIterator<double> raw(v.data());
  CheckedIterator<ContainerIterator<double>> chk(raw);

  // Iterate and read values; no throw expected
  EXPECT_NO_THROW({
    auto it = chk;
    double a = *it;
    (void)a;
    ++it;
    double b = *it;
    (void)b; // NaN ok when disabled
    ++it;
    double c = *it;
    (void)c;
  });

  // operator[] should also pass through
  EXPECT_NO_THROW({
    auto x = chk[1];
    (void)x;
  });

  // operator-> exists; call it (returns double*) and dereference
  EXPECT_NO_THROW({
    auto p = chk.operator->();
    (void)*p;
  });
}

TEST(CheckedIterator, ThrowsWhenEnabled) {
  // Enable finite checking
  auto &opts = NumericOptions::defaults();
  const bool old_flag = opts.check_finite;
  opts.check_finite = true;

  std::vector<double> v = {1.0, 2.0, std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity()};
  ContainerIterator<double> raw(v.data());
  CheckedIterator<ContainerIterator<double>> chk(raw);

  // Advance to NaN and dereference
  auto it = chk;
  ++it; // at 2.0
  ++it; // at NaN
  EXPECT_THROW(*it, ComputationError);

  // operator[] hitting NaN
  EXPECT_THROW((void)chk[2], ComputationError);

  // Advance to Inf and use operator-> (which checks before returning)
  it = chk;
  it++;
  it++;
  it++; // at Inf
  EXPECT_THROW((void)it.operator->(), ComputationError);

  // restore
  opts.check_finite = old_flag;
}

TEST(CheckedIterator, BaseAndEquality) {
  std::vector<int> v = {1, 2, 3};
  ContainerIterator<int> raw(v.data());
  CheckedIterator<ContainerIterator<int>> a(raw), b(raw);
  EXPECT_TRUE(a == b);
  ++b;
  EXPECT_TRUE(a != b);
  EXPECT_EQ(b.base(), raw + 1);
}

TEST(ContainerIterator, DereferenceOperator) {
  // Test line 30 - operator*
  int arr[] = {42, 84};
  ContainerIterator<int> it(arr);
  EXPECT_EQ(*it, 42); // Direct dereference
  *it = 100;          // Modify through iterator
  EXPECT_EQ(arr[0], 100);
}

TEST(ContainerIterator, DifferenceOperator) {
  // Test lines 82-83 - operator- between iterators
  int arr[] = {1, 2, 3, 4, 5};
  ContainerIterator<int> it1(arr);
  ContainerIterator<int> it2(arr + 3);

  auto diff = it2 - it1;
  EXPECT_EQ(diff, 3);

  diff = it1 - it2;
  EXPECT_EQ(diff, -3);
}

TEST(ContainerIterator, NotEqualOperator) {
  // Test lines 93-94 - operator!=
  int arr[] = {1, 2};
  ContainerIterator<int> it1(arr);
  ContainerIterator<int> it2(arr);

  EXPECT_FALSE(it1 != it2); // Same position
  ++it2;
  EXPECT_TRUE(it1 != it2); // Different positions
}

TEST(ContainerIterator, GreaterThanOperator) {
  // Test lines 103-104 - operator>
  int arr[] = {1, 2, 3};
  ContainerIterator<int> it1(arr);
  ContainerIterator<int> it2(arr + 2);

  EXPECT_FALSE(it1 > it2);
  EXPECT_TRUE(it2 > it1);
  EXPECT_FALSE(it1 > it1);
}

TEST(ContainerIterator, LessEqualAndGreaterEqual) {
  // Test lines 108-109, 113-114 - operator<= and operator>=
  int arr[] = {1, 2, 3};
  ContainerIterator<int> it1(arr);
  ContainerIterator<int> it2(arr + 1);

  EXPECT_TRUE(it1 <= it2);
  EXPECT_TRUE(it1 <= it1);
  EXPECT_FALSE(it2 <= it1);

  EXPECT_TRUE(it2 >= it1);
  EXPECT_TRUE(it1 >= it1);
  EXPECT_FALSE(it1 >= it2);
}

// ============================================================================
// Additional StridedIterator coverage
// ============================================================================

TEST(StridedIterator, DereferenceOperator) {
  // Test line 142 - operator*
  int arr[] = {10, 20, 30, 40};
  StridedIterator<int> it(arr, 2);

  EXPECT_EQ(*it, 10);
  *it = 15; // Modify through iterator
  EXPECT_EQ(arr[0], 15);
}

TEST(StridedIterator, PostIncrement) {
  // Test lines 154-158 - operator++(int)
  int arr[] = {0, 1, 2, 3, 4, 5};
  StridedIterator<int> it(arr, 2);

  auto old = it++;
  EXPECT_EQ(*old, 0);
  EXPECT_EQ(*it, 2);
}

TEST(StridedIterator, PreDecrement) {
  // Test lines 160-163 - operator--
  int arr[] = {0, 1, 2, 3, 4, 5};
  StridedIterator<int> it(arr + 4, 2); // Start at element 4

  --it;
  EXPECT_EQ(*it, 2);
  auto &ref = --it;
  EXPECT_EQ(*ref, 0);
}

TEST(StridedIterator, PostDecrement) {
  // Test lines 165-169 - operator--(int)
  int arr[] = {0, 1, 2, 3, 4, 5};
  StridedIterator<int> it(arr + 4, 2);

  auto old = it--;
  EXPECT_EQ(*old, 4);
  EXPECT_EQ(*it, 2);
}

TEST(StridedIterator, PlusEqualsAndMinusEquals) {
  // Test lines 171-174, 176-178
  int arr[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  StridedIterator<int> it(arr, 2);

  it += 3;
  EXPECT_EQ(*it, 6);

  it -= 2;
  EXPECT_EQ(*it, 2);
}

TEST(StridedIterator, CommutativeAddAndSubtract) {
  // Test lines 185-187, 189-191
  int arr[] = {0, 1, 2, 3, 4, 5};
  StridedIterator<int> it(arr, 2);

  // Commutative add (n + iterator)
  auto it2 = 2 + it;
  EXPECT_EQ(*it2, 4);

  // Subtract
  auto it3 = it2 - 1;
  EXPECT_EQ(*it3, 2);
}

TEST(StridedIterator, DifferenceBetweenIterators) {
  // Test lines 193-196
  int arr[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  StridedIterator<int> it1(arr, 2);
  StridedIterator<int> it2(arr + 6, 2); // 3 strides ahead

  auto diff = it2 - it1;
  EXPECT_EQ(diff, 3);
}

TEST(StridedIterator, NotEqualOperator) {
  // Test lines 203-206
  int arr[] = {1, 2, 3};
  StridedIterator<int> it1(arr, 1);
  StridedIterator<int> it2(arr, 1);

  EXPECT_FALSE(it1 != it2);
  ++it2;
  EXPECT_TRUE(it1 != it2);
}

TEST(StridedIterator, GreaterThanOperator) {
  // Test lines 213-216
  int arr[] = {1, 2, 3, 4};
  StridedIterator<int> it1(arr, 1);
  StridedIterator<int> it2(arr + 2, 1);

  EXPECT_TRUE(it2 > it1);
  EXPECT_FALSE(it1 > it2);
}

TEST(StridedIterator, LessEqualAndGreaterEqual) {
  // Test lines 218-221, 223-226
  int arr[] = {1, 2, 3};
  StridedIterator<int> it1(arr, 1);
  StridedIterator<int> it2(arr + 1, 1);

  EXPECT_TRUE(it1 <= it2);
  EXPECT_TRUE(it1 <= it1);

  EXPECT_TRUE(it2 >= it1);
  EXPECT_TRUE(it1 >= it1);
}

TEST(StridedIterator, StrideGetter) {
  // Test line 229
  int arr[] = {1, 2, 3};
  StridedIterator<int> it(arr, 3);

  EXPECT_EQ(it.stride(), 3);
}

// ============================================================================
// Additional MultiDimIterator coverage
// ============================================================================

TEST(MultiDimIterator, PostIncrement) {
  // Test lines 279-283
  int arr[] = {0, 1, 2, 3};
  std::array<size_t, 2> shape{2, 2};
  std::array<size_t, 2> strides{2, 1};

  MultiDimIterator<int, 2> it(arr, strides, shape);

  auto old = it++;
  EXPECT_EQ(*old, 0);
  EXPECT_EQ(*it, 1);
}

TEST(MultiDimIterator, EqualityOperators) {
  // Test lines 285-288, 290-293
  int arr[] = {0, 1, 2, 3};
  std::array<size_t, 2> shape{2, 2};
  std::array<size_t, 2> strides{2, 1};

  MultiDimIterator<int, 2> it1(arr, strides, shape);
  MultiDimIterator<int, 2> it2(arr, strides, shape);

  EXPECT_TRUE(it1 == it2);
  EXPECT_FALSE(it1 != it2);

  ++it2;
  EXPECT_FALSE(it1 == it2);
  EXPECT_TRUE(it1 != it2);
}

TEST(MultiDimIterator, IndicesGetter) {
  // Test line 295
  int arr[] = {0, 1, 2, 3};
  std::array<size_t, 2> shape{2, 2};
  std::array<size_t, 2> strides{2, 1};

  MultiDimIterator<int, 2> it(arr, strides, shape);

  auto &indices = it.indices();
  EXPECT_EQ(indices[0], 0u);
  EXPECT_EQ(indices[1], 0u);

  ++it;
  EXPECT_EQ(indices[0], 0u);
  EXPECT_EQ(indices[1], 1u);
}

// ============================================================================
// Additional CheckedIterator coverage
// ============================================================================

TEST(CheckedIterator, IncrementOperators) {
  // Test lines 347-350, 352-356
  std::vector<int> v = {1, 2, 3};
  ContainerIterator<int> raw(v.data());
  CheckedIterator<ContainerIterator<int>> it(raw);

  // Pre-increment
  auto &ref = ++it;
  EXPECT_EQ(*ref, 2);

  // Post-increment
  auto old = it++;
  EXPECT_EQ(*old, 2);
  EXPECT_EQ(*it, 3);
}

TEST(CheckedIterator, EqualityOperators) {
  // Test lines 382-384, 386-388
  std::vector<int> v = {1, 2, 3};
  ContainerIterator<int> raw1(v.data());
  ContainerIterator<int> raw2(v.data());

  CheckedIterator<ContainerIterator<int>> it1(raw1);
  CheckedIterator<ContainerIterator<int>> it2(raw2);

  EXPECT_TRUE(it1 == it2);
  EXPECT_FALSE(it1 != it2);

  ++it2;
  EXPECT_FALSE(it1 == it2);
  EXPECT_TRUE(it1 != it2);
}

TEST(CheckedIterator, BaseGetter) {
  // Test line 390
  std::vector<int> v = {1, 2, 3};
  ContainerIterator<int> raw(v.data());
  CheckedIterator<ContainerIterator<int>> it(raw);

  EXPECT_EQ(it.base(), raw);
  ++it;
  EXPECT_EQ(it.base(), raw + 1);
}