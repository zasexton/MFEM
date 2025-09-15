#include <gtest/gtest.h>
#include "../../../../../../include/numeric/core/sparse_tensor.h"
#include <vector>
#include <complex>
#include <random>
#include <algorithm>
#include <numeric>
#include <unordered_set>

using namespace numeric::core;

class SparseTensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        gen.seed(rd());
    }

    std::mt19937 gen;
    const double tolerance = 1e-12;
};

TEST_F(SparseTensorTest, DefaultConstructor) {
    SparseTensor<double, 3> tensor;
    EXPECT_EQ(tensor.rank(), 3);
    EXPECT_EQ(tensor.nnz(), 0);
    EXPECT_TRUE(tensor.empty());
    
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(tensor.size(i), 0);
    }
}

TEST_F(SparseTensorTest, ShapeConstructor) {
    std::array<size_t, 3> shape = {4, 5, 6};
    SparseTensor<double, 3> tensor(shape);
    
    EXPECT_EQ(tensor.rank(), 3);
    EXPECT_EQ(tensor.size(0), 4);
    EXPECT_EQ(tensor.size(1), 5);
    EXPECT_EQ(tensor.size(2), 6);
    EXPECT_EQ(tensor.nnz(), 0);
    EXPECT_TRUE(tensor.empty());
    
    EXPECT_EQ(tensor(0, 0, 0), 0.0);
    EXPECT_EQ(tensor(1, 2, 3), 0.0);
}

TEST_F(SparseTensorTest, IndexValueConstructor) {
    std::array<size_t, 3> shape = {3, 3, 3};
    std::vector<std::pair<std::array<size_t, 3>, double>> entries = {
        {{0, 1, 2}, 2.5},
        {{1, 0, 1}, -3.7},
        {{2, 2, 0}, 4.8}
    };
    
    SparseTensor<double, 3> tensor(shape, entries);
    
    EXPECT_EQ(tensor.rank(), 3);
    EXPECT_EQ(tensor.nnz(), 3);
    EXPECT_FALSE(tensor.empty());
    
    EXPECT_EQ(tensor(0, 1, 2), 2.5);
    EXPECT_EQ(tensor(1, 0, 1), -3.7);
    EXPECT_EQ(tensor(2, 2, 0), 4.8);
    EXPECT_EQ(tensor(0, 0, 0), 0.0);
}

TEST_F(SparseTensorTest, CopyConstructor) {
    std::array<size_t, 2> shape = {3, 3};
    std::vector<std::pair<std::array<size_t, 2>, double>> entries = {
        {{0, 1}, 1.5},
        {{1, 2}, 2.5},
        {{2, 0}, -1.0}
    };
    
    SparseTensor<double, 2> original(shape, entries);
    SparseTensor<double, 2> copy(original);
    
    EXPECT_EQ(copy.rank(), original.rank());
    EXPECT_EQ(copy.nnz(), original.nnz());
    
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_EQ(copy(i, j), original(i, j));
        }
    }
}

TEST_F(SparseTensorTest, MoveConstructor) {
    std::array<size_t, 2> shape = {2, 2};
    std::vector<std::pair<std::array<size_t, 2>, double>> entries = {
        {{0, 1}, 3.5},
        {{1, 0}, -2.5}
    };
    
    SparseTensor<double, 2> original(shape, entries);
    size_t original_nnz = original.nnz();
    
    SparseTensor<double, 2> moved(std::move(original));
    
    EXPECT_EQ(moved.rank(), 2);
    EXPECT_EQ(moved.nnz(), original_nnz);
    EXPECT_EQ(moved(0, 1), 3.5);
    EXPECT_EQ(moved(1, 0), -2.5);
}

TEST_F(SparseTensorTest, AssignmentOperator) {
    std::array<size_t, 2> shape = {2, 2};
    std::vector<std::pair<std::array<size_t, 2>, double>> entries = {
        {{0, 0}, 1.0},
        {{1, 1}, 2.0}
    };
    
    SparseTensor<double, 2> tensor1(shape, entries);
    SparseTensor<double, 2> tensor2;
    
    tensor2 = tensor1;
    
    EXPECT_EQ(tensor2.rank(), 2);
    EXPECT_EQ(tensor2.nnz(), 2);
    EXPECT_EQ(tensor2(0, 0), 1.0);
    EXPECT_EQ(tensor2(1, 1), 2.0);
}

TEST_F(SparseTensorTest, ElementAccess) {
    std::array<size_t, 3> shape = {3, 3, 3};
    SparseTensor<double, 3> tensor(shape);
    
    tensor.set({0, 1, 2}, 2.5);
    tensor.set({1, 2, 0}, -3.7);
    tensor.set({2, 0, 1}, 4.1);
    
    EXPECT_EQ(tensor(0, 1, 2), 2.5);
    EXPECT_EQ(tensor(1, 2, 0), -3.7);
    EXPECT_EQ(tensor(2, 0, 1), 4.1);
    EXPECT_EQ(tensor(0, 0, 0), 0.0);
    EXPECT_EQ(tensor.nnz(), 3);
}

TEST_F(SparseTensorTest, ElementUpdate) {
    std::array<size_t, 2> shape = {2, 2};
    SparseTensor<double, 2> tensor(shape);
    
    tensor.set({0, 0}, 1.0);
    EXPECT_EQ(tensor(0, 0), 1.0);
    EXPECT_EQ(tensor.nnz(), 1);
    
    tensor.set({0, 0}, 2.0);
    EXPECT_EQ(tensor(0, 0), 2.0);
    EXPECT_EQ(tensor.nnz(), 1);
    
    tensor.set({0, 0}, 0.0);
    EXPECT_EQ(tensor(0, 0), 0.0);
    EXPECT_EQ(tensor.nnz(), 0);
}

TEST_F(SparseTensorTest, BoundsChecking) {
    std::array<size_t, 2> shape = {3, 3};
    SparseTensor<double, 2> tensor(shape);
    
    EXPECT_THROW(tensor(3, 0), std::out_of_range);
    EXPECT_THROW(tensor(0, 3), std::out_of_range);
    EXPECT_THROW(tensor.set({3, 0}, 1.0), std::out_of_range);
    EXPECT_THROW(tensor.set({0, 3}, 1.0), std::out_of_range);
}

TEST_F(SparseTensorTest, Addition) {
    std::array<size_t, 2> shape = {2, 2};
    
    std::vector<std::pair<std::array<size_t, 2>, double>> entries1 = {
        {{0, 0}, 1.0},
        {{1, 1}, 2.0}
    };
    
    std::vector<std::pair<std::array<size_t, 2>, double>> entries2 = {
        {{0, 0}, 0.5},
        {{0, 1}, 1.5},
        {{1, 1}, -1.0}
    };
    
    SparseTensor<double, 2> tensor1(shape, entries1);
    SparseTensor<double, 2> tensor2(shape, entries2);
    
    auto result = tensor1 + tensor2;
    
    EXPECT_EQ(result(0, 0), 1.5);
    EXPECT_EQ(result(0, 1), 1.5);
    EXPECT_EQ(result(1, 1), 1.0);
    EXPECT_EQ(result(1, 0), 0.0);
}

TEST_F(SparseTensorTest, Subtraction) {
    std::array<size_t, 2> shape = {2, 2};
    
    std::vector<std::pair<std::array<size_t, 2>, double>> entries1 = {
        {{0, 0}, 2.0},
        {{1, 1}, 3.0}
    };
    
    std::vector<std::pair<std::array<size_t, 2>, double>> entries2 = {
        {{0, 0}, 0.5},
        {{0, 1}, 1.0},
        {{1, 1}, 1.0}
    };
    
    SparseTensor<double, 2> tensor1(shape, entries1);
    SparseTensor<double, 2> tensor2(shape, entries2);
    
    auto result = tensor1 - tensor2;
    
    EXPECT_EQ(result(0, 0), 1.5);
    EXPECT_EQ(result(0, 1), -1.0);
    EXPECT_EQ(result(1, 1), 2.0);
    EXPECT_EQ(result(1, 0), 0.0);
}

TEST_F(SparseTensorTest, ScalarMultiplication) {
    std::array<size_t, 3> shape = {3, 3, 3};
    std::vector<std::pair<std::array<size_t, 3>, double>> entries = {
        {{0, 1, 2}, 2.0},
        {{1, 0, 2}, -3.0},
        {{2, 2, 0}, 4.0}
    };
    
    SparseTensor<double, 3> tensor(shape, entries);
    auto result = tensor * 2.5;
    
    EXPECT_EQ(result(0, 1, 2), 5.0);
    EXPECT_EQ(result(1, 0, 2), -7.5);
    EXPECT_EQ(result(2, 2, 0), 10.0);
    EXPECT_EQ(result.nnz(), 3);
}

TEST_F(SparseTensorTest, TensorContraction) {
    std::array<size_t, 3> shape1 = {2, 3, 4};
    std::array<size_t, 3> shape2 = {4, 2, 5};
    
    SparseTensor<double, 3> tensor1(shape1);
    SparseTensor<double, 3> tensor2(shape2);
    
    tensor1.set({0, 1, 2}, 2.0);
    tensor1.set({1, 0, 3}, 3.0);
    tensor2.set({2, 0, 1}, 4.0);
    tensor2.set({3, 1, 2}, 5.0);
    
    auto result = tensor1.contract(tensor2, 2, 0);
    
    EXPECT_EQ(result.rank(), 4);
    EXPECT_EQ(result.size(0), 2);
    EXPECT_EQ(result.size(1), 3);
    EXPECT_EQ(result.size(2), 2);
    EXPECT_EQ(result.size(3), 5);
    
    EXPECT_EQ(result(0, 1, 0, 1), 8.0);  // 2.0 * 4.0
    EXPECT_EQ(result(1, 0, 1, 2), 15.0); // 3.0 * 5.0
}

TEST_F(SparseTensorTest, TensorProduct) {
    std::array<size_t, 2> shape1 = {2, 3};
    std::array<size_t, 2> shape2 = {4, 2};
    
    SparseTensor<double, 2> tensor1(shape1);
    SparseTensor<double, 2> tensor2(shape2);
    
    tensor1.set({0, 1}, 2.0);
    tensor1.set({1, 2}, 3.0);
    tensor2.set({1, 0}, 4.0);
    tensor2.set({3, 1}, 5.0);
    
    auto result = tensor1.outer_product(tensor2);
    
    EXPECT_EQ(result.rank(), 4);
    EXPECT_EQ(result.size(0), 2);
    EXPECT_EQ(result.size(1), 3);
    EXPECT_EQ(result.size(2), 4);
    EXPECT_EQ(result.size(3), 2);
    
    EXPECT_EQ(result(0, 1, 1, 0), 8.0);  // 2.0 * 4.0
    EXPECT_EQ(result(1, 2, 3, 1), 15.0); // 3.0 * 5.0
}

TEST_F(SparseTensorTest, TensorSlicing) {
    std::array<size_t, 3> shape = {3, 4, 5};
    std::vector<std::pair<std::array<size_t, 3>, double>> entries = {
        {{0, 1, 2}, 1.0},
        {{1, 2, 3}, 2.0},
        {{2, 0, 1}, 3.0},
        {{0, 3, 4}, 4.0}
    };
    
    SparseTensor<double, 3> tensor(shape, entries);
    
    auto slice = tensor.slice(0, 0);
    EXPECT_EQ(slice.rank(), 2);
    EXPECT_EQ(slice.size(0), 4);
    EXPECT_EQ(slice.size(1), 5);
    EXPECT_EQ(slice(1, 2), 1.0);
    EXPECT_EQ(slice(3, 4), 4.0);
}

TEST_F(SparseTensorTest, Permutation) {
    std::array<size_t, 3> shape = {2, 3, 4};
    std::vector<std::pair<std::array<size_t, 3>, double>> entries = {
        {{0, 1, 2}, 5.0},
        {{1, 2, 3}, 7.0}
    };
    
    SparseTensor<double, 3> tensor(shape, entries);
    
    std::array<size_t, 3> permutation = {2, 0, 1};
    auto permuted = tensor.permute(permutation);
    
    EXPECT_EQ(permuted.rank(), 3);
    EXPECT_EQ(permuted.size(0), 4);
    EXPECT_EQ(permuted.size(1), 2);
    EXPECT_EQ(permuted.size(2), 3);
    
    EXPECT_EQ(permuted(2, 0, 1), 5.0);
    EXPECT_EQ(permuted(3, 1, 2), 7.0);
}

TEST_F(SparseTensorTest, Norm) {
    std::array<size_t, 2> shape = {3, 3};
    std::vector<std::pair<std::array<size_t, 2>, double>> entries = {
        {{0, 0}, 3.0},
        {{1, 1}, 4.0},
        {{2, 2}, 0.0}
    };
    
    SparseTensor<double, 2> tensor(shape, entries);
    
    EXPECT_NEAR(tensor.frobenius_norm(), 5.0, tolerance);
    EXPECT_NEAR(tensor.one_norm(), 7.0, tolerance);
    EXPECT_NEAR(tensor.infinity_norm(), 4.0, tolerance);
}

TEST_F(SparseTensorTest, IteratorAccess) {
    std::array<size_t, 2> shape = {3, 3};
    std::vector<std::pair<std::array<size_t, 2>, double>> entries = {
        {{0, 1}, 1.5},
        {{1, 0}, 2.5},
        {{2, 2}, 3.5}
    };
    
    SparseTensor<double, 2> tensor(shape, entries);
    
    size_t count = 0;
    double sum = 0.0;
    for (auto it = tensor.begin(); it != tensor.end(); ++it) {
        sum += it->value;
        count++;
    }
    
    EXPECT_EQ(count, 3);
    EXPECT_NEAR(sum, 7.5, tolerance);
}

TEST_F(SparseTensorTest, ComplexNumbers) {
    using Complex = std::complex<double>;
    
    std::array<size_t, 2> shape = {2, 2};
    std::vector<std::pair<std::array<size_t, 2>, Complex>> entries = {
        {{0, 0}, Complex(1.0, 2.0)},
        {{0, 1}, Complex(3.0, -1.0)},
        {{1, 1}, Complex(0.0, 4.0)}
    };
    
    SparseTensor<Complex, 2> tensor(shape, entries);
    
    EXPECT_EQ(tensor(0, 0), Complex(1.0, 2.0));
    EXPECT_EQ(tensor(0, 1), Complex(3.0, -1.0));
    EXPECT_EQ(tensor(1, 1), Complex(0.0, 4.0));
    
    auto conjugated = tensor.conjugate();
    EXPECT_EQ(conjugated(0, 0), Complex(1.0, -2.0));
    EXPECT_EQ(conjugated(0, 1), Complex(3.0, 1.0));
    EXPECT_EQ(conjugated(1, 1), Complex(0.0, -4.0));
}

TEST_F(SparseTensorTest, HighRankTensor) {
    std::array<size_t, 5> shape = {2, 2, 2, 2, 2};
    SparseTensor<double, 5> tensor(shape);
    
    tensor.set({0, 1, 0, 1, 0}, 3.14);
    tensor.set({1, 0, 1, 0, 1}, 2.71);
    
    EXPECT_EQ(tensor.rank(), 5);
    EXPECT_EQ(tensor.nnz(), 2);
    EXPECT_EQ(tensor(0, 1, 0, 1, 0), 3.14);
    EXPECT_EQ(tensor(1, 0, 1, 0, 1), 2.71);
    EXPECT_EQ(tensor(0, 0, 0, 0, 0), 0.0);
}

TEST_F(SparseTensorTest, MemoryEfficiency) {
    const size_t n = 50;
    std::array<size_t, 3> shape = {n, n, n};
    
    SparseTensor<double, 3> dense_like(shape);
    SparseTensor<double, 3> sparse(shape);
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                dense_like.set({i, j, k}, 1.0);
            }
        }
    }
    
    for (size_t i = 0; i < 10; ++i) {
        sparse.set({i, i, i}, 1.0);
    }
    
    EXPECT_EQ(dense_like.nnz(), n * n * n);
    EXPECT_EQ(sparse.nnz(), 10);
    EXPECT_LT(sparse.memory_usage(), dense_like.memory_usage());
}

TEST_F(SparseTensorTest, FEMStiffnessTensor) {
    std::array<size_t, 4> shape = {3, 3, 8, 8};
    SparseTensor<double, 4> C(shape);
    
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 8; ++k) {
                for (size_t l = 0; l < 8; ++l) {
                    if (i == j && k == l) {
                        C.set({i, j, k, l}, 2.0);
                    } else if ((i == j && std::abs(static_cast<int>(k) - static_cast<int>(l)) == 1) ||
                               (k == l && std::abs(static_cast<int>(i) - static_cast<int>(j)) == 1)) {
                        C.set({i, j, k, l}, -0.5);
                    }
                }
            }
        }
    }
    
    EXPECT_GT(C.nnz(), 0);
    EXPECT_EQ(C.rank(), 4);
    
    for (size_t i = 0; i < 3; ++i) {
        for (size_t k = 0; k < 8; ++k) {
            EXPECT_EQ(C(i, i, k, k), 2.0);
        }
    }
}

TEST_F(SparseTensorTest, QuantumTensorProduct) {
    std::array<size_t, 2> qubit_shape = {2, 2};
    
    SparseTensor<std::complex<double>, 2> pauli_x(qubit_shape);
    SparseTensor<std::complex<double>, 2> pauli_z(qubit_shape);
    
    pauli_x.set({0, 1}, std::complex<double>(1.0, 0.0));
    pauli_x.set({1, 0}, std::complex<double>(1.0, 0.0));
    
    pauli_z.set({0, 0}, std::complex<double>(1.0, 0.0));
    pauli_z.set({1, 1}, std::complex<double>(-1.0, 0.0));
    
    auto two_qubit = pauli_x.outer_product(pauli_z);
    
    EXPECT_EQ(two_qubit.rank(), 4);
    EXPECT_EQ(two_qubit.size(0), 2);
    EXPECT_EQ(two_qubit.size(1), 2);
    EXPECT_EQ(two_qubit.size(2), 2);
    EXPECT_EQ(two_qubit.size(3), 2);
    
    EXPECT_EQ(two_qubit(0, 1, 0, 0), std::complex<double>(1.0, 0.0));
    EXPECT_EQ(two_qubit(1, 0, 1, 1), std::complex<double>(-1.0, 0.0));
}

TEST_F(SparseTensorTest, FluidDynamicsConvectionTensor) {
    const size_t nx = 5, ny = 5, nz = 5;
    std::array<size_t, 4> shape = {nx, ny, nz, 3};
    
    SparseTensor<double, 4> convection(shape);
    
    [[maybe_unused]] auto idx = [nx, ny, nz](size_t i, size_t j, size_t k) -> size_t {
        return i * ny * nz + j * nz + k;
    };
    
    for (size_t i = 1; i < nx - 1; ++i) {
        for (size_t j = 1; j < ny - 1; ++j) {
            for (size_t k = 1; k < nz - 1; ++k) {
                convection.set({i, j, k, 0}, 0.5);
                convection.set({i, j, k, 1}, -0.3);
                convection.set({i, j, k, 2}, 0.1);
            }
        }
    }
    
    EXPECT_GT(convection.nnz(), 0);
    EXPECT_EQ(convection.rank(), 4);
    
    for (size_t i = 1; i < nx - 1; ++i) {
        for (size_t j = 1; j < ny - 1; ++j) {
            for (size_t k = 1; k < nz - 1; ++k) {
                EXPECT_EQ(convection(i, j, k, 0), 0.5);
                EXPECT_EQ(convection(i, j, k, 1), -0.3);
                EXPECT_EQ(convection(i, j, k, 2), 0.1);
            }
        }
    }
}

TEST_F(SparseTensorTest, NonlinearFEMJacobian) {
    const size_t n_nodes = 8;
    const size_t n_dofs = n_nodes * 3;
    std::array<size_t, 3> shape = {n_dofs, n_dofs, n_dofs};
    
    SparseTensor<double, 3> jacobian_tensor(shape);
    
    std::vector<std::array<size_t, 8>> elements = {
        {0, 1, 2, 3, 4, 5, 6, 7}
    };
    
    for (const auto& elem : elements) {
        for (size_t i = 0; i < 8; ++i) {
            for (size_t j = 0; j < 8; ++j) {
                for (size_t k = 0; k < 8; ++k) {
                    for (size_t dof = 0; dof < 3; ++dof) {
                        size_t global_i = elem[i] * 3 + dof;
                        size_t global_j = elem[j] * 3 + dof;
                        size_t global_k = elem[k] * 3 + dof;
                        
                        if (global_i < n_dofs && global_j < n_dofs && global_k < n_dofs) {
                            double value = 0.1 * static_cast<double>(i + j + k + 1);
                            jacobian_tensor.set({global_i, global_j, global_k}, value);
                        }
                    }
                }
            }
        }
    }
    
    EXPECT_GT(jacobian_tensor.nnz(), 0);
    EXPECT_EQ(jacobian_tensor.rank(), 3);
    
    for (size_t i = 0; i < n_dofs; i += 3) {
        EXPECT_GT(jacobian_tensor(i, i, i), 0.0);
    }
}

TEST_F(SparseTensorTest, MultiphysicsCouplingTensor) {
    const size_t n_thermal = 50;
    const size_t n_mechanical = 100;
    const size_t n_electrical = 25;
    
    std::array<size_t, 3> shape = {n_thermal + n_mechanical + n_electrical, 
                                   n_thermal + n_mechanical + n_electrical,
                                   3};
    
    SparseTensor<double, 3> coupling(shape);
    
    for (size_t i = 0; i < n_thermal; ++i) {
        for (size_t j = n_thermal; j < n_thermal + n_mechanical; ++j) {
            coupling.set({i, j, 0}, 0.01);
            coupling.set({j, i, 0}, 0.01);
        }
    }
    
    for (size_t i = n_thermal; i < n_thermal + n_mechanical; ++i) {
        for (size_t j = n_thermal + n_mechanical; j < n_thermal + n_mechanical + n_electrical; ++j) {
            coupling.set({i, j, 1}, 0.005);
            coupling.set({j, i, 1}, 0.005);
        }
    }
    
    for (size_t i = 0; i < n_thermal; ++i) {
        for (size_t j = n_thermal + n_mechanical; j < n_thermal + n_mechanical + n_electrical; ++j) {
            coupling.set({i, j, 2}, 0.002);
            coupling.set({j, i, 2}, 0.002);
        }
    }
    
    EXPECT_GT(coupling.nnz(), 0);
    EXPECT_EQ(coupling.rank(), 3);
    
    EXPECT_EQ(coupling(0, n_thermal, 0), 0.01);
    EXPECT_EQ(coupling(n_thermal, n_thermal + n_mechanical, 1), 0.005);
    EXPECT_EQ(coupling(0, n_thermal + n_mechanical, 2), 0.002);
}

TEST_F(SparseTensorTest, MaterialPropertyTensor) {
    const size_t n_gauss_points = 8;
    const size_t n_elements = 100;
    std::array<size_t, 4> shape = {n_elements, n_gauss_points, 6, 6};
    
    SparseTensor<double, 4> material_tensor(shape);
    
    std::vector<double> elastic_moduli = {200e9, 80e9, 80e9, 25e9, 25e9, 25e9};
    std::vector<double> poisson_ratios = {0.3, 0.25, 0.25, 0.2, 0.2, 0.2};
    
    for (size_t elem = 0; elem < n_elements; ++elem) {
        for (size_t gp = 0; gp < n_gauss_points; ++gp) {
            for (size_t i = 0; i < 6; ++i) {
                material_tensor.set({elem, gp, i, i}, elastic_moduli[i]);
                
                for (size_t j = i + 1; j < 6; ++j) {
                    double coupling = elastic_moduli[i] * poisson_ratios[j] / (1.0 + poisson_ratios[j]);
                    material_tensor.set({elem, gp, i, j}, coupling);
                    material_tensor.set({elem, gp, j, i}, coupling);
                }
            }
        }
    }
    
    EXPECT_GT(material_tensor.nnz(), 0);
    EXPECT_EQ(material_tensor.rank(), 4);
    
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(material_tensor(0, 0, i, i), elastic_moduli[i]);
    }
}

TEST_F(SparseTensorTest, PerformanceBenchmark) {
    const size_t n = 20;
    const size_t nnz_target = 1000;
    
    std::array<size_t, 4> shape = {n, n, n, n};
    std::vector<std::pair<std::array<size_t, 4>, double>> entries;
    entries.reserve(nnz_target);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> idx_dist(0, n - 1);
    std::uniform_real_distribution<double> val_dist(-10.0, 10.0);
    
    std::unordered_set<std::string> used_indices;
    
    for (size_t count = 0; count < nnz_target; ++count) {
        std::array<size_t, 4> indices;
        std::string key;
        do {
            indices = {idx_dist(gen), idx_dist(gen), idx_dist(gen), idx_dist(gen)};
            key = std::to_string(indices[0]) + "," + std::to_string(indices[1]) + "," + 
                  std::to_string(indices[2]) + "," + std::to_string(indices[3]);
        } while (used_indices.count(key));
        
        used_indices.insert(key);
        entries.emplace_back(indices, val_dist(gen));
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    SparseTensor<double, 4> tensor(shape, entries);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto construction_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    auto result = tensor * 2.0;
    end = std::chrono::high_resolution_clock::now();
    
    auto multiply_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    EXPECT_EQ(tensor.nnz(), nnz_target);
    EXPECT_EQ(result.nnz(), nnz_target);
    EXPECT_LT(construction_time.count(), 100000);
    EXPECT_LT(multiply_time.count(), 10000);
}
