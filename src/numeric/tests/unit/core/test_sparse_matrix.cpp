#include <gtest/gtest.h>
#include "../../../../../../include/numeric/core/sparse_matrix.h"
#include <vector>
#include <complex>
#include <random>
#include <algorithm>
#include <numeric>

using namespace numeric::core;

class SparseMatrixTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        gen.seed(rd());
    }

    std::mt19937 gen;
    const double tolerance = 1e-12;
};

TEST_F(SparseMatrixTest, DefaultConstructor) {
    SparseMatrix<double> mat;
    EXPECT_EQ(mat.rows(), 0);
    EXPECT_EQ(mat.cols(), 0);
    EXPECT_EQ(mat.nnz(), 0);
    EXPECT_TRUE(mat.empty());
}

TEST_F(SparseMatrixTest, DimensionConstructor) {
    SparseMatrix<double> mat(5, 7);
    EXPECT_EQ(mat.rows(), 5);
    EXPECT_EQ(mat.cols(), 7);
    EXPECT_EQ(mat.nnz(), 0);
    EXPECT_TRUE(mat.empty());
    
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 7; ++j) {
            EXPECT_EQ(mat(i, j), 0.0);
        }
    }
}

TEST_F(SparseMatrixTest, TripletConstructorCOO) {
    std::vector<std::tuple<size_t, size_t, double>> triplets = {
        {0, 1, 2.5},
        {1, 2, 3.7},
        {2, 0, -1.2},
        {2, 2, 4.8}
    };
    
    SparseMatrix<double> mat(3, 3, triplets, SparseMatrix<double>::StorageFormat::COO);
    
    EXPECT_EQ(mat.rows(), 3);
    EXPECT_EQ(mat.cols(), 3);
    EXPECT_EQ(mat.nnz(), 4);
    EXPECT_FALSE(mat.empty());
    
    EXPECT_EQ(mat(0, 1), 2.5);
    EXPECT_EQ(mat(1, 2), 3.7);
    EXPECT_EQ(mat(2, 0), -1.2);
    EXPECT_EQ(mat(2, 2), 4.8);
    EXPECT_EQ(mat(0, 0), 0.0);
    EXPECT_EQ(mat(1, 1), 0.0);
}

TEST_F(SparseMatrixTest, TripletConstructorCSR) {
    std::vector<std::tuple<size_t, size_t, double>> triplets = {
        {0, 1, 2.5},
        {0, 2, 1.5},
        {1, 0, -1.0},
        {1, 2, 3.7},
        {2, 1, -2.1}
    };
    
    SparseMatrix<double> mat(3, 3, triplets, SparseMatrix<double>::StorageFormat::CSR);
    
    EXPECT_EQ(mat.rows(), 3);
    EXPECT_EQ(mat.cols(), 3);
    EXPECT_EQ(mat.nnz(), 5);
    
    EXPECT_EQ(mat(0, 1), 2.5);
    EXPECT_EQ(mat(0, 2), 1.5);
    EXPECT_EQ(mat(1, 0), -1.0);
    EXPECT_EQ(mat(1, 2), 3.7);
    EXPECT_EQ(mat(2, 1), -2.1);
}

TEST_F(SparseMatrixTest, TripletConstructorCSC) {
    std::vector<std::tuple<size_t, size_t, double>> triplets = {
        {0, 0, 1.0},
        {1, 0, 2.0},
        {0, 1, 3.0},
        {2, 1, 4.0},
        {1, 2, 5.0}
    };
    
    SparseMatrix<double> mat(3, 3, triplets, SparseMatrix<double>::StorageFormat::CSC);
    
    EXPECT_EQ(mat.rows(), 3);
    EXPECT_EQ(mat.cols(), 3);
    EXPECT_EQ(mat.nnz(), 5);
    
    EXPECT_EQ(mat(0, 0), 1.0);
    EXPECT_EQ(mat(1, 0), 2.0);
    EXPECT_EQ(mat(0, 1), 3.0);
    EXPECT_EQ(mat(2, 1), 4.0);
    EXPECT_EQ(mat(1, 2), 5.0);
}

TEST_F(SparseMatrixTest, CopyConstructor) {
    std::vector<std::tuple<size_t, size_t, double>> triplets = {
        {0, 1, 2.5},
        {1, 2, 3.7},
        {2, 0, -1.2}
    };
    
    SparseMatrix<double> original(3, 3, triplets);
    SparseMatrix<double> copy(original);
    
    EXPECT_EQ(copy.rows(), original.rows());
    EXPECT_EQ(copy.cols(), original.cols());
    EXPECT_EQ(copy.nnz(), original.nnz());
    
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(copy(i, j), original(i, j));
        }
    }
}

TEST_F(SparseMatrixTest, MoveConstructor) {
    std::vector<std::tuple<size_t, size_t, double>> triplets = {
        {0, 1, 2.5},
        {1, 2, 3.7}
    };
    
    SparseMatrix<double> original(3, 3, triplets);
    size_t original_nnz = original.nnz();
    
    SparseMatrix<double> moved(std::move(original));
    
    EXPECT_EQ(moved.rows(), 3);
    EXPECT_EQ(moved.cols(), 3);
    EXPECT_EQ(moved.nnz(), original_nnz);
    EXPECT_EQ(moved(0, 1), 2.5);
    EXPECT_EQ(moved(1, 2), 3.7);
}

TEST_F(SparseMatrixTest, AssignmentOperator) {
    std::vector<std::tuple<size_t, size_t, double>> triplets = {
        {0, 0, 1.0},
        {1, 1, 2.0}
    };
    
    SparseMatrix<double> mat1(2, 2, triplets);
    SparseMatrix<double> mat2;
    
    mat2 = mat1;
    
    EXPECT_EQ(mat2.rows(), 2);
    EXPECT_EQ(mat2.cols(), 2);
    EXPECT_EQ(mat2.nnz(), 2);
    EXPECT_EQ(mat2(0, 0), 1.0);
    EXPECT_EQ(mat2(1, 1), 2.0);
}

TEST_F(SparseMatrixTest, ElementAccess) {
    SparseMatrix<double> mat(3, 3);
    
    mat.set(0, 1, 2.5);
    mat.set(1, 2, -3.7);
    mat.set(2, 0, 4.1);
    
    EXPECT_EQ(mat(0, 1), 2.5);
    EXPECT_EQ(mat(1, 2), -3.7);
    EXPECT_EQ(mat(2, 0), 4.1);
    EXPECT_EQ(mat(0, 0), 0.0);
    EXPECT_EQ(mat(1, 1), 0.0);
    EXPECT_EQ(mat.nnz(), 3);
}

TEST_F(SparseMatrixTest, ElementUpdate) {
    SparseMatrix<double> mat(2, 2);
    
    mat.set(0, 0, 1.0);
    EXPECT_EQ(mat(0, 0), 1.0);
    EXPECT_EQ(mat.nnz(), 1);
    
    mat.set(0, 0, 2.0);
    EXPECT_EQ(mat(0, 0), 2.0);
    EXPECT_EQ(mat.nnz(), 1);
    
    mat.set(0, 0, 0.0);
    EXPECT_EQ(mat(0, 0), 0.0);
    EXPECT_EQ(mat.nnz(), 0);
}

TEST_F(SparseMatrixTest, BoundsChecking) {
    SparseMatrix<double> mat(3, 3);
    
    EXPECT_THROW(mat(3, 0), std::out_of_range);
    EXPECT_THROW(mat(0, 3), std::out_of_range);
    EXPECT_THROW(mat.set(3, 0, 1.0), std::out_of_range);
    EXPECT_THROW(mat.set(0, 3, 1.0), std::out_of_range);
}

TEST_F(SparseMatrixTest, Addition) {
    std::vector<std::tuple<size_t, size_t, double>> triplets1 = {
        {0, 0, 1.0},
        {1, 1, 2.0}
    };
    
    std::vector<std::tuple<size_t, size_t, double>> triplets2 = {
        {0, 0, 0.5},
        {0, 1, 1.5},
        {1, 1, -1.0}
    };
    
    SparseMatrix<double> mat1(2, 2, triplets1);
    SparseMatrix<double> mat2(2, 2, triplets2);
    
    auto result = mat1 + mat2;
    
    EXPECT_EQ(result(0, 0), 1.5);
    EXPECT_EQ(result(0, 1), 1.5);
    EXPECT_EQ(result(1, 1), 1.0);
    EXPECT_EQ(result(1, 0), 0.0);
}

TEST_F(SparseMatrixTest, Subtraction) {
    std::vector<std::tuple<size_t, size_t, double>> triplets1 = {
        {0, 0, 2.0},
        {1, 1, 3.0}
    };
    
    std::vector<std::tuple<size_t, size_t, double>> triplets2 = {
        {0, 0, 0.5},
        {0, 1, 1.0},
        {1, 1, 1.0}
    };
    
    SparseMatrix<double> mat1(2, 2, triplets1);
    SparseMatrix<double> mat2(2, 2, triplets2);
    
    auto result = mat1 - mat2;
    
    EXPECT_EQ(result(0, 0), 1.5);
    EXPECT_EQ(result(0, 1), -1.0);
    EXPECT_EQ(result(1, 1), 2.0);
    EXPECT_EQ(result(1, 0), 0.0);
}

TEST_F(SparseMatrixTest, ScalarMultiplication) {
    std::vector<std::tuple<size_t, size_t, double>> triplets = {
        {0, 1, 2.0},
        {1, 0, -3.0},
        {2, 2, 4.0}
    };
    
    SparseMatrix<double> mat(3, 3, triplets);
    auto result = mat * 2.5;
    
    EXPECT_EQ(result(0, 1), 5.0);
    EXPECT_EQ(result(1, 0), -7.5);
    EXPECT_EQ(result(2, 2), 10.0);
    EXPECT_EQ(result.nnz(), 3);
}

TEST_F(SparseMatrixTest, MatrixVectorMultiplication) {
    std::vector<std::tuple<size_t, size_t, double>> triplets = {
        {0, 0, 1.0},
        {0, 1, 2.0},
        {1, 0, 3.0},
        {1, 2, 4.0},
        {2, 1, 5.0},
        {2, 2, 6.0}
    };
    
    SparseMatrix<double> mat(3, 3, triplets);
    std::vector<double> vec = {1.0, 2.0, 3.0};
    
    auto result = mat * vec;
    
    EXPECT_EQ(result.size(), 3);
    EXPECT_NEAR(result[0], 5.0, tolerance);   // 1*1 + 2*2 = 5
    EXPECT_NEAR(result[1], 15.0, tolerance);  // 3*1 + 4*3 = 15
    EXPECT_NEAR(result[2], 28.0, tolerance);  // 5*2 + 6*3 = 28
}

TEST_F(SparseMatrixTest, MatrixMatrixMultiplication) {
    std::vector<std::tuple<size_t, size_t, double>> triplets1 = {
        {0, 0, 1.0},
        {0, 1, 2.0},
        {1, 1, 3.0}
    };
    
    std::vector<std::tuple<size_t, size_t, double>> triplets2 = {
        {0, 0, 2.0},
        {1, 0, 1.0},
        {1, 1, 4.0}
    };
    
    SparseMatrix<double> mat1(2, 2, triplets1);
    SparseMatrix<double> mat2(2, 2, triplets2);
    
    auto result = mat1 * mat2;
    
    EXPECT_EQ(result.rows(), 2);
    EXPECT_EQ(result.cols(), 2);
    EXPECT_NEAR(result(0, 0), 4.0, tolerance);  // 1*2 + 2*1 = 4
    EXPECT_NEAR(result(0, 1), 8.0, tolerance);  // 1*0 + 2*4 = 8
    EXPECT_NEAR(result(1, 0), 3.0, tolerance);  // 0*2 + 3*1 = 3
    EXPECT_NEAR(result(1, 1), 12.0, tolerance); // 0*0 + 3*4 = 12
}

TEST_F(SparseMatrixTest, Transpose) {
    std::vector<std::tuple<size_t, size_t, double>> triplets = {
        {0, 1, 2.0},
        {0, 2, 3.0},
        {1, 0, 4.0},
        {2, 1, 5.0}
    };
    
    SparseMatrix<double> mat(3, 3, triplets);
    auto transposed = mat.transpose();
    
    EXPECT_EQ(transposed.rows(), 3);
    EXPECT_EQ(transposed.cols(), 3);
    EXPECT_EQ(transposed.nnz(), 4);
    
    EXPECT_EQ(transposed(1, 0), 2.0);
    EXPECT_EQ(transposed(2, 0), 3.0);
    EXPECT_EQ(transposed(0, 1), 4.0);
    EXPECT_EQ(transposed(1, 2), 5.0);
}

TEST_F(SparseMatrixTest, Norm) {
    std::vector<std::tuple<size_t, size_t, double>> triplets = {
        {0, 0, 3.0},
        {1, 1, 4.0}
    };
    
    SparseMatrix<double> mat(2, 2, triplets);
    
    EXPECT_NEAR(mat.frobenius_norm(), 5.0, tolerance);
    EXPECT_NEAR(mat.one_norm(), 4.0, tolerance);
    EXPECT_NEAR(mat.infinity_norm(), 4.0, tolerance);
}

TEST_F(SparseMatrixTest, StorageFormatConversion) {
    std::vector<std::tuple<size_t, size_t, double>> triplets = {
        {0, 1, 1.0},
        {1, 0, 2.0},
        {1, 2, 3.0},
        {2, 1, 4.0}
    };
    
    SparseMatrix<double> coo_mat(3, 3, triplets, SparseMatrix<double>::StorageFormat::COO);
    SparseMatrix<double> csr_mat(3, 3, triplets, SparseMatrix<double>::StorageFormat::CSR);
    SparseMatrix<double> csc_mat(3, 3, triplets, SparseMatrix<double>::StorageFormat::CSC);
    
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(coo_mat(i, j), csr_mat(i, j)) << "Mismatch at (" << i << "," << j << ")";
            EXPECT_EQ(coo_mat(i, j), csc_mat(i, j)) << "Mismatch at (" << i << "," << j << ")";
        }
    }
}

TEST_F(SparseMatrixTest, IteratorAccess) {
    std::vector<std::tuple<size_t, size_t, double>> triplets = {
        {0, 1, 1.5},
        {1, 0, 2.5},
        {2, 2, 3.5}
    };
    
    SparseMatrix<double> mat(3, 3, triplets);
    
    size_t count = 0;
    double sum = 0.0;
    for (auto it = mat.begin(); it != mat.end(); ++it) {
        sum += it->value;
        count++;
    }
    
    EXPECT_EQ(count, 3);
    EXPECT_NEAR(sum, 7.5, tolerance);
}

TEST_F(SparseMatrixTest, SparsePatternPreservation) {
    std::vector<std::tuple<size_t, size_t, double>> triplets = {
        {0, 0, 1.0},
        {0, 2, 2.0},
        {1, 1, 3.0},
        {2, 0, 4.0},
        {2, 2, 5.0}
    };
    
    SparseMatrix<double> mat1(3, 3, triplets);
    SparseMatrix<double> mat2(3, 3);
    
    mat2.set(0, 0, 2.0);
    mat2.set(1, 1, 6.0);
    mat2.set(2, 2, 10.0);
    
    auto result = mat1 + mat2;
    
    EXPECT_EQ(result.nnz(), 5);
    EXPECT_EQ(result(0, 0), 3.0);
    EXPECT_EQ(result(0, 2), 2.0);
    EXPECT_EQ(result(1, 1), 9.0);
    EXPECT_EQ(result(2, 0), 4.0);
    EXPECT_EQ(result(2, 2), 15.0);
}

TEST_F(SparseMatrixTest, ComplexNumbers) {
    using Complex = std::complex<double>;
    
    std::vector<std::tuple<size_t, size_t, Complex>> triplets = {
        {0, 0, Complex(1.0, 2.0)},
        {0, 1, Complex(3.0, -1.0)},
        {1, 1, Complex(0.0, 4.0)}
    };
    
    SparseMatrix<Complex> mat(2, 2, triplets);
    
    EXPECT_EQ(mat(0, 0), Complex(1.0, 2.0));
    EXPECT_EQ(mat(0, 1), Complex(3.0, -1.0));
    EXPECT_EQ(mat(1, 1), Complex(0.0, 4.0));
    
    auto conjugated = mat.conjugate();
    EXPECT_EQ(conjugated(0, 0), Complex(1.0, -2.0));
    EXPECT_EQ(conjugated(0, 1), Complex(3.0, 1.0));
    EXPECT_EQ(conjugated(1, 1), Complex(0.0, -4.0));
}

TEST_F(SparseMatrixTest, MemoryEfficiency) {
    const size_t n = 1000;
    SparseMatrix<double> dense_like(n, n);
    SparseMatrix<double> sparse(n, n);
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            dense_like.set(i, j, 1.0);
        }
    }
    
    for (size_t i = 0; i < 10; ++i) {
        sparse.set(i, i, 1.0);
    }
    
    EXPECT_EQ(dense_like.nnz(), n * n);
    EXPECT_EQ(sparse.nnz(), 10);
    EXPECT_LT(sparse.memory_usage(), dense_like.memory_usage());
}

TEST_F(SparseMatrixTest, SubmatrixExtraction) {
    std::vector<std::tuple<size_t, size_t, double>> triplets = {
        {0, 0, 1.0}, {0, 1, 2.0}, {0, 2, 3.0},
        {1, 0, 4.0}, {1, 1, 5.0}, {1, 2, 6.0},
        {2, 0, 7.0}, {2, 1, 8.0}, {2, 2, 9.0}
    };
    
    SparseMatrix<double> mat(3, 3, triplets);
    
    std::vector<size_t> rows = {0, 2};
    std::vector<size_t> cols = {1, 2};
    
    auto submat = mat.submatrix(rows, cols);
    
    EXPECT_EQ(submat.rows(), 2);
    EXPECT_EQ(submat.cols(), 2);
    EXPECT_EQ(submat(0, 0), 2.0);  // mat(0,1)
    EXPECT_EQ(submat(0, 1), 3.0);  // mat(0,2)
    EXPECT_EQ(submat(1, 0), 8.0);  // mat(2,1)
    EXPECT_EQ(submat(1, 1), 9.0);  // mat(2,2)
}

TEST_F(SparseMatrixTest, FEMStiffnessMatrix) {
    const size_t n_nodes = 9;
    const size_t n_dofs = n_nodes * 2;
    SparseMatrix<double> K(n_dofs, n_dofs);
    
    std::vector<std::vector<size_t>> elements = {
        {0, 1, 3},
        {1, 2, 4},
        {3, 4, 6},
        {4, 5, 7}
    };
    
    for (const auto& elem : elements) {
        for (size_t i = 0; i < elem.size(); ++i) {
            for (size_t j = 0; j < elem.size(); ++j) {
                size_t dof_i = elem[i] * 2;
                size_t dof_j = elem[j] * 2;
                
                K.set(dof_i, dof_j, K(dof_i, dof_j) + 1.0);
                K.set(dof_i + 1, dof_j + 1, K(dof_i + 1, dof_j + 1) + 1.0);
            }
        }
    }
    
    EXPECT_GT(K.nnz(), 0);
    EXPECT_EQ(K.rows(), n_dofs);
    EXPECT_EQ(K.cols(), n_dofs);
    
    for (size_t i = 0; i < n_dofs; i += 2) {
        EXPECT_GT(K(i, i), 0.0);
        EXPECT_GT(K(i + 1, i + 1), 0.0);
    }
}

TEST_F(SparseMatrixTest, CFDConvectionMatrix) {
    const size_t nx = 10, ny = 10;
    const size_t n = nx * ny;
    SparseMatrix<double> C(n, n);
    
    auto idx = [nx](size_t i, size_t j) -> size_t {
        return i * nx + j;
    };
    
    for (size_t i = 1; i < ny - 1; ++i) {
        for (size_t j = 1; j < nx - 1; ++j) {
            size_t center = idx(i, j);
            size_t east = idx(i, j + 1);
            size_t west = idx(i, j - 1);
            size_t north = idx(i + 1, j);
            size_t south = idx(i - 1, j);
            
            C.set(center, east, 0.25);
            C.set(center, west, -0.25);
            C.set(center, north, 0.25);
            C.set(center, south, -0.25);
        }
    }
    
    EXPECT_GT(C.nnz(), 0);
    EXPECT_EQ(C.rows(), n);
    EXPECT_EQ(C.cols(), n);
    
    auto row_sum = C.row_sum();
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(std::abs(row_sum[i]), 0.0, tolerance);
    }
}

TEST_F(SparseMatrixTest, StructuralMechanicsPlateBending) {
    const size_t n_nodes = 25;
    const size_t dofs_per_node = 3;
    const size_t n_dofs = n_nodes * dofs_per_node;
    
    SparseMatrix<double> K(n_dofs, n_dofs);
    
    std::vector<std::array<size_t, 4>> plate_elements;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            size_t n1 = i * 5 + j;
            size_t n2 = i * 5 + j + 1;
            size_t n3 = (i + 1) * 5 + j + 1;
            size_t n4 = (i + 1) * 5 + j;
            plate_elements.push_back({n1, n2, n3, n4});
        }
    }
    
    for (const auto& elem : plate_elements) {
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                for (size_t dof = 0; dof < dofs_per_node; ++dof) {
                    size_t global_i = elem[i] * dofs_per_node + dof;
                    size_t global_j = elem[j] * dofs_per_node + dof;
                    K.set(global_i, global_j, K(global_i, global_j) + 1.0);
                }
            }
        }
    }
    
    EXPECT_GT(K.nnz(), 0);
    EXPECT_EQ(K.rows(), n_dofs);
    EXPECT_EQ(K.cols(), n_dofs);
    
    for (size_t i = 0; i < n_dofs; ++i) {
        EXPECT_GT(K(i, i), 0.0);
    }
}

TEST_F(SparseMatrixTest, ElectromagneticsMaxwellMatrix) {
    const size_t n_edges = 100;
    const size_t n_faces = 80;
    
    SparseMatrix<double> curl_matrix(n_faces, n_edges);
    SparseMatrix<double> mass_matrix(n_edges, n_edges);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    
    for (size_t face = 0; face < n_faces; ++face) {
        for (size_t edge_local = 0; edge_local < 3; ++edge_local) {
            size_t edge_global = (face * 3 + edge_local) % n_edges;
            double orientation = (edge_local % 2 == 0) ? 1.0 : -1.0;
            curl_matrix.set(face, edge_global, orientation);
        }
    }
    
    for (size_t edge = 0; edge < n_edges; ++edge) {
        mass_matrix.set(edge, edge, 1.0);
        if (edge > 0) {
            mass_matrix.set(edge, edge - 1, 0.1);
        }
        if (edge < n_edges - 1) {
            mass_matrix.set(edge, edge + 1, 0.1);
        }
    }
    
    auto curl_T = curl_matrix.transpose();
    auto A = curl_T * curl_matrix + mass_matrix;
    
    EXPECT_EQ(A.rows(), n_edges);
    EXPECT_EQ(A.cols(), n_edges);
    EXPECT_GT(A.nnz(), 0);
    
    for (size_t i = 0; i < n_edges; ++i) {
        EXPECT_GE(A(i, i), 1.0);
    }
}

TEST_F(SparseMatrixTest, MultiphysicsBlockMatrix) {
    const size_t n_velocity = 100;
    const size_t n_pressure = 50;
    const size_t total_dofs = n_velocity + n_pressure;
    
    SparseMatrix<double> A(total_dofs, total_dofs);
    
    for (size_t i = 0; i < n_velocity; ++i) {
        A.set(i, i, 2.0);
        if (i > 0) A.set(i, i - 1, -1.0);
        if (i < n_velocity - 1) A.set(i, i + 1, -1.0);
    }
    
    for (size_t i = 0; i < n_pressure; ++i) {
        size_t row = n_velocity + i;
        A.set(row, row, 1.0);
        
        if (i < n_velocity) {
            A.set(row, i, 0.5);
            A.set(i, row, 0.5);
        }
    }
    
    EXPECT_EQ(A.rows(), total_dofs);
    EXPECT_EQ(A.cols(), total_dofs);
    EXPECT_GT(A.nnz(), 0);
    
    for (size_t i = 0; i < n_velocity; ++i) {
        EXPECT_EQ(A(i, i), 2.0);
    }
    
    for (size_t i = 0; i < n_pressure; ++i) {
        size_t row = n_velocity + i;
        EXPECT_EQ(A(row, row), 1.0);
    }
}

TEST_F(SparseMatrixTest, PerformanceBenchmark) {
    const size_t n = 1000;
    const size_t nnz_target = n * 5;
    
    std::vector<std::tuple<size_t, size_t, double>> triplets;
    triplets.reserve(nnz_target);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> row_dist(0, n - 1);
    std::uniform_int_distribution<size_t> col_dist(0, n - 1);
    std::uniform_real_distribution<double> val_dist(-10.0, 10.0);
    
    for (size_t i = 0; i < nnz_target; ++i) {
        triplets.emplace_back(row_dist(gen), col_dist(gen), val_dist(gen));
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    SparseMatrix<double> mat(n, n, triplets);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto construction_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::vector<double> x(n, 1.0);
    
    start = std::chrono::high_resolution_clock::now();
    auto y = mat * x;
    end = std::chrono::high_resolution_clock::now();
    
    auto multiply_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    EXPECT_GT(mat.nnz(), 0);
    EXPECT_EQ(y.size(), n);
    EXPECT_LT(construction_time.count(), 100000);
    EXPECT_LT(multiply_time.count(), 10000);
}