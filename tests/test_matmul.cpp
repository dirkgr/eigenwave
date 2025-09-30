#include <gtest/gtest.h>
#include <eigenwave/tensor.hpp>
#include <eigenwave/matmul.hpp>
#include <cmath>

using namespace eigenwave;

// Test fixture for matrix multiplication tests
class MatmulTest : public ::testing::Test {
protected:
    void SetUp() override {}

    // Helper for floating point comparison
    template<typename T>
    bool near(T a, T b, T eps = 1e-6) {
        return std::abs(a - b) < eps;
    }
};

// ============================================================================
// Transpose Tests
// ============================================================================

TEST_F(MatmulTest, Transpose2x3) {
    Tensor<int, 2, 3> matrix{1, 2, 3,
                             4, 5, 6};

    auto transposed = transpose(matrix);
    static_assert(std::is_same_v<decltype(transposed), Tensor<int, 3, 2>>);

    EXPECT_EQ(transposed(0, 0), 1);
    EXPECT_EQ(transposed(0, 1), 4);
    EXPECT_EQ(transposed(1, 0), 2);
    EXPECT_EQ(transposed(1, 1), 5);
    EXPECT_EQ(transposed(2, 0), 3);
    EXPECT_EQ(transposed(2, 1), 6);
}

TEST_F(MatmulTest, TransposeSquare) {
    Tensor<float, 3, 3> matrix{1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f,
                               7.0f, 8.0f, 9.0f};

    auto transposed = transpose(matrix);
    static_assert(std::is_same_v<decltype(transposed), Tensor<float, 3, 3>>);

    EXPECT_FLOAT_EQ(transposed(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(transposed(0, 1), 4.0f);
    EXPECT_FLOAT_EQ(transposed(0, 2), 7.0f);
    EXPECT_FLOAT_EQ(transposed(1, 0), 2.0f);
    EXPECT_FLOAT_EQ(transposed(1, 1), 5.0f);
    EXPECT_FLOAT_EQ(transposed(1, 2), 8.0f);
}

TEST_F(MatmulTest, DoubleTranspose) {
    Tensor<int, 2, 3> original{1, 2, 3,
                               4, 5, 6};

    auto double_transposed = transpose(transpose(original));
    static_assert(std::is_same_v<decltype(double_transposed), Tensor<int, 2, 3>>);

    // Should be identical to original
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(double_transposed(i, j), original(i, j));
        }
    }
}

// ============================================================================
// Matrix Multiplication Tests
// ============================================================================

TEST_F(MatmulTest, MatrixMatrixMultiplication) {
    Tensor<int, 2, 3> a{1, 2, 3,
                        4, 5, 6};

    Tensor<int, 3, 2> b{7,  8,
                        9,  10,
                        11, 12};

    auto result = matmul(a, b);
    static_assert(std::is_same_v<decltype(result), Tensor<int, 2, 2>>);

    // Expected: [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12]
    //           [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]
    //         = [58, 64]
    //           [139, 154]
    EXPECT_EQ(result(0, 0), 58);
    EXPECT_EQ(result(0, 1), 64);
    EXPECT_EQ(result(1, 0), 139);
    EXPECT_EQ(result(1, 1), 154);
}

TEST_F(MatmulTest, SquareMatrixMultiplication) {
    Tensor<float, 2, 2> a{1.0f, 2.0f,
                          3.0f, 4.0f};

    Tensor<float, 2, 2> b{5.0f, 6.0f,
                          7.0f, 8.0f};

    auto result = matmul(a, b);

    // [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
    // [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
    EXPECT_FLOAT_EQ(result(0, 0), 19.0f);
    EXPECT_FLOAT_EQ(result(0, 1), 22.0f);
    EXPECT_FLOAT_EQ(result(1, 0), 43.0f);
    EXPECT_FLOAT_EQ(result(1, 1), 50.0f);
}

TEST_F(MatmulTest, MatrixVectorMultiplication) {
    Tensor<float, 3, 2> matrix{1.0f, 2.0f,
                               3.0f, 4.0f,
                               5.0f, 6.0f};

    Tensor<float, 2> vector{7.0f, 8.0f};

    auto result = matmul(matrix, vector);
    static_assert(std::is_same_v<decltype(result), Tensor<float, 3>>);

    // [1*7 + 2*8] = [23]
    // [3*7 + 4*8] = [53]
    // [5*7 + 6*8] = [83]
    EXPECT_FLOAT_EQ(result(0), 23.0f);
    EXPECT_FLOAT_EQ(result(1), 53.0f);
    EXPECT_FLOAT_EQ(result(2), 83.0f);
}

TEST_F(MatmulTest, VectorMatrixMultiplication) {
    Tensor<float, 2> vector{1.0f, 2.0f};

    Tensor<float, 2, 3> matrix{3.0f, 4.0f, 5.0f,
                               6.0f, 7.0f, 8.0f};

    auto result = matmul(vector, matrix);
    static_assert(std::is_same_v<decltype(result), Tensor<float, 3>>);

    // [1*3 + 2*6, 1*4 + 2*7, 1*5 + 2*8] = [15, 18, 21]
    EXPECT_FLOAT_EQ(result(0), 15.0f);
    EXPECT_FLOAT_EQ(result(1), 18.0f);
    EXPECT_FLOAT_EQ(result(2), 21.0f);
}

TEST_F(MatmulTest, IdentityMatrix) {
    auto identity = eye<float, 3>();

    Tensor<float, 3, 2> matrix{1.0f, 2.0f,
                               3.0f, 4.0f,
                               5.0f, 6.0f};

    auto result = matmul(identity, matrix);

    // Identity multiplication should preserve the matrix
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(result(i, j), matrix(i, j));
        }
    }
}

// ============================================================================
// Dot Product Tests
// ============================================================================

TEST_F(MatmulTest, DotProduct1D) {
    Tensor<float, 3> a{1.0f, 2.0f, 3.0f};
    Tensor<float, 3> b{4.0f, 5.0f, 6.0f};

    float result = dot(a, b);

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    EXPECT_FLOAT_EQ(result, 32.0f);
}

TEST_F(MatmulTest, DotProduct2D) {
    Tensor<int, 2, 3> a{1, 2, 3,
                        4, 5, 6};

    Tensor<int, 3, 2> b{7,  8,
                        9,  10,
                        11, 12};

    auto result = dot(a, b);
    static_assert(std::is_same_v<decltype(result), Tensor<int, 2, 2>>);

    // Same as matrix multiplication
    EXPECT_EQ(result(0, 0), 58);
    EXPECT_EQ(result(0, 1), 64);
    EXPECT_EQ(result(1, 0), 139);
    EXPECT_EQ(result(1, 1), 154);
}

TEST_F(MatmulTest, DotMatrixVector) {
    Tensor<double, 2, 3> matrix{1.0, 2.0, 3.0,
                                4.0, 5.0, 6.0};

    Tensor<double, 3> vector{0.5, 1.0, 1.5};

    auto result = dot(matrix, vector);
    static_assert(std::is_same_v<decltype(result), Tensor<double, 2>>);

    // [1*0.5 + 2*1 + 3*1.5] = [7]
    // [4*0.5 + 5*1 + 6*1.5] = [16]
    EXPECT_DOUBLE_EQ(result(0), 7.0);
    EXPECT_DOUBLE_EQ(result(1), 16.0);
}

// ============================================================================
// Inner and Outer Product Tests
// ============================================================================

TEST_F(MatmulTest, InnerProduct) {
    Tensor<float, 2, 3> a{1.0f, 2.0f, 3.0f,
                          4.0f, 5.0f, 6.0f};

    Tensor<float, 2, 3> b{7.0f, 8.0f, 9.0f,
                          10.0f, 11.0f, 12.0f};

    float result = inner(a, b);

    // Sum of element-wise multiplication
    // 1*7 + 2*8 + 3*9 + 4*10 + 5*11 + 6*12
    // = 7 + 16 + 27 + 40 + 55 + 72 = 217
    EXPECT_FLOAT_EQ(result, 217.0f);
}

TEST_F(MatmulTest, OuterProduct) {
    Tensor<int, 3> a{1, 2, 3};
    Tensor<int, 4> b{4, 5, 6, 7};

    auto result = outer(a, b);
    static_assert(std::is_same_v<decltype(result), Tensor<int, 3, 4>>);

    // Outer product creates matrix where result(i,j) = a(i) * b(j)
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_EQ(result(i, j), a(i) * b(j));
        }
    }
}

// ============================================================================
// Matrix Utilities Tests
// ============================================================================

TEST_F(MatmulTest, DiagonalMatrix) {
    Tensor<float, 3> vec{1.0f, 2.0f, 3.0f};

    auto diag_matrix = diag(vec);
    static_assert(std::is_same_v<decltype(diag_matrix), Tensor<float, 3, 3>>);

    EXPECT_FLOAT_EQ(diag_matrix(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(diag_matrix(1, 1), 2.0f);
    EXPECT_FLOAT_EQ(diag_matrix(2, 2), 3.0f);

    // Off-diagonal elements should be zero
    EXPECT_FLOAT_EQ(diag_matrix(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(diag_matrix(0, 2), 0.0f);
    EXPECT_FLOAT_EQ(diag_matrix(1, 0), 0.0f);
}

TEST_F(MatmulTest, ExtractDiagonal) {
    Tensor<int, 3, 3> matrix{1, 2, 3,
                             4, 5, 6,
                             7, 8, 9};

    auto diagonal = diag(matrix);
    static_assert(std::is_same_v<decltype(diagonal), Tensor<int, 3>>);

    EXPECT_EQ(diagonal(0), 1);
    EXPECT_EQ(diagonal(1), 5);
    EXPECT_EQ(diagonal(2), 9);
}

TEST_F(MatmulTest, Trace) {
    Tensor<float, 3, 3> matrix{1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f,
                               7.0f, 8.0f, 9.0f};

    float tr = trace(matrix);

    // Sum of diagonal: 1 + 5 + 9 = 15
    EXPECT_FLOAT_EQ(tr, 15.0f);
}

// ============================================================================
// Chain Operations Tests
// ============================================================================

TEST_F(MatmulTest, ChainedMatrixMultiplication) {
    Tensor<float, 2, 3> a{1.0f, 2.0f, 3.0f,
                          4.0f, 5.0f, 6.0f};

    Tensor<float, 3, 2> b{1.0f, 0.0f,
                          0.0f, 1.0f,
                          1.0f, 1.0f};

    Tensor<float, 2, 2> c{2.0f, 0.0f,
                          0.0f, 3.0f};

    // (A × B) × C
    auto ab = matmul(a, b);
    auto result = matmul(ab, c);

    // Verify result dimensions
    static_assert(std::is_same_v<decltype(result), Tensor<float, 2, 2>>);

    // Manual calculation:
    // AB = [[4, 5], [10, 11]]
    // (AB)C = [[8, 15], [20, 33]]
    EXPECT_FLOAT_EQ(result(0, 0), 8.0f);
    EXPECT_FLOAT_EQ(result(0, 1), 15.0f);
    EXPECT_FLOAT_EQ(result(1, 0), 20.0f);
    EXPECT_FLOAT_EQ(result(1, 1), 33.0f);
}

TEST_F(MatmulTest, TransposeMatmul) {
    Tensor<float, 2, 3> a{1.0f, 2.0f, 3.0f,
                          4.0f, 5.0f, 6.0f};

    // A^T × A should give 3×3 matrix
    auto at_a = matmul(transpose(a), a);
    static_assert(std::is_same_v<decltype(at_a), Tensor<float, 3, 3>>);

    // A × A^T should give 2×2 matrix
    auto a_at = matmul(a, transpose(a));
    static_assert(std::is_same_v<decltype(a_at), Tensor<float, 2, 2>>);

    // Verify symmetry of A × A^T
    EXPECT_FLOAT_EQ(a_at(0, 1), a_at(1, 0));
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(MatmulTest, SingleElementMatrices) {
    Tensor<int, 1, 1> a{5};
    Tensor<int, 1, 1> b{3};

    auto result = matmul(a, b);
    static_assert(std::is_same_v<decltype(result), Tensor<int, 1, 1>>);

    EXPECT_EQ(result(0, 0), 15);
}

TEST_F(MatmulTest, ZeroMatrix) {
    auto zeros = Tensor<float, 2, 3>::zeros();
    Tensor<float, 3, 2> other{1.0f, 2.0f,
                              3.0f, 4.0f,
                              5.0f, 6.0f};

    auto result = matmul(zeros, other);

    // Result should be all zeros
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(result(i, j), 0.0f);
        }
    }
}

// Compile-time dimension checking tests
// These should not compile (uncomment to verify):
/*
TEST_F(MatmulTest, InvalidDimensions) {
    Tensor<int, 2, 3> a;
    Tensor<int, 2, 3> b;

    // This should fail: inner dimensions don't match
    // auto result = matmul(a, b);
}
*/