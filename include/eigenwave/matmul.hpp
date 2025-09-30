#ifndef EIGENWAVE_MATMUL_HPP
#define EIGENWAVE_MATMUL_HPP

#include "tensor.hpp"
#include <numeric>
#include <type_traits>

namespace eigenwave {

// ============================================================================
// Matrix Transpose
// ============================================================================

template<typename T, size_t M, size_t N>
Tensor<T, N, M> transpose(const Tensor<T, M, N>& matrix) {
    Tensor<T, N, M> result;

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(j, i) = matrix(i, j);
        }
    }

    return result;
}

// Note: We can't use T() as a function name since T is often a template parameter
// Users should use transpose() directly

// ============================================================================
// Matrix Multiplication (matmul)
// ============================================================================

// Matrix-matrix multiplication: (M×K) @ (K×N) -> (M×N)
template<typename T, size_t M, size_t K, size_t N>
Tensor<T, M, N> matmul(const Tensor<T, M, K>& a, const Tensor<T, K, N>& b) {
    Tensor<T, M, N> result = Tensor<T, M, N>::zeros();

    // Standard matrix multiplication algorithm
    // TODO: Future optimization with cache blocking, SIMD, etc.
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            T sum = T(0);
            for (size_t k = 0; k < K; ++k) {
                sum += a(i, k) * b(k, j);
            }
            result(i, j) = sum;
        }
    }

    return result;
}

// Matrix-vector multiplication: (M×K) @ (K) -> (M)
template<typename T, size_t M, size_t K>
Tensor<T, M> matmul(const Tensor<T, M, K>& matrix, const Tensor<T, K>& vector) {
    Tensor<T, M> result = Tensor<T, M>::zeros();

    for (size_t i = 0; i < M; ++i) {
        T sum = T(0);
        for (size_t k = 0; k < K; ++k) {
            sum += matrix(i, k) * vector(k);
        }
        result(i) = sum;
    }

    return result;
}

// Vector-matrix multiplication: (K) @ (K×N) -> (N)
template<typename T, size_t K, size_t N>
Tensor<T, N> matmul(const Tensor<T, K>& vector, const Tensor<T, K, N>& matrix) {
    Tensor<T, N> result = Tensor<T, N>::zeros();

    for (size_t j = 0; j < N; ++j) {
        T sum = T(0);
        for (size_t k = 0; k < K; ++k) {
            sum += vector(k) * matrix(k, j);
        }
        result(j) = sum;
    }

    return result;
}

// Operator@ for matrix multiplication (C++23 style, but using regular operator for now)
// Users can use matmul() directly

// ============================================================================
// Dot Product
// ============================================================================

// Helper trait to determine dot product return type
template<typename... Args>
struct dot_result;

// 1D @ 1D -> scalar
template<typename T, size_t N>
struct dot_result<Tensor<T, N>, Tensor<T, N>> {
    using type = T;
};

// 2D @ 1D -> 1D
template<typename T, size_t M, size_t N>
struct dot_result<Tensor<T, M, N>, Tensor<T, N>> {
    using type = Tensor<T, M>;
};

// 1D @ 2D -> 1D
template<typename T, size_t M, size_t N>
struct dot_result<Tensor<T, M>, Tensor<T, M, N>> {
    using type = Tensor<T, N>;
};

// 2D @ 2D -> 2D
template<typename T, size_t M, size_t K, size_t N>
struct dot_result<Tensor<T, M, K>, Tensor<T, K, N>> {
    using type = Tensor<T, M, N>;
};

template<typename... Args>
using dot_result_t = typename dot_result<Args...>::type;

// Dot product implementations
// 1D · 1D -> scalar
template<typename T, size_t N>
T dot(const Tensor<T, N>& a, const Tensor<T, N>& b) {
    T result = T(0);
    for (size_t i = 0; i < N; ++i) {
        result += a(i) * b(i);
    }
    return result;
}

// 2D · 1D -> 1D (matrix-vector multiplication)
template<typename T, size_t M, size_t N>
Tensor<T, M> dot(const Tensor<T, M, N>& matrix, const Tensor<T, N>& vector) {
    return matmul(matrix, vector);
}

// 1D · 2D -> 1D (vector-matrix multiplication)
template<typename T, size_t M, size_t N>
Tensor<T, N> dot(const Tensor<T, M>& vector, const Tensor<T, M, N>& matrix) {
    return matmul(vector, matrix);
}

// 2D · 2D -> 2D (matrix multiplication)
template<typename T, size_t M, size_t K, size_t N>
Tensor<T, M, N> dot(const Tensor<T, M, K>& a, const Tensor<T, K, N>& b) {
    return matmul(a, b);
}

// ============================================================================
// Inner Product (generalized dot product)
// ============================================================================

// Inner product - sum of element-wise multiplication
template<typename T, size_t... Dims>
T inner(const Tensor<T, Dims...>& a, const Tensor<T, Dims...>& b) {
    T result = T(0);

    auto it_a = a.begin();
    auto it_b = b.begin();

    for (; it_a != a.end(); ++it_a, ++it_b) {
        result += (*it_a) * (*it_b);
    }

    return result;
}

// ============================================================================
// Outer Product
// ============================================================================

// Outer product of two 1D vectors: (M) ⊗ (N) -> (M×N)
template<typename T, size_t M, size_t N>
Tensor<T, M, N> outer(const Tensor<T, M>& a, const Tensor<T, N>& b) {
    Tensor<T, M, N> result;

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = a(i) * b(j);
        }
    }

    return result;
}

// ============================================================================
// Matrix-specific utilities
// ============================================================================

// Create identity matrix
template<typename T, size_t N>
Tensor<T, N, N> eye() {
    Tensor<T, N, N> result = Tensor<T, N, N>::zeros();
    for (size_t i = 0; i < N; ++i) {
        result(i, i) = T(1);
    }
    return result;
}

// Create identity matrix with explicit type
template<typename T, size_t N>
Tensor<T, N, N> identity() {
    return eye<T, N>();
}

// Create diagonal matrix from vector
template<typename T, size_t N>
Tensor<T, N, N> diag(const Tensor<T, N>& vec) {
    Tensor<T, N, N> result = Tensor<T, N, N>::zeros();
    for (size_t i = 0; i < N; ++i) {
        result(i, i) = vec(i);
    }
    return result;
}

// Extract diagonal from matrix
template<typename T, size_t M, size_t N>
Tensor<T, (M < N ? M : N)> diag(const Tensor<T, M, N>& matrix) {
    constexpr size_t min_dim = (M < N ? M : N);
    Tensor<T, min_dim> result;

    for (size_t i = 0; i < min_dim; ++i) {
        result(i) = matrix(i, i);
    }

    return result;
}

// Trace (sum of diagonal elements)
template<typename T, size_t N>
T trace(const Tensor<T, N, N>& matrix) {
    T result = T(0);
    for (size_t i = 0; i < N; ++i) {
        result += matrix(i, i);
    }
    return result;
}

// ============================================================================
// Validation helpers for compile-time dimension checking
// ============================================================================

// Check if matrix multiplication is valid
template<typename A, typename B>
struct can_matmul : std::false_type {};

template<typename T, size_t M, size_t K1, size_t K2, size_t N>
struct can_matmul<Tensor<T, M, K1>, Tensor<T, K2, N>>
    : std::conditional_t<K1 == K2, std::true_type, std::false_type> {};

template<typename T, size_t M, size_t K1, size_t K2>
struct can_matmul<Tensor<T, M, K1>, Tensor<T, K2>>
    : std::conditional_t<K1 == K2, std::true_type, std::false_type> {};

template<typename T, size_t K1, size_t K2, size_t N>
struct can_matmul<Tensor<T, K1>, Tensor<T, K2, N>>
    : std::conditional_t<K1 == K2, std::true_type, std::false_type> {};

template<typename A, typename B>
inline constexpr bool can_matmul_v = can_matmul<A, B>::value;

} // namespace eigenwave

#endif // EIGENWAVE_MATMUL_HPP