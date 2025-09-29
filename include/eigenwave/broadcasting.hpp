#ifndef EIGENWAVE_BROADCASTING_HPP
#define EIGENWAVE_BROADCASTING_HPP

#include "tensor.hpp"
#include "broadcast_view.hpp"
#include <algorithm>
#include <type_traits>
#include <utility>

namespace eigenwave {

// Note: This file provides convenience functions for broadcasting operations.
// These functions now use O(1) broadcast views internally and only materialize
// when returning the final result.

// Broadcasting scalar to tensor (creates a tensor filled with the scalar value)
template<typename T, size_t... Dims>
Tensor<T, Dims...> broadcast_scalar(T scalar, const Tensor<T, Dims...>& shape_tensor) {
    return Tensor<T, Dims...>(scalar);
}

// Broadcasting 1D to 2D (along columns) - each row is identical to the vector
// Uses O(1) broadcast view internally
template<typename T, size_t D1, size_t D2>
Tensor<T, D1, D2> broadcast_1d_to_2d_cols(const Tensor<T, D2>& vec) {
    auto view = broadcast_vector_to_matrix<T, D1, D2>(vec);
    return view.materialize();
}

// Broadcasting 1D to 2D (along rows) - each column is identical
// Note: This broadcasts a column vector [D1] to [D1, D2]
template<typename T, size_t D1, size_t D2>
Tensor<T, D1, D2> broadcast_1d_to_2d_rows(const Tensor<T, D1>& vec) {
    // This is a special case that doesn't fit standard broadcasting rules
    // We keep the simple implementation for clarity
    Tensor<T, D1, D2> result;
    for (size_t i = 0; i < D1; ++i) {
        for (size_t j = 0; j < D2; ++j) {
            result(i, j) = vec(i);
        }
    }
    return result;
}

// Broadcast addition for vector and matrix
// Uses O(1) broadcast view internally
template<typename T, size_t D1, size_t D2>
Tensor<T, D1, D2> operator+(const Tensor<T, D2>& vec, const Tensor<T, D1, D2>& mat) {
    auto broadcast_view = broadcast_vector_to_matrix<T, D1, D2>(vec);
    Tensor<T, D1, D2> result;
    for (size_t i = 0; i < D1; ++i) {
        for (size_t j = 0; j < D2; ++j) {
            result(i, j) = broadcast_view(i, j) + mat(i, j);
        }
    }
    return result;
}

template<typename T, size_t D1, size_t D2>
Tensor<T, D1, D2> operator+(const Tensor<T, D1, D2>& mat, const Tensor<T, D2>& vec) {
    return vec + mat;
}

// Broadcast subtraction for vector and matrix
// Uses O(1) broadcast view internally
template<typename T, size_t D1, size_t D2>
Tensor<T, D1, D2> operator-(const Tensor<T, D1, D2>& mat, const Tensor<T, D2>& vec) {
    auto broadcast_view = broadcast_vector_to_matrix<T, D1, D2>(vec);
    Tensor<T, D1, D2> result;
    for (size_t i = 0; i < D1; ++i) {
        for (size_t j = 0; j < D2; ++j) {
            result(i, j) = mat(i, j) - broadcast_view(i, j);
        }
    }
    return result;
}

template<typename T, size_t D1, size_t D2>
Tensor<T, D1, D2> operator-(const Tensor<T, D2>& vec, const Tensor<T, D1, D2>& mat) {
    auto broadcast_view = broadcast_vector_to_matrix<T, D1, D2>(vec);
    Tensor<T, D1, D2> result;
    for (size_t i = 0; i < D1; ++i) {
        for (size_t j = 0; j < D2; ++j) {
            result(i, j) = broadcast_view(i, j) - mat(i, j);
        }
    }
    return result;
}

// Broadcast multiplication for vector and matrix
// Uses O(1) broadcast view internally
template<typename T, size_t D1, size_t D2>
Tensor<T, D1, D2> operator*(const Tensor<T, D2>& vec, const Tensor<T, D1, D2>& mat) {
    auto broadcast_view = broadcast_vector_to_matrix<T, D1, D2>(vec);
    Tensor<T, D1, D2> result;
    for (size_t i = 0; i < D1; ++i) {
        for (size_t j = 0; j < D2; ++j) {
            result(i, j) = broadcast_view(i, j) * mat(i, j);
        }
    }
    return result;
}

template<typename T, size_t D1, size_t D2>
Tensor<T, D1, D2> operator*(const Tensor<T, D1, D2>& mat, const Tensor<T, D2>& vec) {
    return vec * mat;
}

// Broadcast division for vector and matrix
// Uses O(1) broadcast view internally
template<typename T, size_t D1, size_t D2>
Tensor<T, D1, D2> operator/(const Tensor<T, D1, D2>& mat, const Tensor<T, D2>& vec) {
    auto broadcast_view = broadcast_vector_to_matrix<T, D1, D2>(vec);
    Tensor<T, D1, D2> result;
    for (size_t i = 0; i < D1; ++i) {
        for (size_t j = 0; j < D2; ++j) {
            result(i, j) = mat(i, j) / broadcast_view(i, j);
        }
    }
    return result;
}

template<typename T, size_t D1, size_t D2>
Tensor<T, D1, D2> operator/(const Tensor<T, D2>& vec, const Tensor<T, D1, D2>& mat) {
    auto broadcast_view = broadcast_vector_to_matrix<T, D1, D2>(vec);
    Tensor<T, D1, D2> result;
    for (size_t i = 0; i < D1; ++i) {
        for (size_t j = 0; j < D2; ++j) {
            result(i, j) = broadcast_view(i, j) / mat(i, j);
        }
    }
    return result;
}

// Helper function to explicitly broadcast a vector to a square matrix
// Each row is identical to the input vector
template<typename T, size_t D>
Tensor<T, D, D> broadcast_to_square(const Tensor<T, D>& vec) {
    return broadcast_1d_to_2d_cols<T, D, D>(vec);
}

// Reshape function for compatible dimensions (total size must match)
template<typename T, size_t... NewDims, size_t... OldDims>
Tensor<T, NewDims...> reshape(const Tensor<T, OldDims...>& tensor) {
    static_assert(product_of_dims_v<OldDims...> == product_of_dims_v<NewDims...>,
                  "Total size must remain the same when reshaping");

    Tensor<T, NewDims...> result;
    std::copy(tensor.begin(), tensor.end(), result.begin());
    return result;
}

// Expand dimensions (add dimension of size 1)
template<typename T, size_t... Dims>
Tensor<T, 1, Dims...> expand_dims_front(const Tensor<T, Dims...>& tensor) {
    Tensor<T, 1, Dims...> result;
    std::copy(tensor.begin(), tensor.end(), result.begin());
    return result;
}

template<typename T, size_t... Dims>
Tensor<T, Dims..., 1> expand_dims_back(const Tensor<T, Dims...>& tensor) {
    Tensor<T, Dims..., 1> result;
    std::copy(tensor.begin(), tensor.end(), result.begin());
    return result;
}

// Squeeze dimensions (remove dimensions of size 1)
template<typename T, size_t... Dims>
auto squeeze(const Tensor<T, 1, Dims...>& tensor) {
    Tensor<T, Dims...> result;
    std::copy(tensor.begin(), tensor.end(), result.begin());
    return result;
}

template<typename T, size_t... Dims>
auto squeeze(const Tensor<T, Dims..., 1>& tensor) {
    Tensor<T, Dims...> result;
    std::copy(tensor.begin(), tensor.end(), result.begin());
    return result;
}

} // namespace eigenwave

#endif // EIGENWAVE_BROADCASTING_HPP