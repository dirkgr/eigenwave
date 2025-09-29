#ifndef EIGENWAVE_BROADCASTING_HPP
#define EIGENWAVE_BROADCASTING_HPP

#include "tensor.hpp"
#include "broadcast_view.hpp"
#include <algorithm>
#include <type_traits>
#include <utility>

namespace eigenwave {

// Note: This file implements NumPy-style broadcasting for tensors.
// Broadcasting rules:
// 1. Dimensions are aligned from the right
// 2. Dimensions of size 1 broadcast to any size
// 3. Missing dimensions are treated as size 1

// Broadcasting scalar to tensor (fills tensor with scalar value)
template<typename T, size_t... Dims>
Tensor<T, Dims...> broadcast_scalar(T scalar, const Tensor<T, Dims...>& shape_tensor) {
    return Tensor<T, Dims...>(scalar);
}

// Broadcast addition for vector and matrix
// Vector [D2] broadcasts to [D1, D2] (NumPy-style: aligned from right)
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

// Note: For explicit broadcasting, use the operators above or create a BroadcastView.
// To broadcast a column vector (e.g., [D1] to [D1, D2]), first reshape it to [D1, 1]
// using expand_dims, then use the broadcasting operators.

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