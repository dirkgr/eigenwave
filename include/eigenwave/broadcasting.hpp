#ifndef EIGENWAVE_BROADCASTING_HPP
#define EIGENWAVE_BROADCASTING_HPP

#include "tensor.hpp"
#include <algorithm>
#include <type_traits>
#include <utility>

namespace eigenwave {

// Broadcast binary operation for scalars
template<typename T, typename BinaryOp>
Tensor<T, 2, 2> broadcast_binary_op(const Tensor<T, 1>& vec,
                                    const Tensor<T, 2, 2>& mat,
                                    BinaryOp op,
                                    bool vec_first = true) {
    Tensor<T, 2, 2> result;

    // Broadcasting a vector across rows
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            if (vec_first) {
                result(i, j) = op(vec(0), mat(i, j));
            } else {
                result(i, j) = op(mat(i, j), vec(0));
            }
        }
    }

    return result;
}

// Specialized broadcasting for common cases

// Broadcasting scalar to tensor
template<typename T, size_t... Dims>
Tensor<T, Dims...> broadcast_scalar(T scalar, const Tensor<T, Dims...>& shape_tensor) {
    return Tensor<T, Dims...>(scalar);
}

// Broadcasting 1D to 2D (along columns)
template<typename T, size_t D1, size_t D2>
Tensor<T, D1, D2> broadcast_1d_to_2d_cols(const Tensor<T, D2>& vec) {
    Tensor<T, D1, D2> result;
    for (size_t i = 0; i < D1; ++i) {
        for (size_t j = 0; j < D2; ++j) {
            result(i, j) = vec(j);
        }
    }
    return result;
}

// Broadcasting 1D to 2D (along rows)
template<typename T, size_t D1, size_t D2>
Tensor<T, D1, D2> broadcast_1d_to_2d_rows(const Tensor<T, D1>& vec) {
    Tensor<T, D1, D2> result;
    for (size_t i = 0; i < D1; ++i) {
        for (size_t j = 0; j < D2; ++j) {
            result(i, j) = vec(i);
        }
    }
    return result;
}

// Broadcast addition for vector and matrix
template<typename T, size_t D1, size_t D2>
Tensor<T, D1, D2> operator+(const Tensor<T, D2>& vec, const Tensor<T, D1, D2>& mat) {
    Tensor<T, D1, D2> result;
    for (size_t i = 0; i < D1; ++i) {
        for (size_t j = 0; j < D2; ++j) {
            result(i, j) = vec(j) + mat(i, j);
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
    Tensor<T, D1, D2> result;
    for (size_t i = 0; i < D1; ++i) {
        for (size_t j = 0; j < D2; ++j) {
            result(i, j) = mat(i, j) - vec(j);
        }
    }
    return result;
}

template<typename T, size_t D1, size_t D2>
Tensor<T, D1, D2> operator-(const Tensor<T, D2>& vec, const Tensor<T, D1, D2>& mat) {
    Tensor<T, D1, D2> result;
    for (size_t i = 0; i < D1; ++i) {
        for (size_t j = 0; j < D2; ++j) {
            result(i, j) = vec(j) - mat(i, j);
        }
    }
    return result;
}

// Broadcast multiplication for vector and matrix
template<typename T, size_t D1, size_t D2>
Tensor<T, D1, D2> operator*(const Tensor<T, D2>& vec, const Tensor<T, D1, D2>& mat) {
    Tensor<T, D1, D2> result;
    for (size_t i = 0; i < D1; ++i) {
        for (size_t j = 0; j < D2; ++j) {
            result(i, j) = vec(j) * mat(i, j);
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
    Tensor<T, D1, D2> result;
    for (size_t i = 0; i < D1; ++i) {
        for (size_t j = 0; j < D2; ++j) {
            result(i, j) = mat(i, j) / vec(j);
        }
    }
    return result;
}

template<typename T, size_t D1, size_t D2>
Tensor<T, D1, D2> operator/(const Tensor<T, D2>& vec, const Tensor<T, D1, D2>& mat) {
    Tensor<T, D1, D2> result;
    for (size_t i = 0; i < D1; ++i) {
        for (size_t j = 0; j < D2; ++j) {
            result(i, j) = vec(j) / mat(i, j);
        }
    }
    return result;
}

// Helper function to explicitly broadcast a tensor to a new shape
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