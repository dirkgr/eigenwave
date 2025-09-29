#ifndef EIGENWAVE_BROADCASTING_HPP
#define EIGENWAVE_BROADCASTING_HPP

#include "tensor.hpp"
#include <array>
#include <algorithm>
#include <type_traits>
#include <utility>

namespace eigenwave {

// ============================================================================
// BroadcastView: O(1) broadcasting implementation
// ============================================================================

// BroadcastView: A view that virtually broadcasts a tensor to a larger shape
// without copying data. Uses stride tricks to repeat elements.
template<typename T, typename OrigShape, typename BroadcastShape>
class BroadcastView;

// Specialization for index sequences
template<typename T, size_t... OrigDims, size_t... BroadcastDims>
class BroadcastView<T, std::index_sequence<OrigDims...>, std::index_sequence<BroadcastDims...>> {
public:
    using value_type = T;
    static constexpr size_t orig_ndim = sizeof...(OrigDims);
    static constexpr size_t broadcast_ndim = sizeof...(BroadcastDims);
    static constexpr std::array<size_t, orig_ndim> orig_shape = {OrigDims...};
    static constexpr std::array<size_t, broadcast_ndim> broadcast_shape = {BroadcastDims...};

private:
    const T* data_ptr_;
    std::array<size_t, broadcast_ndim> effective_strides_;

    // Compute effective strides for broadcasting
    void compute_broadcast_strides() {
        // Special case for vector to matrix broadcasting [D2] -> [D1, D2]
        if constexpr (orig_ndim == 1 && broadcast_ndim == 2) {
            // First dimension (rows): stride 0 means same vector for each row
            // Second dimension (cols): stride 1 for normal vector progression
            effective_strides_[0] = 0;
            effective_strides_[1] = 1;
        } else {
            // General case - align dimensions from the right
            // Fill with zeros first
            for (size_t i = 0; i < broadcast_ndim; ++i) {
                effective_strides_[i] = 0;
            }

            // Calculate strides for dimensions that exist in original
            if constexpr (orig_ndim > 0) {
                size_t stride = 1;
                int orig_idx = orig_ndim - 1;
                int broadcast_idx = broadcast_ndim - 1;

                while (orig_idx >= 0 && broadcast_idx >= 0) {
                    if (orig_shape[orig_idx] == broadcast_shape[broadcast_idx]) {
                        effective_strides_[broadcast_idx] = stride;
                        stride *= orig_shape[orig_idx];
                    } else if (orig_shape[orig_idx] == 1) {
                        // Broadcasting dimension of size 1 - stride is 0
                        effective_strides_[broadcast_idx] = 0;
                    }
                    orig_idx--;
                    broadcast_idx--;
                }
            }
        }
    }

public:
    // Constructor from tensor
    explicit BroadcastView(const Tensor<T, OrigDims...>& tensor)
        : data_ptr_(tensor.data()) {
        compute_broadcast_strides();
    }

    // Element access (read-only)
    template<typename... Indices>
    const T& operator()(Indices... indices) const {
        static_assert(sizeof...(indices) == broadcast_ndim,
                      "Number of indices must match broadcast dimensions");

        size_t idx_array[broadcast_ndim] = {static_cast<size_t>(indices)...};

        // Calculate flat index using broadcast strides
        size_t flat_idx = 0;
        for (size_t i = 0; i < broadcast_ndim; ++i) {
            flat_idx += idx_array[i] * effective_strides_[i];
        }

        return data_ptr_[flat_idx];
    }

    // Convert broadcast view to an actual tensor (performs copy)
    Tensor<T, BroadcastDims...> materialize() const {
        Tensor<T, BroadcastDims...> result;

        // Simple iterative approach for all dimensions
        size_t total_elements = 1;
        for (size_t dim : broadcast_shape) {
            total_elements *= dim;
        }

        // Iterate through all output elements
        for (size_t flat_idx = 0; flat_idx < total_elements; ++flat_idx) {
            // Convert flat index to multi-dimensional indices
            std::array<size_t, broadcast_ndim> indices{};
            size_t temp_idx = flat_idx;

            for (int i = broadcast_ndim - 1; i >= 0; --i) {
                indices[i] = temp_idx % broadcast_shape[i];
                temp_idx /= broadcast_shape[i];
            }

            // Calculate source index using broadcast strides
            size_t src_idx = 0;
            for (size_t i = 0; i < broadcast_ndim; ++i) {
                src_idx += indices[i] * effective_strides_[i];
            }

            result.data()[flat_idx] = data_ptr_[src_idx];
        }

        return result;
    }
};

// Helper for scalar broadcasting
template<typename T, size_t... Dims>
class ScalarBroadcastView {
    T value_;

public:
    explicit ScalarBroadcastView(T value) : value_(value) {}

    template<typename... Indices>
    T operator()(Indices... indices) const {
        return value_;
    }

    Tensor<T, Dims...> materialize() const {
        return Tensor<T, Dims...>(value_);
    }
};

// ============================================================================
// Broadcasting helper functions
// ============================================================================

// Helper function to create broadcast view for vector -> matrix broadcasting
template<typename T, size_t D1, size_t D2>
auto broadcast_vector_to_matrix(const Tensor<T, D2>& vec) {
    // Broadcasting [D2] -> [D1, D2]
    // Each row will be identical to the vector
    return BroadcastView<T, std::index_sequence<D2>, std::index_sequence<D1, D2>>(vec);
}

// Create a broadcast view that virtually expands dimensions
template<typename T, size_t... Dims>
auto broadcast_expand_dims_front(const Tensor<T, Dims...>& tensor) {
    return BroadcastView<T, std::index_sequence<Dims...>, std::index_sequence<1, Dims...>>(tensor);
}

// Broadcasting scalar to tensor
template<typename T, size_t... Dims>
Tensor<T, Dims...> broadcast_scalar(T scalar, const Tensor<T, Dims...>& shape_tensor) {
    return Tensor<T, Dims...>(scalar);
}

// ============================================================================
// Broadcasting compatibility checking
// ============================================================================

// Helper to check broadcasting compatibility using index sequences
template<typename Shape1, typename Shape2>
struct can_broadcast_helper;

template<size_t... Dims1, size_t... Dims2>
struct can_broadcast_helper<std::index_sequence<Dims1...>, std::index_sequence<Dims2...>> {
    static constexpr bool value() {
        constexpr size_t shape1[] = {Dims1...};
        constexpr size_t shape2[] = {Dims2...};
        constexpr size_t ndim1 = sizeof...(Dims1);
        constexpr size_t ndim2 = sizeof...(Dims2);

        if constexpr (ndim1 == 0 || ndim2 == 0) {
            return true;
        }

        // Check dimensions from right to left
        size_t min_ndim = (ndim1 < ndim2) ? ndim1 : ndim2;
        for (size_t k = 0; k < min_ndim; ++k) {
            size_t dim1 = shape1[ndim1 - 1 - k];
            size_t dim2 = shape2[ndim2 - 1 - k];
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                return false;
            }
        }
        return true;
    }
};

// Check if broadcasting is possible between two shapes
template<size_t... Dims1>
struct can_broadcast_impl {
    template<size_t... Dims2>
    static constexpr bool with() {
        return can_broadcast_helper<
            std::index_sequence<Dims1...>,
            std::index_sequence<Dims2...>
        >::value();
    }
};

// Compute broadcast result shape
template<typename Shape1, typename Shape2>
struct broadcast_shape_helper;

template<size_t... Dims1, size_t... Dims2>
struct broadcast_shape_helper<std::index_sequence<Dims1...>, std::index_sequence<Dims2...>> {
    static constexpr auto compute() {
        constexpr size_t shape1[] = {Dims1...};
        constexpr size_t shape2[] = {Dims2...};
        constexpr size_t ndim1 = sizeof...(Dims1);
        constexpr size_t ndim2 = sizeof...(Dims2);
        constexpr size_t result_ndim = (ndim1 > ndim2) ? ndim1 : ndim2;

        std::array<size_t, result_ndim> result{};

        // Fill from the right
        for (size_t k = 0; k < result_ndim; ++k) {
            size_t dim1 = (k < ndim1) ? shape1[ndim1 - 1 - k] : 1;
            size_t dim2 = (k < ndim2) ? shape2[ndim2 - 1 - k] : 1;

            result[result_ndim - 1 - k] = (dim1 > dim2) ? dim1 : dim2;
        }

        return result;
    }
};

// ============================================================================
// Broadcasting operators
// ============================================================================

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

// ============================================================================
// Reshape operations
// ============================================================================

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

// ============================================================================
// Optimized broadcasting operations namespace (for future lazy evaluation)
// ============================================================================

namespace broadcast_ops {

// Addition with broadcasting (returns a lazy evaluation object in the future)
template<typename T, size_t D1, size_t D2>
auto add(const Tensor<T, D2>& vec, const Tensor<T, D1, D2>& mat) {
    // For now, we materialize, but this should return a lazy expression
    auto broadcast_view = broadcast_vector_to_matrix<T, D1, D2>(vec);
    Tensor<T, D1, D2> result;

    for (size_t i = 0; i < D1; ++i) {
        for (size_t j = 0; j < D2; ++j) {
            result(i, j) = broadcast_view(i, j) + mat(i, j);
        }
    }

    return result;
}

} // namespace broadcast_ops

} // namespace eigenwave

#endif // EIGENWAVE_BROADCASTING_HPP