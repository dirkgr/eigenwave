#ifndef EIGENWAVE_BROADCAST_VIEW_HPP
#define EIGENWAVE_BROADCAST_VIEW_HPP

#include "tensor.hpp"
#include <array>
#include <algorithm>

namespace eigenwave {

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
        // Original strides (for the source tensor)
        std::array<size_t, orig_ndim> orig_strides{};
        if constexpr (orig_ndim > 0) {
            orig_strides[orig_ndim - 1] = 1;
            if constexpr (orig_ndim > 1) {
                for (int i = static_cast<int>(orig_ndim) - 2; i >= 0; --i) {
                    orig_strides[i] = orig_strides[i + 1] * orig_shape[i + 1];
                }
            }
        }

        // Compute broadcast strides
        // Start from the right and align dimensions
        size_t orig_idx = orig_ndim - 1;
        for (int i = broadcast_ndim - 1; i >= 0; --i) {
            if (orig_idx < orig_ndim) {
                if (orig_shape[orig_idx] == broadcast_shape[i]) {
                    // Dimension matches - use original stride
                    effective_strides_[i] = orig_strides[orig_idx];
                } else if (orig_shape[orig_idx] == 1) {
                    // Broadcasting a dimension of size 1 - stride is 0
                    effective_strides_[i] = 0;
                } else if (broadcast_shape[i] == 1) {
                    // Target dimension is 1 (shouldn't happen in broadcasting)
                    effective_strides_[i] = orig_strides[orig_idx];
                }
                if (orig_idx > 0) orig_idx--;
            } else {
                // No corresponding dimension in original tensor
                // This means we're broadcasting a smaller tensor to more dimensions
                effective_strides_[i] = 0;
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

        // Copy all elements using the broadcast view
        size_t flat_idx = 0;
        materialize_recursive<0>(result.data(), flat_idx);

        return result;
    }

private:
    // Helper to materialize the view recursively
    template<size_t Dim>
    void materialize_recursive(T* output, size_t& out_idx) const {
        if constexpr (Dim == broadcast_ndim) {
            // Base case - we've indexed all dimensions
            std::array<size_t, broadcast_ndim> indices{};
            size_t temp_idx = out_idx;

            // Convert flat index back to multi-dimensional indices
            for (int i = broadcast_ndim - 1; i >= 0; --i) {
                indices[i] = temp_idx % broadcast_shape[i];
                temp_idx /= broadcast_shape[i];
            }

            // Calculate source index using broadcast strides
            size_t src_idx = 0;
            for (size_t i = 0; i < broadcast_ndim; ++i) {
                src_idx += indices[i] * effective_strides_[i];
            }

            output[out_idx++] = data_ptr_[src_idx];
        } else {
            // Recursive case
            for (size_t i = 0; i < broadcast_shape[Dim]; ++i) {
                materialize_recursive<Dim + 1>(output, out_idx);
            }
        }
    }
};

// Helper function to create broadcast view for vector -> matrix broadcasting
template<typename T, size_t D1, size_t D2>
auto broadcast_vector_to_matrix(const Tensor<T, D2>& vec) {
    // Broadcasting [D2] -> [D1, D2]
    // Each row will be identical to the vector
    return BroadcastView<T, std::index_sequence<D2>, std::index_sequence<D1, D2>>(vec);
}

// Helper for scalar broadcasting (though scalar is a special case)
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

// Create a broadcast view that virtually expands dimensions
template<typename T, size_t... Dims>
auto broadcast_expand_dims_front(const Tensor<T, Dims...>& tensor) {
    return BroadcastView<T, std::index_sequence<Dims...>, std::index_sequence<1, Dims...>>(tensor);
}

// Binary operations with broadcast views
template<typename T, size_t... ADims, size_t... BDims, size_t... ResultDims, typename BinaryOp>
Tensor<T, ResultDims...> broadcast_binary_op(
    const Tensor<T, ADims...>& a,
    const Tensor<T, BDims...>& b,
    BinaryOp op) {

    // Create broadcast views if needed
    // This is a simplified version - full implementation would determine
    // ResultDims based on NumPy broadcasting rules

    Tensor<T, ResultDims...> result;

    // Apply operation element by element
    // (Implementation would need proper index iteration)

    return result;
}

// Optimized broadcasting operations that return views when possible
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

// Check if broadcasting is possible between two shapes
template<size_t... Dims1, size_t... Dims2>
constexpr bool can_broadcast() {
    constexpr size_t shape1[] = {Dims1...};
    constexpr size_t shape2[] = {Dims2...};
    constexpr size_t ndim1 = sizeof...(Dims1);
    constexpr size_t ndim2 = sizeof...(Dims2);

    // Broadcasting rules: dimensions are compatible if:
    // 1. They are equal, or
    // 2. One of them is 1
    size_t i = ndim1 - 1;
    size_t j = ndim2 - 1;

    while (i < ndim1 && j < ndim2) {
        if (shape1[i] != shape2[j] && shape1[i] != 1 && shape2[j] != 1) {
            return false;
        }
        if (i == 0 || j == 0) break;
        --i;
        --j;
    }

    return true;
}

// Compute broadcast result shape
template<size_t... Dims1, size_t... Dims2>
constexpr auto broadcast_shape() {
    constexpr size_t shape1[] = {Dims1...};
    constexpr size_t shape2[] = {Dims2...};
    constexpr size_t ndim1 = sizeof...(Dims1);
    constexpr size_t ndim2 = sizeof...(Dims2);
    constexpr size_t result_ndim = std::max(ndim1, ndim2);

    std::array<size_t, result_ndim> result{};

    // Fill from the right
    for (size_t k = 0; k < result_ndim; ++k) {
        size_t dim1 = (k < ndim1) ? shape1[ndim1 - 1 - k] : 1;
        size_t dim2 = (k < ndim2) ? shape2[ndim2 - 1 - k] : 1;

        result[result_ndim - 1 - k] = std::max(dim1, dim2);
    }

    return result;
}

} // namespace eigenwave

#endif // EIGENWAVE_BROADCAST_VIEW_HPP