#ifndef EIGENWAVE_CONCATENATE_HPP
#define EIGENWAVE_CONCATENATE_HPP

#include "tensor.hpp"
#include <array>
#include <algorithm>
#include <numeric>
#include <type_traits>

namespace eigenwave {

// ============================================================================
// Helper templates for compile-time dimension validation
// ============================================================================

// Check if all tensors have the same shape except along a specific axis
template<size_t Axis, typename... Tensors>
struct can_concatenate;

template<size_t Axis, typename T, size_t... Dims1, size_t... Dims2>
struct can_concatenate<Axis, Tensor<T, Dims1...>, Tensor<T, Dims2...>> {
    static constexpr size_t ndim1 = sizeof...(Dims1);
    static constexpr size_t ndim2 = sizeof...(Dims2);
    static constexpr std::array<size_t, ndim1> shape1 = {Dims1...};
    static constexpr std::array<size_t, ndim2> shape2 = {Dims2...};

    static constexpr bool value() {
        if constexpr (ndim1 != ndim2) {
            return false;
        }
        if constexpr (Axis >= ndim1) {
            return false;
        }
        // Check all dimensions except the concatenation axis
        for (size_t i = 0; i < ndim1; ++i) {
            if (i != Axis && shape1[i] != shape2[i]) {
                return false;
            }
        }
        return true;
    }
};

// Variadic version for multiple tensors
template<size_t Axis, typename T1, typename T2, typename T3, typename... Rest>
struct can_concatenate<Axis, T1, T2, T3, Rest...> {
    static constexpr bool value() {
        return can_concatenate<Axis, T1, T2>::value() &&
               can_concatenate<Axis, T1, T3, Rest...>::value();
    }
};

// Helper to compute concatenated shape
template<size_t Axis, typename... Tensors>
struct concatenated_shape;

template<size_t Axis, typename T, size_t... Dims1, size_t... Dims2>
struct concatenated_shape<Axis, Tensor<T, Dims1...>, Tensor<T, Dims2...>> {
    static constexpr std::array<size_t, sizeof...(Dims1)> shape1 = {Dims1...};
    static constexpr std::array<size_t, sizeof...(Dims2)> shape2 = {Dims2...};

    static constexpr auto compute() {
        std::array<size_t, sizeof...(Dims1)> result = shape1;
        result[Axis] += shape2[Axis];
        return result;
    }
};

// Helper to extract dimension at compile time
template<size_t Idx, size_t... Values>
struct get_at_index;

template<size_t... Values>
struct get_at_index<0, Values...> {
    static constexpr size_t value = 0;  // Will be specialized
};

// ============================================================================
// Concatenate implementation for 1D tensors
// ============================================================================

template<typename T, size_t D1, size_t D2>
Tensor<T, D1 + D2> concatenate(const Tensor<T, D1>& t1, const Tensor<T, D2>& t2) {
    Tensor<T, D1 + D2> result;

    // Copy first tensor
    std::copy(t1.begin(), t1.end(), result.begin());
    // Copy second tensor
    std::copy(t2.begin(), t2.end(), result.begin() + D1);

    return result;
}

// Variadic version for multiple 1D tensors
template<typename T, size_t D1, size_t D2, size_t D3, size_t... Rest>
auto concatenate(const Tensor<T, D1>& t1, const Tensor<T, D2>& t2,
                 const Tensor<T, D3>& t3, const Tensor<T, Rest>&... rest) {
    auto first_concat = concatenate(t1, t2);
    return concatenate(first_concat, t3, rest...);
}

// ============================================================================
// Concatenate implementation for 2D tensors
// ============================================================================

// Concatenate along axis 0 (rows)
template<typename T, size_t R1, size_t R2, size_t C>
Tensor<T, R1 + R2, C> concatenate_axis0(const Tensor<T, R1, C>& t1,
                                        const Tensor<T, R2, C>& t2) {
    Tensor<T, R1 + R2, C> result;

    // Copy first tensor
    for (size_t i = 0; i < R1; ++i) {
        for (size_t j = 0; j < C; ++j) {
            result(i, j) = t1(i, j);
        }
    }

    // Copy second tensor
    for (size_t i = 0; i < R2; ++i) {
        for (size_t j = 0; j < C; ++j) {
            result(R1 + i, j) = t2(i, j);
        }
    }

    return result;
}

// Concatenate along axis 1 (columns)
template<typename T, size_t R, size_t C1, size_t C2>
Tensor<T, R, C1 + C2> concatenate_axis1(const Tensor<T, R, C1>& t1,
                                        const Tensor<T, R, C2>& t2) {
    Tensor<T, R, C1 + C2> result;

    for (size_t i = 0; i < R; ++i) {
        // Copy from first tensor
        for (size_t j = 0; j < C1; ++j) {
            result(i, j) = t1(i, j);
        }
        // Copy from second tensor
        for (size_t j = 0; j < C2; ++j) {
            result(i, C1 + j) = t2(i, j);
        }
    }

    return result;
}

// General 2D concatenate with axis template parameter
template<size_t Axis, typename T, size_t... Dims1, size_t... Dims2>
auto concatenate(const Tensor<T, Dims1...>& t1, const Tensor<T, Dims2...>& t2) {
    static_assert(sizeof...(Dims1) == 2 && sizeof...(Dims2) == 2,
                  "This overload is for 2D tensors");

    if constexpr (Axis == 0) {
        return concatenate_axis0(t1, t2);
    } else if constexpr (Axis == 1) {
        return concatenate_axis1(t1, t2);
    } else {
        static_assert(Axis < 2, "Axis must be 0 or 1 for 2D tensors");
    }
}

// ============================================================================
// Stack implementation - Creates a new axis
// ============================================================================

// Stack 1D tensors to create a 2D tensor
template<typename T, size_t D>
Tensor<T, 2, D> stack(const Tensor<T, D>& t1, const Tensor<T, D>& t2) {
    Tensor<T, 2, D> result;

    for (size_t j = 0; j < D; ++j) {
        result(0, j) = t1(j);
        result(1, j) = t2(j);
    }

    return result;
}

// Stack multiple 1D tensors
template<typename T, size_t D, typename... Tensors>
auto stack(const Tensor<T, D>& t1, const Tensor<T, D>& t2, const Tensors&... rest) {
    static_assert((std::is_same_v<Tensors, Tensor<T, D>> && ...),
                  "All tensors must have the same shape");

    constexpr size_t num_tensors = 2 + sizeof...(rest);
    Tensor<T, num_tensors, D> result;

    // Copy first tensor
    for (size_t j = 0; j < D; ++j) {
        result(0, j) = t1(j);
    }

    // Copy second tensor
    for (size_t j = 0; j < D; ++j) {
        result(1, j) = t2(j);
    }

    // Helper to copy remaining tensors
    size_t row_idx = 2;
    ((void)([&](const auto& tensor) {
        for (size_t j = 0; j < D; ++j) {
            result(row_idx, j) = tensor(j);
        }
        ++row_idx;
    }(rest)), ...);

    return result;
}

// Stack 2D tensors to create a 3D tensor
template<typename T, size_t R, size_t C>
Tensor<T, 2, R, C> stack(const Tensor<T, R, C>& t1, const Tensor<T, R, C>& t2) {
    Tensor<T, 2, R, C> result;

    for (size_t i = 0; i < R; ++i) {
        for (size_t j = 0; j < C; ++j) {
            result(0, i, j) = t1(i, j);
            result(1, i, j) = t2(i, j);
        }
    }

    return result;
}

// ============================================================================
// Convenience functions matching NumPy API
// ============================================================================

// Horizontal stack (column-wise concatenation for 2D)
template<typename T, size_t... Dims1, size_t... Dims2>
auto hstack(const Tensor<T, Dims1...>& t1, const Tensor<T, Dims2...>& t2) {
    constexpr size_t ndim = sizeof...(Dims1);

    if constexpr (ndim == 1) {
        // For 1D, hstack is same as concatenate
        return concatenate(t1, t2);
    } else if constexpr (ndim == 2) {
        // For 2D, concatenate along axis 1 (columns)
        return concatenate<1>(t1, t2);
    } else {
        static_assert(ndim <= 2, "hstack currently supports only 1D and 2D tensors");
    }
}

// Vertical stack (row-wise concatenation for 2D)
template<typename T, size_t... Dims1, size_t... Dims2>
auto vstack(const Tensor<T, Dims1...>& t1, const Tensor<T, Dims2...>& t2) {
    constexpr size_t ndim1 = sizeof...(Dims1);
    constexpr size_t ndim2 = sizeof...(Dims2);

    if constexpr (ndim1 == 1 && ndim2 == 1) {
        // For 1D tensors, vstack creates a new axis
        static_assert(std::is_same_v<Tensor<T, Dims1...>, Tensor<T, Dims2...>>,
                      "1D tensors must have the same shape for vstack");
        return stack(t1, t2);
    } else if constexpr (ndim1 == 2 && ndim2 == 2) {
        // For 2D, concatenate along axis 0 (rows)
        return concatenate<0>(t1, t2);
    } else {
        static_assert(ndim1 <= 2, "vstack currently supports only 1D and 2D tensors");
    }
}

// Depth stack - stack along third axis
template<typename T, size_t R, size_t C>
Tensor<T, R, C, 2> dstack(const Tensor<T, R, C>& t1, const Tensor<T, R, C>& t2) {
    Tensor<T, R, C, 2> result;

    for (size_t i = 0; i < R; ++i) {
        for (size_t j = 0; j < C; ++j) {
            result(i, j, 0) = t1(i, j);
            result(i, j, 1) = t2(i, j);
        }
    }

    return result;
}

// Overload for 1D tensors (reshape to Nx1x2)
template<typename T, size_t D>
Tensor<T, D, 1, 2> dstack(const Tensor<T, D>& t1, const Tensor<T, D>& t2) {
    Tensor<T, D, 1, 2> result;

    for (size_t i = 0; i < D; ++i) {
        result(i, 0, 0) = t1(i);
        result(i, 0, 1) = t2(i);
    }

    return result;
}

} // namespace eigenwave

#endif // EIGENWAVE_CONCATENATE_HPP