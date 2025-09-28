#ifndef EIGENWAVE_SLICING_HPP
#define EIGENWAVE_SLICING_HPP

#include "tensor.hpp"
#include <utility>
#include <tuple>

namespace eigenwave {

// Range struct for specifying slice ranges
struct Range {
    size_t start;
    size_t stop;
    size_t step;

    constexpr Range(size_t stop_) : start(0), stop(stop_), step(1) {}
    constexpr Range(size_t start_, size_t stop_) : start(start_), stop(stop_), step(1) {}
    constexpr Range(size_t start_, size_t stop_, size_t step_)
        : start(start_), stop(stop_), step(step_) {}

    constexpr size_t size() const {
        return (stop > start) ? (stop - start + step - 1) / step : 0;
    }
};

// Helper to create a range
inline constexpr Range range(size_t stop) { return Range(stop); }
inline constexpr Range range(size_t start, size_t stop) { return Range(start, stop); }
inline constexpr Range range(size_t start, size_t stop, size_t step) {
    return Range(start, stop, step);
}

// Slice result dimensions calculator
template<typename... Ranges>
struct slice_dims;

template<>
struct slice_dims<> {
    static constexpr size_t count = 0;
};

template<typename First, typename... Rest>
struct slice_dims<First, Rest...> {
    static constexpr size_t count = slice_dims<Rest...>::count +
        (std::is_same_v<First, Range> ? 1 : 0);
};

// TensorView class for non-owning slices
template<typename T, size_t... ViewDims>
class TensorView {
public:
    using value_type = T;
    static constexpr size_t ndim = sizeof...(ViewDims);
    static constexpr size_t size = product_of_dims_v<ViewDims...>;
    static constexpr std::array<size_t, ndim> shape = {ViewDims...};

private:
    T* data_ptr_;
    std::array<size_t, ndim> strides_;

public:
    TensorView(T* ptr, const std::array<size_t, ndim>& strides)
        : data_ptr_(ptr), strides_(strides) {}

    // Element access
    template<typename... Indices>
    T& operator()(Indices... indices) {
        static_assert(sizeof...(indices) == ndim,
                      "Number of indices must match tensor dimensions");
        size_t idx_array[ndim] = {static_cast<size_t>(indices)...};
        size_t offset = 0;
        for (size_t i = 0; i < ndim; ++i) {
            offset += idx_array[i] * strides_[i];
        }
        return data_ptr_[offset];
    }

    template<typename... Indices>
    const T& operator()(Indices... indices) const {
        return const_cast<TensorView*>(this)->operator()(indices...);
    }

    // Convert view to owned tensor
    Tensor<T, ViewDims...> copy() const {
        Tensor<T, ViewDims...> result;

        // Copy data from view to new tensor
        if constexpr (ndim == 1) {
            for (size_t i = 0; i < shape[0]; ++i) {
                result(i) = (*this)(i);
            }
        } else if constexpr (ndim == 2) {
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    result(i, j) = (*this)(i, j);
                }
            }
        } else if constexpr (ndim == 3) {
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    for (size_t k = 0; k < shape[2]; ++k) {
                        result(i, j, k) = (*this)(i, j, k);
                    }
                }
            }
        }

        return result;
    }

    // Assignment from tensor
    template<size_t... OtherDims>
    TensorView& operator=(const Tensor<T, OtherDims...>& other) {
        // Allow assignment from tensors with compatible dimensions
        if constexpr (ndim == 1) {
            for (size_t i = 0; i < shape[0]; ++i) {
                (*this)(i) = other(i);
            }
        } else if constexpr (ndim == 2) {
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    (*this)(i, j) = other(i, j);
                }
            }
        } else if constexpr (ndim == 3) {
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    for (size_t k = 0; k < shape[2]; ++k) {
                        (*this)(i, j, k) = other(i, j, k);
                    }
                }
            }
        }
        return *this;
    }

    // Fill with value
    void fill(T value) {
        if constexpr (ndim == 1) {
            for (size_t i = 0; i < shape[0]; ++i) {
                (*this)(i) = value;
            }
        } else if constexpr (ndim == 2) {
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    (*this)(i, j) = value;
                }
            }
        } else if constexpr (ndim == 3) {
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    for (size_t k = 0; k < shape[2]; ++k) {
                        (*this)(i, j, k) = value;
                    }
                }
            }
        }
    }
};

// Slicing implementation for 1D tensors
template<typename T, size_t D1>
auto slice(Tensor<T, D1>& tensor, Range r) {
    // Validate range
    if (r.stop > D1) r.stop = D1;

    size_t new_size = r.size();
    std::array<size_t, 1> strides = {r.step};

    // For simplicity, returning a dynamically sized view
    // In production, we'd compute size at compile time
    return TensorView<T, D1>(
        tensor.data() + r.start, strides
    );
}

// Helper for compile-time dimension calculation for 2D slicing
template<typename R1, typename R2>
struct slice_2d_dims {
    static constexpr size_t d1 = std::is_same_v<R1, Range> ? 0 : 1;
    static constexpr size_t d2 = std::is_same_v<R2, Range> ? 0 : 1;
    static constexpr size_t ndim = (d1 == 0 ? 1 : 0) + (d2 == 0 ? 1 : 0);
};

// Slicing implementation for 2D tensors - returns 1D view for single row/column
template<typename T, size_t D1, size_t D2>
auto slice(Tensor<T, D1, D2>& tensor, size_t row, Range col_range) {
    if (col_range.stop > D2) col_range.stop = D2;
    size_t new_size = col_range.size();

    // Create a view that references the specific row
    std::array<size_t, 1> strides = {col_range.step};
    T* start_ptr = tensor.data() + row * D2 + col_range.start;

    // Note: We need to compute the actual size at compile time
    // For now, returning a dynamically sized view
    return TensorView<T, D2>(start_ptr, strides);
}

template<typename T, size_t D1, size_t D2>
auto slice(Tensor<T, D1, D2>& tensor, Range row_range, size_t col) {
    if (row_range.stop > D1) row_range.stop = D1;
    size_t new_size = row_range.size();

    std::array<size_t, 1> strides = {D2 * row_range.step};
    T* start_ptr = tensor.data() + row_range.start * D2 + col;

    return TensorView<T, D1>(start_ptr, strides);
}

// Slicing for 2D tensors - returns 2D view
template<typename T, size_t D1, size_t D2>
auto slice(Tensor<T, D1, D2>& tensor, Range row_range, Range col_range) {
    if (row_range.stop > D1) row_range.stop = D1;
    if (col_range.stop > D2) col_range.stop = D2;

    size_t new_rows = row_range.size();
    size_t new_cols = col_range.size();

    std::array<size_t, 2> strides = {D2 * row_range.step, col_range.step};
    T* start_ptr = tensor.data() + row_range.start * D2 + col_range.start;

    // Note: Returning dynamically sized for simplicity
    // In a full implementation, we'd compute sizes at compile time
    return TensorView<T, D1, D2>(start_ptr, strides);
}

// Helper function to get a row from a 2D tensor
template<typename T, size_t D1, size_t D2>
auto row(Tensor<T, D1, D2>& tensor, size_t row_idx) {
    return slice(tensor, row_idx, range(0, D2));
}

// Helper function to get a column from a 2D tensor
template<typename T, size_t D1, size_t D2>
auto col(Tensor<T, D1, D2>& tensor, size_t col_idx) {
    return slice(tensor, range(0, D1), col_idx);
}

// Helper to get a submatrix
template<typename T, size_t D1, size_t D2>
auto submatrix(Tensor<T, D1, D2>& tensor,
               size_t row_start, size_t row_end,
               size_t col_start, size_t col_end) {
    return slice(tensor, range(row_start, row_end), range(col_start, col_end));
}

} // namespace eigenwave

#endif // EIGENWAVE_SLICING_HPP