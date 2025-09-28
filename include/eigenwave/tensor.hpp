#ifndef EIGENWAVE_TENSOR_HPP
#define EIGENWAVE_TENSOR_HPP

#include <array>
#include <algorithm>
#include <numeric>
#include <initializer_list>
#include <type_traits>
#include <utility>
#include <cstddef>
#include <iostream>
#include <iomanip>

namespace eigenwave {

// Helper template to calculate product of dimensions
template<size_t... Dims>
struct product_of_dims;

template<>
struct product_of_dims<> {
    static constexpr size_t value = 1;
};

template<size_t First, size_t... Rest>
struct product_of_dims<First, Rest...> {
    static constexpr size_t value = First * product_of_dims<Rest...>::value;
};

template<size_t... Dims>
inline constexpr size_t product_of_dims_v = product_of_dims<Dims...>::value;

// Helper to get dimension at index
template<size_t Index, size_t... Dims>
struct dim_at_index;

template<size_t First, size_t... Rest>
struct dim_at_index<0, First, Rest...> {
    static constexpr size_t value = First;
};

template<size_t Index, size_t First, size_t... Rest>
struct dim_at_index<Index, First, Rest...> {
    static_assert(Index <= sizeof...(Rest), "Index out of bounds");
    static constexpr size_t value = dim_at_index<Index - 1, Rest...>::value;
};

// Forward declaration
template<typename T, size_t... Dims>
class Tensor;

// Type trait to check if type is a Tensor
template<typename T>
struct is_tensor : std::false_type {};

template<typename T, size_t... Dims>
struct is_tensor<Tensor<T, Dims...>> : std::true_type {};

template<typename T>
inline constexpr bool is_tensor_v = is_tensor<T>::value;

// Main Tensor class template
template<typename T, size_t... Dims>
class Tensor {
public:
    using value_type = T;
    static constexpr size_t ndim = sizeof...(Dims);
    static constexpr size_t size = product_of_dims_v<Dims...>;
    static constexpr std::array<size_t, ndim> shape = {Dims...};

private:
    std::array<T, size> data_;

    // Helper for multi-dimensional indexing
    template<size_t... Is>
    static constexpr size_t compute_index(std::index_sequence<Is...>, size_t indices[ndim]) {
        size_t strides[ndim];
        strides[ndim - 1] = 1;
        if constexpr (ndim > 1) {
            for (int i = static_cast<int>(ndim) - 2; i >= 0; --i) {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        return ((indices[Is] * strides[Is]) + ...);
    }

public:
    // Default constructor - zero initialization
    Tensor() : data_{} {}

    // Fill constructor
    explicit Tensor(T value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    // Initializer list constructor
    Tensor(std::initializer_list<T> init) {
        if (init.size() != size) {
            throw std::runtime_error("Initializer list size does not match tensor size");
        }
        std::copy(init.begin(), init.end(), data_.begin());
    }

    // Copy constructor
    Tensor(const Tensor& other) = default;

    // Move constructor
    Tensor(Tensor&& other) noexcept = default;

    // Copy assignment
    Tensor& operator=(const Tensor& other) = default;

    // Move assignment
    Tensor& operator=(Tensor&& other) noexcept = default;

    // Element access with bounds checking
    template<typename... Indices>
    T& at(Indices... indices) {
        static_assert(sizeof...(indices) == ndim, "Number of indices must match tensor dimensions");
        size_t idx_array[ndim] = {static_cast<size_t>(indices)...};

        // Bounds checking
        for (size_t i = 0; i < ndim; ++i) {
            if (idx_array[i] >= shape[i]) {
                throw std::out_of_range("Index out of bounds");
            }
        }

        size_t flat_idx = compute_index(std::make_index_sequence<ndim>{}, idx_array);
        return data_[flat_idx];
    }

    template<typename... Indices>
    const T& at(Indices... indices) const {
        return const_cast<Tensor*>(this)->at(indices...);
    }

    // Element access without bounds checking (operator())
    template<typename... Indices>
    T& operator()(Indices... indices) {
        static_assert(sizeof...(indices) == ndim, "Number of indices must match tensor dimensions");
        size_t idx_array[ndim] = {static_cast<size_t>(indices)...};
        size_t flat_idx = compute_index(std::make_index_sequence<ndim>{}, idx_array);
        return data_[flat_idx];
    }

    template<typename... Indices>
    const T& operator()(Indices... indices) const {
        return const_cast<Tensor*>(this)->operator()(indices...);
    }

    // Raw data access
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    // Iterator support
    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto begin() const { return data_.begin(); }
    auto end() const { return data_.end(); }

    // Fill tensor with a value
    void fill(T value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    // Static factory methods
    static Tensor zeros() {
        return Tensor(T(0));
    }

    static Tensor ones() {
        return Tensor(T(1));
    }

    static Tensor full(T value) {
        return Tensor(value);
    }

    // Print tensor (for debugging)
    void print(std::ostream& os = std::cout) const {
        os << "Tensor<";
        for (size_t i = 0; i < ndim; ++i) {
            os << shape[i];
            if (i < ndim - 1) os << ", ";
        }
        os << ">" << std::endl;

        // For 1D tensors
        if constexpr (ndim == 1) {
            os << "[";
            for (size_t i = 0; i < size; ++i) {
                os << std::setw(8) << data_[i];
                if (i < size - 1) os << ", ";
            }
            os << "]" << std::endl;
        }
        // For 2D tensors
        else if constexpr (ndim == 2) {
            os << "[" << std::endl;
            for (size_t i = 0; i < shape[0]; ++i) {
                os << "  [";
                for (size_t j = 0; j < shape[1]; ++j) {
                    os << std::setw(8) << (*this)(i, j);
                    if (j < shape[1] - 1) os << ", ";
                }
                os << "]";
                if (i < shape[0] - 1) os << ",";
                os << std::endl;
            }
            os << "]" << std::endl;
        }
        // For higher dimensions, just print flat array
        else {
            os << "Data (flat): [";
            for (size_t i = 0; i < std::min(size, size_t(10)); ++i) {
                os << data_[i];
                if (i < size - 1) os << ", ";
            }
            if (size > 10) os << ", ...";
            os << "]" << std::endl;
        }
    }
};

// Type aliases for common tensor types
template<typename T>
using Vector = Tensor<T, 0>;  // Will need to be specialized

template<typename T, size_t N>
using Vector1D = Tensor<T, N>;

template<typename T, size_t M, size_t N>
using Matrix = Tensor<T, M, N>;

template<typename T, size_t D1, size_t D2, size_t D3>
using Tensor3D = Tensor<T, D1, D2, D3>;

} // namespace eigenwave

#endif // EIGENWAVE_TENSOR_HPP