#ifndef EIGENWAVE_OPERATIONS_HPP
#define EIGENWAVE_OPERATIONS_HPP

#include "tensor.hpp"
#include <cmath>
#include <functional>

namespace eigenwave {

// Helper to check if dimensions match
template<typename T1, typename T2>
struct dimensions_match : std::false_type {};

template<typename T, size_t... Dims>
struct dimensions_match<Tensor<T, Dims...>, Tensor<T, Dims...>> : std::true_type {};

template<typename T1, typename T2>
inline constexpr bool dimensions_match_v = dimensions_match<T1, T2>::value;

// Element-wise binary operations
template<typename T, size_t... Dims, typename BinaryOp>
Tensor<T, Dims...> element_wise_binary_op(const Tensor<T, Dims...>& a,
                                          const Tensor<T, Dims...>& b,
                                          BinaryOp op) {
    Tensor<T, Dims...> result;
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), op);
    return result;
}

// Element-wise unary operations
template<typename T, size_t... Dims, typename UnaryOp>
Tensor<T, Dims...> element_wise_unary_op(const Tensor<T, Dims...>& a, UnaryOp op) {
    Tensor<T, Dims...> result;
    std::transform(a.begin(), a.end(), result.begin(), op);
    return result;
}

// Scalar operations
template<typename T, size_t... Dims, typename ScalarOp>
Tensor<T, Dims...> scalar_op(const Tensor<T, Dims...>& a, T scalar, ScalarOp op) {
    Tensor<T, Dims...> result;
    std::transform(a.begin(), a.end(), result.begin(),
                   [scalar, op](const T& val) { return op(val, scalar); });
    return result;
}

// Addition
template<typename T, size_t... Dims>
Tensor<T, Dims...> operator+(const Tensor<T, Dims...>& a, const Tensor<T, Dims...>& b) {
    return element_wise_binary_op(a, b, std::plus<T>());
}

template<typename T, size_t... Dims>
Tensor<T, Dims...> operator+(const Tensor<T, Dims...>& a, T scalar) {
    return scalar_op(a, scalar, std::plus<T>());
}

template<typename T, size_t... Dims>
Tensor<T, Dims...> operator+(T scalar, const Tensor<T, Dims...>& a) {
    return a + scalar;
}

// Subtraction
template<typename T, size_t... Dims>
Tensor<T, Dims...> operator-(const Tensor<T, Dims...>& a, const Tensor<T, Dims...>& b) {
    return element_wise_binary_op(a, b, std::minus<T>());
}

template<typename T, size_t... Dims>
Tensor<T, Dims...> operator-(const Tensor<T, Dims...>& a, T scalar) {
    return scalar_op(a, scalar, std::minus<T>());
}

template<typename T, size_t... Dims>
Tensor<T, Dims...> operator-(T scalar, const Tensor<T, Dims...>& a) {
    Tensor<T, Dims...> result;
    std::transform(a.begin(), a.end(), result.begin(),
                   [scalar](const T& val) { return scalar - val; });
    return result;
}

// Multiplication (element-wise)
template<typename T, size_t... Dims>
Tensor<T, Dims...> operator*(const Tensor<T, Dims...>& a, const Tensor<T, Dims...>& b) {
    return element_wise_binary_op(a, b, std::multiplies<T>());
}

template<typename T, size_t... Dims>
Tensor<T, Dims...> operator*(const Tensor<T, Dims...>& a, T scalar) {
    return scalar_op(a, scalar, std::multiplies<T>());
}

template<typename T, size_t... Dims>
Tensor<T, Dims...> operator*(T scalar, const Tensor<T, Dims...>& a) {
    return a * scalar;
}

// Division (element-wise)
template<typename T, size_t... Dims>
Tensor<T, Dims...> operator/(const Tensor<T, Dims...>& a, const Tensor<T, Dims...>& b) {
    return element_wise_binary_op(a, b, std::divides<T>());
}

template<typename T, size_t... Dims>
Tensor<T, Dims...> operator/(const Tensor<T, Dims...>& a, T scalar) {
    return scalar_op(a, scalar, std::divides<T>());
}

template<typename T, size_t... Dims>
Tensor<T, Dims...> operator/(T scalar, const Tensor<T, Dims...>& a) {
    Tensor<T, Dims...> result;
    std::transform(a.begin(), a.end(), result.begin(),
                   [scalar](const T& val) { return scalar / val; });
    return result;
}

// Unary negation
template<typename T, size_t... Dims>
Tensor<T, Dims...> operator-(const Tensor<T, Dims...>& a) {
    return element_wise_unary_op(a, std::negate<T>());
}

// Compound assignment operators
template<typename T, size_t... Dims>
Tensor<T, Dims...>& operator+=(Tensor<T, Dims...>& a, const Tensor<T, Dims...>& b) {
    std::transform(a.begin(), a.end(), b.begin(), a.begin(), std::plus<T>());
    return a;
}

template<typename T, size_t... Dims>
Tensor<T, Dims...>& operator+=(Tensor<T, Dims...>& a, T scalar) {
    std::transform(a.begin(), a.end(), a.begin(),
                   [scalar](const T& val) { return val + scalar; });
    return a;
}

template<typename T, size_t... Dims>
Tensor<T, Dims...>& operator-=(Tensor<T, Dims...>& a, const Tensor<T, Dims...>& b) {
    std::transform(a.begin(), a.end(), b.begin(), a.begin(), std::minus<T>());
    return a;
}

template<typename T, size_t... Dims>
Tensor<T, Dims...>& operator-=(Tensor<T, Dims...>& a, T scalar) {
    std::transform(a.begin(), a.end(), a.begin(),
                   [scalar](const T& val) { return val - scalar; });
    return a;
}

template<typename T, size_t... Dims>
Tensor<T, Dims...>& operator*=(Tensor<T, Dims...>& a, const Tensor<T, Dims...>& b) {
    std::transform(a.begin(), a.end(), b.begin(), a.begin(), std::multiplies<T>());
    return a;
}

template<typename T, size_t... Dims>
Tensor<T, Dims...>& operator*=(Tensor<T, Dims...>& a, T scalar) {
    std::transform(a.begin(), a.end(), a.begin(),
                   [scalar](const T& val) { return val * scalar; });
    return a;
}

template<typename T, size_t... Dims>
Tensor<T, Dims...>& operator/=(Tensor<T, Dims...>& a, const Tensor<T, Dims...>& b) {
    std::transform(a.begin(), a.end(), b.begin(), a.begin(), std::divides<T>());
    return a;
}

template<typename T, size_t... Dims>
Tensor<T, Dims...>& operator/=(Tensor<T, Dims...>& a, T scalar) {
    std::transform(a.begin(), a.end(), a.begin(),
                   [scalar](const T& val) { return val / scalar; });
    return a;
}

// Mathematical functions
template<typename T, size_t... Dims>
Tensor<T, Dims...> abs(const Tensor<T, Dims...>& a) {
    return element_wise_unary_op(a, [](const T& val) { return std::abs(val); });
}

template<typename T, size_t... Dims>
Tensor<T, Dims...> sqrt(const Tensor<T, Dims...>& a) {
    return element_wise_unary_op(a, [](const T& val) { return std::sqrt(val); });
}

template<typename T, size_t... Dims>
Tensor<T, Dims...> exp(const Tensor<T, Dims...>& a) {
    return element_wise_unary_op(a, [](const T& val) { return std::exp(val); });
}

template<typename T, size_t... Dims>
Tensor<T, Dims...> log(const Tensor<T, Dims...>& a) {
    return element_wise_unary_op(a, [](const T& val) { return std::log(val); });
}

template<typename T, size_t... Dims>
Tensor<T, Dims...> sin(const Tensor<T, Dims...>& a) {
    return element_wise_unary_op(a, [](const T& val) { return std::sin(val); });
}

template<typename T, size_t... Dims>
Tensor<T, Dims...> cos(const Tensor<T, Dims...>& a) {
    return element_wise_unary_op(a, [](const T& val) { return std::cos(val); });
}

template<typename T, size_t... Dims>
Tensor<T, Dims...> tan(const Tensor<T, Dims...>& a) {
    return element_wise_unary_op(a, [](const T& val) { return std::tan(val); });
}

template<typename T, size_t... Dims>
Tensor<T, Dims...> pow(const Tensor<T, Dims...>& a, T exponent) {
    return element_wise_unary_op(a, [exponent](const T& val) { return std::pow(val, exponent); });
}

template<typename T, size_t... Dims>
Tensor<T, Dims...> pow(const Tensor<T, Dims...>& a, const Tensor<T, Dims...>& b) {
    return element_wise_binary_op(a, b, [](const T& x, const T& y) { return std::pow(x, y); });
}

// Comparison operations
template<typename T, size_t... Dims>
Tensor<bool, Dims...> operator==(const Tensor<T, Dims...>& a, const Tensor<T, Dims...>& b) {
    Tensor<bool, Dims...> result;
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::equal_to<T>());
    return result;
}

template<typename T, size_t... Dims>
Tensor<bool, Dims...> operator!=(const Tensor<T, Dims...>& a, const Tensor<T, Dims...>& b) {
    Tensor<bool, Dims...> result;
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::not_equal_to<T>());
    return result;
}

template<typename T, size_t... Dims>
Tensor<bool, Dims...> operator<(const Tensor<T, Dims...>& a, const Tensor<T, Dims...>& b) {
    Tensor<bool, Dims...> result;
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::less<T>());
    return result;
}

template<typename T, size_t... Dims>
Tensor<bool, Dims...> operator>(const Tensor<T, Dims...>& a, const Tensor<T, Dims...>& b) {
    Tensor<bool, Dims...> result;
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::greater<T>());
    return result;
}

template<typename T, size_t... Dims>
Tensor<bool, Dims...> operator<=(const Tensor<T, Dims...>& a, const Tensor<T, Dims...>& b) {
    Tensor<bool, Dims...> result;
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::less_equal<T>());
    return result;
}

template<typename T, size_t... Dims>
Tensor<bool, Dims...> operator>=(const Tensor<T, Dims...>& a, const Tensor<T, Dims...>& b) {
    Tensor<bool, Dims...> result;
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::greater_equal<T>());
    return result;
}

// Reduction operations
template<typename T, size_t... Dims>
T sum(const Tensor<T, Dims...>& a) {
    return std::accumulate(a.begin(), a.end(), T(0));
}

template<typename T, size_t... Dims>
T product(const Tensor<T, Dims...>& a) {
    return std::accumulate(a.begin(), a.end(), T(1), std::multiplies<T>());
}

template<typename T, size_t... Dims>
T min(const Tensor<T, Dims...>& a) {
    return *std::min_element(a.begin(), a.end());
}

template<typename T, size_t... Dims>
T max(const Tensor<T, Dims...>& a) {
    return *std::max_element(a.begin(), a.end());
}

template<typename T, size_t... Dims>
T mean(const Tensor<T, Dims...>& a) {
    return sum(a) / static_cast<T>(a.size);
}

} // namespace eigenwave

#endif // EIGENWAVE_OPERATIONS_HPP