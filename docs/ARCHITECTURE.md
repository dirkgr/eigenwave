# EigenWave Architecture Overview

## Design Philosophy

EigenWave is designed around three core principles:

1. **Compile-Time Safety**: Dimension errors are caught at compile time, not runtime
2. **Zero-Cost Abstractions**: Template metaprogramming ensures no runtime overhead
3. **Familiar API**: NumPy-like interface for ease of use

## Component Overview

```
eigenwave/
├── tensor.hpp         # Core tensor class with owned storage
├── operations.hpp     # Element-wise operations and reductions
├── slicing.hpp       # View-based slicing functionality
└── broadcasting.hpp  # Broadcasting operations
```

### Core Components

#### 1. Tensor Class (`tensor.hpp`)
The fundamental data structure that owns and manages tensor data.

**Key Features:**
- Fixed-size, stack-allocated storage
- Compile-time dimension checking
- STL-compatible iterators
- Factory methods (zeros, ones, full)

**When to Use:**
- When you need to store tensor data
- For small to medium-sized tensors (< 1MB)
- When you need value semantics

#### 2. Operations (`operations.hpp`)
Element-wise and reduction operations on tensors.

**Categories:**
- Arithmetic: `+`, `-`, `*`, `/`
- Mathematical: `exp`, `log`, `sin`, `cos`, `sqrt`, `pow`
- Reductions: `sum`, `product`, `min`, `max`, `mean`
- Comparisons: `==`, `!=`, `<`, `>`, `<=`, `>=`

**Design Pattern:**
All operations return new tensors (functional style), except compound assignments (`+=`, `-=`, etc.)

#### 3. TensorView Class (`slicing.hpp`)
Non-owning views into tensor data for zero-copy slicing.

**Key Features:**
- Reference semantics (non-owning)
- Custom strides for flexible slicing
- Mutable access to underlying data
- Conversion to owned tensor via `copy()`

**When to Use:**
- Extracting rows/columns/submatrices
- Passing tensor subsets to functions
- Modifying specific regions of a tensor

#### 4. Broadcasting (`broadcasting.hpp`)
NumPy-style broadcasting for operations between tensors of different shapes.

**Supported Patterns:**
- Scalar-tensor broadcasting
- Vector-matrix broadcasting
- Dimension manipulation (reshape, squeeze, expand_dims)

## Type System

### Template Parameters

```cpp
template<typename T, size_t... Dims>
class Tensor;
```

- `T`: Element type (float, double, int, etc.)
- `Dims...`: Variadic template for dimensions

### Type Aliases

```cpp
// Convenience aliases
template<typename T, size_t N>
using Vector1D = Tensor<T, N>;

template<typename T, size_t M, size_t N>
using Matrix = Tensor<T, M, N>;

template<typename T, size_t D1, size_t D2, size_t D3>
using Tensor3D = Tensor<T, D1, D2, D3>;
```

## Compile-Time Mechanisms

### Dimension Checking

```cpp
// These errors are caught at compile time:
Tensor<float, 2, 3> a;
Tensor<float, 3, 2> b;
auto c = a + b;  // ERROR: Dimension mismatch
```

### Static Assertions

```cpp
template<typename T, size_t... Dims>
class Tensor {
    static_assert(sizeof...(Dims) > 0, "Tensor must have at least one dimension");
    static_assert(product_of_dims_v<Dims...> > 0, "All dimensions must be positive");
    // ...
};
```

### SFINAE and Concepts (C++20)

```cpp
// Type trait to detect tensors
template<typename T>
struct is_tensor : std::false_type {};

template<typename T, size_t... Dims>
struct is_tensor<Tensor<T, Dims...>> : std::true_type {};

// Enable operations only for tensors
template<typename T, typename = std::enable_if_t<is_tensor_v<T>>>
auto some_operation(const T& tensor) { /* ... */ }
```

## Memory Model

### Stack vs Heap

Currently, all tensors are stack-allocated:

**Advantages:**
- No dynamic allocation overhead
- Automatic memory management (RAII)
- Better cache locality
- Predictable performance

**Limitations:**
- Size limited by stack size (typically 1-8MB)
- Size must be known at compile time
- Cannot resize tensors

### Future: Hybrid Model

Planned support for heap allocation:
```cpp
// Future API
template<typename T>
class DynamicTensor;  // Heap-allocated, runtime dimensions
```

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Element access | O(1) | Direct array indexing |
| Slice creation | O(1) | Only creates view metadata |
| Element-wise ops | O(n) | n = tensor size |
| Reductions | O(n) | Single pass through data |
| Broadcasting | O(n) | n = result tensor size |
| Reshape | O(n) | Requires data copy |

### Space Complexity

| Structure | Stack Usage | Heap Usage |
|-----------|-------------|------------|
| Tensor<T, Dims...> | sizeof(T) × ∏Dims | 0 |
| TensorView<T, Dims...> | ~16-32 bytes | 0 |

## Extension Points

### Adding New Operations

1. **Element-wise operations**: Add to `operations.hpp`
```cpp
template<typename T, size_t... Dims>
Tensor<T, Dims...> your_operation(const Tensor<T, Dims...>& input) {
    return element_wise_unary_op(input,
        [](const T& val) { return /* your operation */; });
}
```

2. **Reductions**: Follow the pattern in `operations.hpp`
```cpp
template<typename T, size_t... Dims>
T your_reduction(const Tensor<T, Dims...>& input) {
    return std::accumulate(input.begin(), input.end(),
        initial_value, your_binary_op);
}
```

### Adding New Tensor Types

Future tensor types can inherit common functionality:
```cpp
// Potential future design
template<typename T>
class TensorBase {
    // Common interface
};

template<typename T, size_t... Dims>
class Tensor : public TensorBase<T> {
    // Static tensor implementation
};

template<typename T>
class DynamicTensor : public TensorBase<T> {
    // Dynamic tensor implementation
};
```

## Thread Safety

Current implementation:
- **Tensor**: Thread-safe for read-only access
- **TensorView**: Thread-safe for read-only access
- **Modifications**: Not thread-safe (require external synchronization)

## Error Handling

### Compile-Time Errors
Most errors are caught at compile time through template instantiation failures.

### Runtime Errors
Limited to:
- Bounds checking in `at()` method (throws `std::out_of_range`)
- Initializer list size mismatch (throws `std::runtime_error`)

## Testing Strategy

### Unit Tests
- Google Test framework
- Comprehensive coverage of all operations
- Compile-time dimension checking tests
- Edge cases and boundary conditions

### Compile-Time Tests
```cpp
// Static assertions verify compile-time properties
static_assert(Tensor<float, 2, 3>::ndim == 2);
static_assert(Tensor<float, 2, 3>::size == 6);
```

## Future Roadmap

### Near-term
1. Matrix multiplication and dot products
2. More sophisticated broadcasting rules
3. Tensor concatenation and stacking

### Long-term
1. Dynamic tensors with runtime dimensions
2. GPU acceleration (CUDA/OpenCL)
3. Automatic differentiation
4. Expression templates for lazy evaluation
5. SIMD optimizations

## Comparison with Other Libraries

| Feature | EigenWave | Eigen | NumPy | PyTorch |
|---------|-----------|-------|--------|---------|
| Language | C++ | C++ | Python | Python/C++ |
| Compile-time dims | ✓ | Partial | ✗ | ✗ |
| Header-only | ✓ | ✓ | ✗ | ✗ |
| Zero-cost abstractions | ✓ | ✓ | ✗ | Partial |
| Broadcasting | ✓ | Limited | ✓ | ✓ |
| GPU support | Future | Limited | Via CuPy | ✓ |
| Automatic differentiation | Future | ✗ | ✗ | ✓ |