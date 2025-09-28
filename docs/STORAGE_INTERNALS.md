# EigenWave Storage Internals

This document provides detailed technical information about how data storage works in the EigenWave tensor library, covering both the `Tensor` class and the `TensorView` class.

## Table of Contents
1. [Tensor Storage](#tensor-storage)
2. [TensorView Storage](#tensorview-storage)
3. [Memory Layouts](#memory-layouts)
4. [Performance Considerations](#performance-considerations)
5. [Implementation Details](#implementation-details)

## Tensor Storage

### Overview

The `Tensor` class uses **owned, contiguous storage** with compile-time fixed dimensions. All tensor data is stored in a single, flat `std::array` that lives on the stack.

### Storage Declaration

```cpp
template<typename T, size_t... Dims>
class Tensor {
private:
    std::array<T, product_of_dims_v<Dims...>> data_;
    // ...
};
```

### Key Characteristics

1. **Stack Allocation**: Data is allocated on the stack, not the heap
2. **Contiguous Memory**: All elements are stored consecutively in memory
3. **Row-Major Order**: Multi-dimensional data is stored in row-major (C-style) order
4. **Compile-Time Size**: Total storage size is computed at compile time
5. **Zero-Cost Abstraction**: No runtime overhead for dimension tracking

### Memory Layout Example

For a `Tensor<float, 2, 3>` (2×3 matrix):

```
Logical view:
┌─────┬─────┬─────┐
│ 0,0 │ 0,1 │ 0,2 │  Row 0
├─────┼─────┼─────┤
│ 1,0 │ 1,1 │ 1,2 │  Row 1
└─────┴─────┴─────┘

Physical memory layout (row-major):
┌─────┬─────┬─────┬─────┬─────┬─────┐
│ 0,0 │ 0,1 │ 0,2 │ 1,0 │ 1,1 │ 1,2 │
└─────┴─────┴─────┴─────┴─────┴─────┘
  ↑                   ↑
  data_[0]           data_[3]
```

### Index Calculation

Multi-dimensional indices are converted to flat array indices using strides:

```cpp
// For tensor(i, j, k) in a 3D tensor
flat_index = i * (shape[1] * shape[2]) + j * shape[2] + k
```

The actual implementation:

```cpp
template<size_t... Is>
static constexpr size_t compute_index(std::index_sequence<Is...>,
                                      size_t indices[ndim]) {
    size_t strides[ndim];
    strides[ndim - 1] = 1;
    if constexpr (ndim > 1) {
        for (int i = static_cast<int>(ndim) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    return ((indices[Is] * strides[Is]) + ...);
}
```

### Storage Size

The total storage size is computed at compile time:

```cpp
static constexpr size_t size = product_of_dims_v<Dims...>;
```

For example:
- `Tensor<float, 10>`: 10 × 4 bytes = 40 bytes
- `Tensor<double, 3, 3>`: 9 × 8 bytes = 72 bytes
- `Tensor<int, 2, 3, 4>`: 24 × 4 bytes = 96 bytes

## TensorView Storage

### Overview

The `TensorView` class provides **non-owning views** into tensor data with custom strides, enabling slicing and submatrix extraction without copying data.

### Storage Declaration

```cpp
template<typename T, size_t... ViewDims>
class TensorView {
private:
    T* data_ptr_;                           // Pointer to first element
    std::array<size_t, ndim> strides_;      // Stride for each dimension
    // ...
};
```

### Key Characteristics

1. **Non-Owning**: Views don't own the data, just reference it
2. **Custom Strides**: Can skip elements to implement slicing
3. **Zero-Copy**: Creating a view doesn't copy any tensor data
4. **Mutable Access**: Can modify the original tensor through views
5. **Lifetime Management**: User must ensure original tensor outlives views

### How Views Work

Views use a combination of a base pointer and strides to access elements:

```cpp
// Accessing element (i, j) in a 2D view
element = data_ptr_[i * strides_[0] + j * strides_[1]]
```

### Slicing Examples

#### Row Extraction
```cpp
Tensor<float, 3, 4> matrix;
auto row_view = row(matrix, 1);  // Extract row 1

// Original layout (3×4):
// [0,  1,  2,  3]
// [4,  5,  6,  7]  <- extracted row
// [8,  9, 10, 11]

// View parameters:
// data_ptr_ = &matrix(1, 0) = &matrix.data_[4]
// strides_ = {1}  // Move 1 element for each column
```

#### Column Extraction
```cpp
auto col_view = col(matrix, 2);  // Extract column 2

// Original layout (3×4):
// [0,  1, |2|,  3]
// [4,  5, |6|,  7]
// [8,  9,|10|, 11]

// View parameters:
// data_ptr_ = &matrix(0, 2) = &matrix.data_[2]
// strides_ = {4}  // Move 4 elements to next row
```

#### Submatrix Extraction
```cpp
auto sub = submatrix(matrix, 0, 2, 1, 3);  // Rows [0,2), Cols [1,3)

// Original layout (3×4):
// [0, |1,  2|,  3]
// [4, |5,  6|,  7]
// [8,  9, 10, 11]

// View parameters:
// data_ptr_ = &matrix(0, 1) = &matrix.data_[1]
// strides_ = {4, 1}  // 4 elements to next row, 1 to next column
```

#### Strided Slicing
```cpp
Tensor<int, 10> vec;  // [0,1,2,3,4,5,6,7,8,9]
auto slice = slice(vec, range(1, 9, 2));  // Start=1, Stop=9, Step=2

// Extracts: [1,3,5,7]
// data_ptr_ = &vec.data_[1]
// strides_ = {2}  // Skip every other element
```

## Memory Layouts

### Row-Major vs Column-Major

EigenWave uses **row-major** ordering (C-style), where the last index varies fastest:

```cpp
// For tensor(i, j, k):
// Elements are ordered as:
// (0,0,0), (0,0,1), (0,0,2), ..., (0,1,0), (0,1,1), ...
```

### Stride Calculation

Strides represent the number of elements to skip in the flat array to move one step along each dimension:

```cpp
// For a tensor with shape [D0, D1, D2]:
stride[0] = D1 * D2  // Elements to skip for next row
stride[1] = D2       // Elements to skip for next column
stride[2] = 1        // Elements to skip for next depth
```

### Memory Alignment

Since we use `std::array`, memory alignment follows standard C++ rules:
- Arrays are aligned to the alignment requirement of the element type
- No padding between elements
- Cache-friendly access patterns for contiguous data

## Performance Considerations

### Cache Efficiency

1. **Tensor Class**:
   - Excellent cache locality for sequential access
   - Entire small tensors may fit in L1/L2 cache
   - Predictable memory access patterns

2. **TensorView Class**:
   - Good cache performance for contiguous views (rows)
   - Potentially poor cache performance for strided views (columns)
   - Non-unit strides may cause cache misses

### Access Patterns

```cpp
// Good: Row-wise access (cache-friendly)
for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
        tensor(i, j) = value;  // Sequential memory access
    }
}

// Poor: Column-wise access (cache-unfriendly)
for (size_t j = 0; j < cols; ++j) {
    for (size_t i = 0; i < rows; ++i) {
        tensor(i, j) = value;  // Strided memory access
    }
}
```

### Stack Size Limitations

Since tensors are stack-allocated, large tensors may cause stack overflow:

```cpp
// OK: 1000 floats = 4KB
Tensor<float, 10, 10, 10> small_tensor;

// Dangerous: 1M floats = 4MB (may exceed stack size)
Tensor<float, 100, 100, 100> large_tensor;

// Typical stack sizes:
// - Linux: 8MB default
// - Windows: 1MB default
// - macOS: 8MB default
```

## Implementation Details

### Compile-Time Computations

All dimension-related calculations happen at compile time:

```cpp
// Product of dimensions
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
```

### Iterator Support

Both classes provide STL-compatible iterators for flat access:

```cpp
// Tensor iterators directly wrap array iterators
auto begin() { return data_.begin(); }
auto end() { return data_.end(); }

// TensorView would need custom iterators (not implemented)
// to handle non-contiguous access with strides
```

### Copy Semantics

1. **Tensor Copy**: Deep copy of all data
   ```cpp
   Tensor<float, 2, 2> a{1, 2, 3, 4};
   Tensor<float, 2, 2> b = a;  // Copies all 4 elements
   ```

2. **View Copy**: Shallow copy (copies pointer and strides)
   ```cpp
   auto view1 = row(matrix, 0);
   auto view2 = view1;  // Both views point to same data
   ```

3. **View to Tensor**: Explicit deep copy required
   ```cpp
   auto view = row(matrix, 0);
   auto tensor = view.copy();  // Creates new tensor with copied data
   ```

### Memory Safety

1. **Bounds Checking**:
   - `at()` method performs runtime bounds checking
   - `operator()` does not check bounds (faster but less safe)

2. **Lifetime Management**:
   - Tensors own their data (RAII)
   - Views require manual lifetime management
   - Dangling view example:
   ```cpp
   TensorView<float, 3> get_bad_view() {
       Tensor<float, 3, 3> local;
       return row(local, 0);  // DANGER: Returns view to local tensor
   }  // local is destroyed, view now dangles
   ```

### Template Metaprogramming

The library heavily uses template metaprogramming for compile-time safety:

```cpp
// Compile-time dimension checking
static_assert(t1.ndim == 2);
static_assert(t1.size == 6);
static_assert(t1.shape[0] == 2);

// SFINAE for type traits
template<typename T>
struct is_tensor : std::false_type {};

template<typename T, size_t... Dims>
struct is_tensor<Tensor<T, Dims...>> : std::true_type {};
```

## Best Practices

1. **Use Tensor for owned data**: When you need to store tensor data
2. **Use TensorView for algorithms**: When passing subsets of data to functions
3. **Prefer row-wise access**: Better cache performance
4. **Watch stack size**: Use heap allocation for very large tensors (future feature)
5. **Ensure view lifetime safety**: Original tensor must outlive all its views
6. **Use at() during development**: Switch to operator() for performance-critical code

## Future Improvements

1. **Heap-allocated tensors**: For large data that exceeds stack limits
2. **Custom allocators**: Support for aligned memory, memory pools
3. **Lazy evaluation**: Delay computation for complex expressions
4. **SIMD optimization**: Vectorized operations for better performance
5. **Dynamic views**: Runtime-determined dimensions for views