# EigenWave

A header-only C++ tensor math library with compile-time dimension checking, inspired by NumPy but with static typing for safer tensor operations.

## Features

- **Compile-time dimension checking**: Tensor dimensions are validated at compile time, preventing runtime errors
- **NumPy-like API**: Familiar interface for Python users
- **Header-only**: Simple integration - just include the headers
- **Zero-cost abstractions**: No runtime overhead for dimension checking
- **Comprehensive operations**: Element-wise operations, mathematical functions, reductions, and more
- **Tensor slicing**: Extract rows, columns, and submatrices with view semantics
- **Broadcasting**: Automatic broadcasting for operations between tensors of compatible shapes
- **Modern C++17**: Uses modern C++ features for clean, efficient code

## Quick Start

### Requirements

- C++17 compatible compiler
- CMake 3.14 or higher
- Google Test (automatically fetched by CMake for tests)

### Building

```bash
mkdir build
cd build
cmake ..
make -j
```

### Running Tests

```bash
./tests/test_tensor
```

### Running Example

```bash
./basic_usage
```

## Usage

```cpp
#include <eigenwave/tensor.hpp>
#include <eigenwave/operations.hpp>
#include <eigenwave/slicing.hpp>
#include <eigenwave/broadcasting.hpp>

using namespace eigenwave;

int main() {
    // Create tensors with compile-time dimensions
    Tensor<float, 2, 3> a{1, 2, 3, 4, 5, 6};
    Tensor<float, 2, 3> b = Tensor<float, 2, 3>::ones();

    // Element-wise operations
    auto c = a + b;
    auto d = a * 2.0f;

    // Mathematical functions
    auto e = sqrt(abs(a));

    // Reductions
    float sum_a = sum(a);
    float mean_a = mean(a);

    // Slicing
    Tensor<float, 4, 5> matrix;
    auto row_view = row(matrix, 2);        // Get row 2
    auto col_view = col(matrix, 3);        // Get column 3
    auto sub = submatrix(matrix, 1, 3, 2, 4); // Get submatrix

    // Broadcasting
    Tensor<float, 3> vec{10, 20, 30};
    auto broadcasted = a + vec;  // Broadcast vector to each row

    // Compile-time dimension checking
    // Tensor<float, 3, 2> wrong_dims;
    // auto error = a + wrong_dims;  // Compilation error!

    return 0;
}
```

## Supported Operations

### Element-wise Operations
- Addition (`+`, `+=`)
- Subtraction (`-`, `-=`)
- Multiplication (`*`, `*=`)
- Division (`/`, `/=`)
- Unary negation (`-`)

### Mathematical Functions
- `abs()`, `sqrt()`, `exp()`, `log()`
- `sin()`, `cos()`, `tan()`
- `pow()`

### Reduction Operations
- `sum()`, `product()`
- `min()`, `max()`
- `mean()`

### Comparison Operations
- `==`, `!=`, `<`, `>`, `<=`, `>=`

### Factory Methods
- `zeros()`, `ones()`, `full(value)`

### Slicing Operations
- `slice()` - Extract a portion of a tensor
- `row()` - Get a specific row from a matrix
- `col()` - Get a specific column from a matrix
- `submatrix()` - Extract a rectangular region
- `range()` - Create slice ranges with start, stop, and step

### Broadcasting Operations
- Automatic broadcasting in arithmetic operations
- `broadcast_1d_to_2d_cols()` - Broadcast vector to matrix columns
- `broadcast_1d_to_2d_rows()` - Broadcast vector to matrix rows
- `reshape()` - Change tensor dimensions (preserving size)
- `expand_dims_front/back()` - Add singleton dimensions
- `squeeze()` - Remove singleton dimensions

## API Documentation

### Creating Tensors

```cpp
// Default constructor (zero-initialized)
Tensor<float, 2, 3> t1;

// Fill constructor
Tensor<float, 2, 3> t2(5.0f);

// Initializer list
Tensor<float, 2, 2> t3{1, 2, 3, 4};

// Factory methods
auto zeros = Tensor<float, 3, 3>::zeros();
auto ones = Tensor<float, 3, 3>::ones();
auto full = Tensor<float, 3, 3>::full(42.0f);
```

### Accessing Elements

```cpp
Tensor<float, 3, 3> tensor;

// Using operator() (no bounds checking)
tensor(0, 0) = 1.0f;
float val = tensor(1, 2);

// Using at() (with bounds checking)
tensor.at(0, 0) = 1.0f;
float val = tensor.at(1, 2);  // throws if out of bounds
```

### Type Aliases

```cpp
// 1D tensors (vectors)
Vector1D<float, 5> vec;

// 2D tensors (matrices)
Matrix<float, 3, 4> mat;

// 3D tensors
Tensor3D<float, 2, 3, 4> t3d;
```

### Slicing Examples

```cpp
Tensor<float, 4, 5> matrix;

// Get views of rows and columns
auto row2 = row(matrix, 2);          // Returns a view of row 2
auto col3 = col(matrix, 3);          // Returns a view of column 3

// Get a submatrix
auto sub = submatrix(matrix, 1, 3, 2, 4);  // Rows [1,3), Cols [2,4)

// 1D slicing with ranges
Tensor<int, 10> vec;
auto slice1 = slice(vec, range(2, 7));      // Elements [2,7)
auto slice2 = slice(vec, range(1, 9, 2));   // Elements 1,3,5,7

// Views can be modified to affect the original tensor
row2.fill(0.0f);  // Sets all elements in row 2 to 0
```

### Broadcasting Examples

```cpp
// Broadcasting vector to matrix
Tensor<float, 2, 3> mat{1, 2, 3, 4, 5, 6};
Tensor<float, 3> vec{10, 20, 30};

auto result = mat + vec;  // Broadcasts vec to each row of mat

// Reshape operations
Tensor<int, 2, 3> original{1, 2, 3, 4, 5, 6};
auto reshaped = reshape<int, 3, 2>(original);  // 2x3 -> 3x2
auto flattened = reshape<int, 6>(original);    // 2x3 -> 6

// Dimension manipulation
Tensor<float, 3> vec1d{1, 2, 3};
auto expanded = expand_dims_front(vec1d);  // Shape: [1, 3]
auto squeezed = squeeze(expanded);         // Back to shape: [3]
```

## Design Philosophy

EigenWave prioritizes compile-time safety and zero-cost abstractions. All dimension checking happens at compile time, meaning:

1. **No runtime overhead**: Dimension checks are performed by the compiler
2. **Early error detection**: Dimension mismatches are caught during compilation
3. **Optimizable code**: The compiler can fully optimize operations knowing all dimensions

## License

This project is open source. Feel free to use it in your projects.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Future Enhancements

Planned features for future versions:
- Matrix multiplication and dot products
- More advanced broadcasting patterns
- Dynamic tensor views with runtime dimensions
- More advanced linear algebra operations
- GPU acceleration support
- Automatic differentiation
- Einsum operations
- Tensor concatenation and stacking