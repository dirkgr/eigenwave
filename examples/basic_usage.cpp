#include <eigenwave/tensor.hpp>
#include <eigenwave/operations.hpp>
#include <iostream>
#include <iomanip>

using namespace eigenwave;

int main() {
    std::cout << "=== EigenWave: Static Tensor Library Demo ===" << std::endl << std::endl;

    // Creating tensors
    std::cout << "1. Creating Tensors:" << std::endl;
    std::cout << "--------------------" << std::endl;

    // Default constructor (zero-initialized)
    Tensor<float, 2, 3> zeros;
    std::cout << "Zero tensor:" << std::endl;
    zeros.print();

    // Fill constructor
    Tensor<float, 2, 3> fives(5.0f);
    std::cout << "Tensor filled with 5:" << std::endl;
    fives.print();

    // Initializer list constructor
    Tensor<float, 2, 3> initialized{1.0f, 2.0f, 3.0f,
                                    4.0f, 5.0f, 6.0f};
    std::cout << "Initialized tensor:" << std::endl;
    initialized.print();

    // Factory methods
    auto ones = Tensor<float, 3, 3>::ones();
    std::cout << "Ones tensor (3x3):" << std::endl;
    ones.print();

    // Element-wise operations
    std::cout << "\n2. Element-wise Operations:" << std::endl;
    std::cout << "---------------------------" << std::endl;

    Tensor<float, 2, 3> a{1, 2, 3, 4, 5, 6};
    Tensor<float, 2, 3> b{10, 20, 30, 40, 50, 60};

    std::cout << "Tensor A:" << std::endl;
    a.print();

    std::cout << "Tensor B:" << std::endl;
    b.print();

    auto sum = a + b;
    std::cout << "A + B:" << std::endl;
    sum.print();

    auto diff = b - a;
    std::cout << "B - A:" << std::endl;
    diff.print();

    auto product = a * b;
    std::cout << "A * B (element-wise):" << std::endl;
    product.print();

    auto quotient = b / a;
    std::cout << "B / A (element-wise):" << std::endl;
    quotient.print();

    // Scalar operations
    std::cout << "\n3. Scalar Operations:" << std::endl;
    std::cout << "---------------------" << std::endl;

    auto scaled = a * 10.0f;
    std::cout << "A * 10:" << std::endl;
    scaled.print();

    auto shifted = a + 100.0f;
    std::cout << "A + 100:" << std::endl;
    shifted.print();

    // Mathematical functions
    std::cout << "\n4. Mathematical Functions:" << std::endl;
    std::cout << "--------------------------" << std::endl;

    Tensor<float, 2, 2> values{-4, -1, 1, 4};
    std::cout << "Original values:" << std::endl;
    values.print();

    auto abs_values = abs(values);
    std::cout << "Absolute values:" << std::endl;
    abs_values.print();

    Tensor<float, 2, 2> positive{1, 4, 9, 16};
    auto sqrt_values = sqrt(positive);
    std::cout << "Square roots of [1, 4, 9, 16]:" << std::endl;
    sqrt_values.print();

    Tensor<float, 2, 2> small{0, 1, 2, 3};
    auto exp_values = exp(small);
    std::cout << "Exponentials of [0, 1, 2, 3]:" << std::endl;
    exp_values.print();

    // Trigonometric functions
    std::cout << "\n5. Trigonometric Functions:" << std::endl;
    std::cout << "----------------------------" << std::endl;

    Tensor<float, 1, 4> angles{0, 3.14159f/6, 3.14159f/4, 3.14159f/2};
    std::cout << "Angles (in radians): [0, π/6, π/4, π/2]" << std::endl;

    auto sines = sin(angles);
    std::cout << "sin(angles):" << std::endl;
    sines.print();

    auto cosines = cos(angles);
    std::cout << "cos(angles):" << std::endl;
    cosines.print();

    // Reduction operations
    std::cout << "\n6. Reduction Operations:" << std::endl;
    std::cout << "------------------------" << std::endl;

    Tensor<int, 3, 3> matrix{1, 2, 3,
                             4, 5, 6,
                             7, 8, 9};
    std::cout << "Matrix:" << std::endl;
    matrix.print();

    std::cout << "Sum: " << eigenwave::sum(matrix) << std::endl;
    std::cout << "Product: " << eigenwave::product(matrix) << std::endl;
    std::cout << "Min: " << eigenwave::min(matrix) << std::endl;
    std::cout << "Max: " << eigenwave::max(matrix) << std::endl;
    std::cout << "Mean: " << eigenwave::mean(matrix) << std::endl;

    // Comparison operations
    std::cout << "\n7. Comparison Operations:" << std::endl;
    std::cout << "-------------------------" << std::endl;

    Tensor<int, 2, 3> x{1, 2, 3, 4, 5, 6};
    Tensor<int, 2, 3> y{1, 3, 2, 4, 4, 7};

    std::cout << "X:" << std::endl;
    x.print();
    std::cout << "Y:" << std::endl;
    y.print();

    auto equal = (x == y);
    std::cout << "X == Y:" << std::endl;
    equal.print();

    auto less = (x < y);
    std::cout << "X < Y:" << std::endl;
    less.print();

    // Working with different dimensions
    std::cout << "\n8. Different Tensor Dimensions:" << std::endl;
    std::cout << "--------------------------------" << std::endl;

    // 1D tensor (vector)
    Vector1D<float, 5> vec{1, 2, 3, 4, 5};
    std::cout << "1D Tensor (Vector):" << std::endl;
    vec.print();

    // 3D tensor
    Tensor3D<int, 2, 2, 2> cube;
    int val = 1;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                cube(i, j, k) = val++;
            }
        }
    }
    std::cout << "3D Tensor (2x2x2 cube):" << std::endl;
    cube.print();

    // Compile-time dimension information
    std::cout << "\n9. Compile-time Information:" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    std::cout << "Matrix<float, 3, 4> properties:" << std::endl;
    using Mat3x4 = Matrix<float, 3, 4>;
    std::cout << "  Number of dimensions: " << Mat3x4::ndim << std::endl;
    std::cout << "  Total size: " << Mat3x4::size << std::endl;
    std::cout << "  Shape: [";
    for (size_t i = 0; i < Mat3x4::ndim; ++i) {
        std::cout << Mat3x4::shape[i];
        if (i < Mat3x4::ndim - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // In-place operations
    std::cout << "\n10. In-place Operations:" << std::endl;
    std::cout << "------------------------" << std::endl;

    Tensor<float, 2, 2> inplace{1, 2, 3, 4};
    std::cout << "Original:" << std::endl;
    inplace.print();

    inplace += 10.0f;
    std::cout << "After += 10:" << std::endl;
    inplace.print();

    inplace *= 2.0f;
    std::cout << "After *= 2:" << std::endl;
    inplace.print();

    std::cout << "\n=== Demo Complete ===" << std::endl;

    return 0;
}