#include <eigenwave/tensor.hpp>
#include <eigenwave/matmul.hpp>
#include <iostream>
#include <iomanip>

using namespace eigenwave;

int main() {
    std::cout << "=== EigenWave: Matrix Multiplication and Dot Products Demo ===" << std::endl << std::endl;

    // ========== BASIC MATRIX MULTIPLICATION ==========
    std::cout << "1. BASIC MATRIX MULTIPLICATION" << std::endl;
    std::cout << "===============================" << std::endl << std::endl;

    Tensor<float, 2, 3> A{1.0f, 2.0f, 3.0f,
                          4.0f, 5.0f, 6.0f};

    Tensor<float, 3, 2> B{7.0f,  8.0f,
                          9.0f, 10.0f,
                         11.0f, 12.0f};

    std::cout << "Matrix A (2x3):" << std::endl;
    A.print();

    std::cout << "Matrix B (3x2):" << std::endl;
    B.print();

    auto C = matmul(A, B);
    std::cout << "Result C = A × B (2x2):" << std::endl;
    C.print();
    std::cout << std::endl;

    // ========== TRANSPOSE OPERATIONS ==========
    std::cout << "2. TRANSPOSE OPERATIONS" << std::endl;
    std::cout << "=======================" << std::endl << std::endl;

    Tensor<int, 3, 2> matrix{1, 2,
                             3, 4,
                             5, 6};

    std::cout << "Original matrix (3x2):" << std::endl;
    matrix.print();

    auto transposed = transpose(matrix);
    std::cout << "Transposed matrix (2x3):" << std::endl;
    transposed.print();
    std::cout << std::endl;

    // ========== DOT PRODUCTS ==========
    std::cout << "3. DOT PRODUCTS" << std::endl;
    std::cout << "===============" << std::endl << std::endl;

    // Vector dot product
    Tensor<float, 4> vec1{1.0f, 2.0f, 3.0f, 4.0f};
    Tensor<float, 4> vec2{5.0f, 6.0f, 7.0f, 8.0f};

    std::cout << "Vector 1: ";
    vec1.print();
    std::cout << "Vector 2: ";
    vec2.print();

    float dot_result = dot(vec1, vec2);
    std::cout << "Dot product: " << dot_result << std::endl << std::endl;

    // Matrix-vector multiplication
    Tensor<float, 3, 3> M{1.0f, 0.0f, 2.0f,
                          0.0f, 1.0f, 1.0f,
                          1.0f, 1.0f, 0.0f};

    Tensor<float, 3> v{1.0f, 2.0f, 3.0f};

    std::cout << "Matrix M (3x3):" << std::endl;
    M.print();
    std::cout << "Vector v: ";
    v.print();

    auto Mv = dot(M, v);
    std::cout << "M · v = ";
    Mv.print();
    std::cout << std::endl;

    // ========== IDENTITY AND DIAGONAL MATRICES ==========
    std::cout << "4. SPECIAL MATRICES" << std::endl;
    std::cout << "===================" << std::endl << std::endl;

    auto I = eye<float, 3>();
    std::cout << "3x3 Identity matrix:" << std::endl;
    I.print();

    Tensor<float, 3> diag_values{2.0f, 3.0f, 4.0f};
    auto D = diag(diag_values);
    std::cout << "Diagonal matrix from [2, 3, 4]:" << std::endl;
    D.print();

    // Multiply by diagonal matrix (scaling)
    auto scaled = matmul(D, I);
    std::cout << "Diagonal × Identity:" << std::endl;
    scaled.print();
    std::cout << std::endl;

    // ========== INNER AND OUTER PRODUCTS ==========
    std::cout << "5. INNER AND OUTER PRODUCTS" << std::endl;
    std::cout << "============================" << std::endl << std::endl;

    Tensor<int, 3> u{1, 2, 3};
    Tensor<int, 3> w{4, 5, 6};

    std::cout << "Vector u: ";
    u.print();
    std::cout << "Vector w: ";
    w.print();

    int inner_prod = inner(u, w);
    std::cout << "Inner product (u · w): " << inner_prod << std::endl;

    auto outer_prod = outer(u, w);
    std::cout << "Outer product (u ⊗ w):" << std::endl;
    outer_prod.print();
    std::cout << std::endl;

    // ========== PRACTICAL EXAMPLE: LINEAR TRANSFORMATION ==========
    std::cout << "6. PRACTICAL EXAMPLE: 2D ROTATION" << std::endl;
    std::cout << "==================================" << std::endl << std::endl;

    // Create a 2D rotation matrix (45 degrees)
    float angle = 3.14159f / 4.0f;  // 45 degrees in radians
    float cos_theta = 0.7071f;  // cos(45°)
    float sin_theta = 0.7071f;  // sin(45°)

    Tensor<float, 2, 2> rotation{cos_theta, -sin_theta,
                                 sin_theta,  cos_theta};

    std::cout << "Rotation matrix (45°):" << std::endl;
    rotation.print();

    // Apply rotation to several points
    Tensor<float, 2> point1{1.0f, 0.0f};
    Tensor<float, 2> point2{0.0f, 1.0f};
    Tensor<float, 2> point3{1.0f, 1.0f};

    std::cout << "\nRotating points:" << std::endl;
    std::cout << "Point 1: ";
    point1.print();
    auto rotated1 = matmul(rotation, point1);
    std::cout << "After rotation: ";
    rotated1.print();

    std::cout << "Point 2: ";
    point2.print();
    auto rotated2 = matmul(rotation, point2);
    std::cout << "After rotation: ";
    rotated2.print();

    std::cout << "Point 3: ";
    point3.print();
    auto rotated3 = matmul(rotation, point3);
    std::cout << "After rotation: ";
    rotated3.print();
    std::cout << std::endl;

    // ========== NEURAL NETWORK LAYER SIMULATION ==========
    std::cout << "7. NEURAL NETWORK LAYER" << std::endl;
    std::cout << "========================" << std::endl << std::endl;

    // Simulate a simple neural network layer: output = W·input + bias
    Tensor<float, 3, 4> W{0.5f, -0.2f,  0.1f,  0.3f,   // Weights
                          0.2f,  0.4f, -0.3f,  0.1f,
                         -0.1f,  0.3f,  0.2f, -0.4f};

    Tensor<float, 4> input{1.0f, 0.5f, -0.3f, 0.8f};
    Tensor<float, 3> bias{0.1f, -0.2f, 0.3f};

    std::cout << "Weight matrix W (3x4):" << std::endl;
    W.print();
    std::cout << "Input vector: ";
    input.print();
    std::cout << "Bias vector: ";
    bias.print();

    // Forward pass
    auto linear_output = matmul(W, input);
    std::cout << "W · input = ";
    linear_output.print();

    // Add bias (manual element-wise for now)
    Tensor<float, 3> output;
    for (size_t i = 0; i < 3; ++i) {
        output(i) = linear_output(i) + bias(i);
    }
    std::cout << "Output (W·input + bias) = ";
    output.print();
    std::cout << std::endl;

    // ========== MATRIX CHAIN MULTIPLICATION ==========
    std::cout << "8. MATRIX CHAIN MULTIPLICATION" << std::endl;
    std::cout << "===============================" << std::endl << std::endl;

    Tensor<float, 2, 3> X{1.0f, 2.0f, 3.0f,
                          4.0f, 5.0f, 6.0f};

    Tensor<float, 3, 3> Y{1.0f, 0.0f, 0.0f,
                          0.0f, 2.0f, 0.0f,
                          0.0f, 0.0f, 3.0f};

    Tensor<float, 3, 2> Z{1.0f, 1.0f,
                          2.0f, 2.0f,
                          3.0f, 3.0f};

    std::cout << "Computing X × Y × Z:" << std::endl;
    std::cout << "X (2x3):" << std::endl;
    X.print();
    std::cout << "Y (3x3) - diagonal scaling:" << std::endl;
    Y.print();
    std::cout << "Z (3x2):" << std::endl;
    Z.print();

    auto XY = matmul(X, Y);
    std::cout << "X × Y (2x3):" << std::endl;
    XY.print();

    auto XYZ = matmul(XY, Z);
    std::cout << "Final result (X × Y × Z) (2x2):" << std::endl;
    XYZ.print();
    std::cout << std::endl;

    // ========== TRACE AND DETERMINANT-LIKE OPERATIONS ==========
    std::cout << "9. MATRIX PROPERTIES" << std::endl;
    std::cout << "====================" << std::endl << std::endl;

    Tensor<float, 3, 3> symmetric{2.0f, 1.0f, 0.0f,
                                  1.0f, 3.0f, 1.0f,
                                  0.0f, 1.0f, 2.0f};

    std::cout << "Symmetric matrix:" << std::endl;
    symmetric.print();

    float tr = trace(symmetric);
    std::cout << "Trace (sum of diagonal): " << tr << std::endl;

    auto diag_elements = diag(symmetric);
    std::cout << "Diagonal elements: ";
    diag_elements.print();

    // Verify symmetry: A = A^T
    auto sym_transposed = transpose(symmetric);
    std::cout << "\nVerifying symmetry (A = A^T):" << std::endl;
    bool is_symmetric = true;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (std::abs(symmetric(i, j) - sym_transposed(i, j)) > 1e-6f) {
                is_symmetric = false;
            }
        }
    }
    std::cout << "Matrix is symmetric: " << (is_symmetric ? "true" : "false") << std::endl;

    std::cout << "\n=== Demo Complete ===" << std::endl;

    return 0;
}