#include <eigenwave/tensor.hpp>
#include <eigenwave/operations.hpp>
#include <eigenwave/slicing.hpp>
#include <eigenwave/broadcasting.hpp>
#include <eigenwave/broadcast_view.hpp>
#include <iostream>
#include <iomanip>

using namespace eigenwave;

int main() {
    std::cout << "=== EigenWave: Slicing and Broadcasting Demo ===" << std::endl << std::endl;

    // ========== SLICING DEMONSTRATIONS ==========
    std::cout << "1. TENSOR SLICING" << std::endl;
    std::cout << "=================" << std::endl << std::endl;

    // 1D Tensor Slicing
    std::cout << "1.1 One-dimensional slicing:" << std::endl;
    std::cout << "-----------------------------" << std::endl;

    Tensor<int, 10> vec1d;
    for (int i = 0; i < 10; ++i) {
        vec1d(i) = i * 10;
    }

    std::cout << "Original vector: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << vec1d(i) << " ";
    }
    std::cout << std::endl;

    // Slice [2:7]
    auto slice1 = slice(vec1d, range(2, 7));
    std::cout << "Slice [2:7]: ";
    for (size_t i = 0; i < 5; ++i) {
        std::cout << slice1(i) << " ";
    }
    std::cout << std::endl;

    // Slice with step [1:9:2]
    auto slice2 = slice(vec1d, range(1, 9, 2));
    std::cout << "Slice [1:9:2]: ";
    for (size_t i = 0; i < 4; ++i) {
        std::cout << slice2(i) << " ";
    }
    std::cout << std::endl << std::endl;

    // 2D Tensor Slicing
    std::cout << "1.2 Two-dimensional slicing:" << std::endl;
    std::cout << "-----------------------------" << std::endl;

    Tensor<float, 4, 5> matrix2d;
    float val = 0.0f;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            matrix2d(i, j) = val++;
        }
    }

    std::cout << "Original matrix (4x5):" << std::endl;
    matrix2d.print();

    // Get a single row
    auto row2 = row(matrix2d, 2);
    std::cout << "Row 2: ";
    for (size_t j = 0; j < 5; ++j) {
        std::cout << row2(j) << " ";
    }
    std::cout << std::endl;

    // Get a single column
    auto col3 = col(matrix2d, 3);
    std::cout << "Column 3: ";
    for (size_t i = 0; i < 4; ++i) {
        std::cout << col3(i) << " ";
    }
    std::cout << std::endl;

    // Get a submatrix
    auto sub = submatrix(matrix2d, 1, 3, 1, 4);
    std::cout << "Submatrix [1:3, 1:4]:" << std::endl;
    for (size_t i = 0; i < 2; ++i) {
        std::cout << "  ";
        for (size_t j = 0; j < 3; ++j) {
            std::cout << std::setw(4) << sub(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Modifying through views
    std::cout << "1.3 Modifying through views:" << std::endl;
    std::cout << "-----------------------------" << std::endl;

    Tensor<int, 3, 3> modifiable = Tensor<int, 3, 3>::zeros();
    std::cout << "Initial matrix:" << std::endl;
    modifiable.print();

    // Modify a row
    auto row_view = row(modifiable, 1);
    row_view.fill(5);
    std::cout << "After setting row 1 to 5:" << std::endl;
    modifiable.print();

    // Modify a submatrix - reinitialize to avoid undefined behavior
    modifiable = Tensor<int, 3, 3>::zeros();
    modifiable(0, 0) = 1;
    modifiable(0, 1) = 2;
    modifiable(1, 0) = 3;
    modifiable(1, 1) = 4;
    std::cout << "After setting top-left 2x2 to [1,2,3,4]:" << std::endl;
    modifiable.print();

    // ========== BROADCASTING DEMONSTRATIONS ==========
    std::cout << "\n2. BROADCASTING" << std::endl;
    std::cout << "===============" << std::endl << std::endl;

    std::cout << "2.1 Vector-Matrix Broadcasting:" << std::endl;
    std::cout << "--------------------------------" << std::endl;

    Tensor<float, 2, 3> mat{1, 2, 3, 4, 5, 6};
    Tensor<float, 3> vec{10, 20, 30};

    std::cout << "Matrix (2x3):" << std::endl;
    mat.print();

    std::cout << "Vector (3):" << std::endl;
    std::cout << "[";
    for (size_t i = 0; i < 3; ++i) {
        std::cout << vec(i);
        if (i < 2) std::cout << ", ";
    }
    std::cout << "]" << std::endl << std::endl;

    // Broadcasting addition
    auto sum = mat + vec;
    std::cout << "Matrix + Vector (broadcast):" << std::endl;
    sum.print();

    // Broadcasting multiplication
    auto prod = mat * vec;
    std::cout << "Matrix * Vector (element-wise broadcast):" << std::endl;
    prod.print();

    std::cout << "2.2 Implicit vs Explicit Broadcasting:" << std::endl;
    std::cout << "--------------------------------------" << std::endl;

    Tensor<float, 3> row_vec{1, 2, 3};

    // Implicit broadcasting through operators
    Tensor<float, 2, 3> zeros = Tensor<float, 2, 3>::zeros();
    auto implicit_broadcast = zeros + row_vec;
    std::cout << "Implicit broadcast: zeros + [1,2,3]:" << std::endl;
    implicit_broadcast.print();

    // Explicit broadcasting using views (when you need the broadcast tensor itself)
    auto broadcast_view = broadcast_vector_to_matrix<float, 2, 3>(row_vec);
    auto explicit_broadcast = broadcast_view.materialize();
    std::cout << "Explicit broadcast view materialized:" << std::endl;
    explicit_broadcast.print();

    std::cout << "2.3 Reshape Operations:" << std::endl;
    std::cout << "------------------------" << std::endl;

    Tensor<int, 2, 3> original{1, 2, 3, 4, 5, 6};
    std::cout << "Original (2x3):" << std::endl;
    original.print();

    // Reshape to 3x2
    auto reshaped_3x2 = reshape<int, 3, 2>(original);
    std::cout << "Reshaped to (3x2):" << std::endl;
    reshaped_3x2.print();

    // Reshape to 1D
    auto flattened = reshape<int, 6>(original);
    std::cout << "Flattened to (6):" << std::endl;
    flattened.print();

    std::cout << "2.4 Dimension Manipulation:" << std::endl;
    std::cout << "----------------------------" << std::endl;

    Tensor<float, 3> vec3{1, 2, 3};
    std::cout << "Original vector (3): [1, 2, 3]" << std::endl;

    // Expand dimensions
    auto expanded_front = expand_dims_front(vec3);
    std::cout << "After expand_dims_front (1x3):" << std::endl;
    expanded_front.print();

    auto expanded_back = expand_dims_back(vec3);
    std::cout << "After expand_dims_back (3x1):" << std::endl;
    expanded_back.print();

    // Squeeze dimensions
    auto squeezed = squeeze(expanded_front);
    std::cout << "After squeezing (1x3) back to (3):" << std::endl;
    squeezed.print();

    // ========== COMBINED OPERATIONS ==========
    std::cout << "\n3. COMBINING SLICING AND BROADCASTING" << std::endl;
    std::cout << "======================================" << std::endl << std::endl;

    Tensor<float, 4, 4> large_matrix;
    val = 1.0f;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            large_matrix(i, j) = val++;
        }
    }

    std::cout << "Original 4x4 matrix:" << std::endl;
    large_matrix.print();

    // Extract a row and broadcast it
    auto extracted_row = row(large_matrix, 2).copy();
    std::cout << "Extracted row 2: ";
    for (size_t i = 0; i < 4; ++i) {
        std::cout << extracted_row(i) << " ";
    }
    std::cout << std::endl;

    // Expand dimensions and use in broadcasting
    auto row_for_broadcast = expand_dims_front(extracted_row);
    std::cout << "Row expanded to (1x4) for broadcasting:" << std::endl;
    row_for_broadcast.print();

    // Create another matrix and add the broadcasted row
    Tensor<float, 3, 4> target{1, 1, 1, 1,
                               2, 2, 2, 2,
                               3, 3, 3, 3};

    std::cout << "Target matrix (3x4):" << std::endl;
    target.print();

    // Note: For full broadcasting to work, we'd need to handle dimension matching
    // This is a simplified demonstration

    std::cout << "\n4. PRACTICAL EXAMPLE: NORMALIZATION" << std::endl;
    std::cout << "====================================" << std::endl << std::endl;

    // Create sample data
    Tensor<float, 3, 4> data{1, 2, 3, 4,
                             5, 6, 7, 8,
                             9, 10, 11, 12};

    std::cout << "Original data:" << std::endl;
    data.print();

    // Calculate mean for each column (feature)
    Tensor<float, 4> col_means;
    for (size_t j = 0; j < 4; ++j) {
        float sum = 0;
        for (size_t i = 0; i < 3; ++i) {
            sum += data(i, j);
        }
        col_means(j) = sum / 3.0f;
    }

    std::cout << "Column means: ";
    for (size_t j = 0; j < 4; ++j) {
        std::cout << col_means(j) << " ";
    }
    std::cout << std::endl;

    // Subtract mean from each column (broadcasting)
    auto normalized = data - col_means;
    std::cout << "Data after mean subtraction (broadcasting):" << std::endl;
    normalized.print();

    std::cout << "\n=== Demo Complete ===" << std::endl;

    return 0;
}