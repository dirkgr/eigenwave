#include <eigenwave/tensor.hpp>
#include <eigenwave/concatenate.hpp>
#include <iostream>
#include <iomanip>

using namespace eigenwave;

int main() {
    std::cout << "=== EigenWave: Tensor Concatenation and Stacking Demo ===" << std::endl << std::endl;

    // ========== 1D CONCATENATION ==========
    std::cout << "1. ONE-DIMENSIONAL CONCATENATION" << std::endl;
    std::cout << "=================================" << std::endl << std::endl;

    Tensor<int, 3> vec1{1, 2, 3};
    Tensor<int, 4> vec2{4, 5, 6, 7};
    Tensor<int, 2> vec3{8, 9};

    std::cout << "Vector 1: ";
    vec1.print();
    std::cout << "Vector 2: ";
    vec2.print();
    std::cout << "Vector 3: ";
    vec3.print();

    auto concat_1d = concatenate(vec1, vec2, vec3);
    std::cout << "Concatenated result: ";
    concat_1d.print();
    std::cout << std::endl;

    // ========== 2D CONCATENATION ==========
    std::cout << "2. TWO-DIMENSIONAL CONCATENATION" << std::endl;
    std::cout << "=================================" << std::endl << std::endl;

    Tensor<float, 2, 3> mat1{1.0f, 2.0f, 3.0f,
                             4.0f, 5.0f, 6.0f};

    Tensor<float, 1, 3> mat2{7.0f, 8.0f, 9.0f};

    std::cout << "Matrix 1 (2x3):" << std::endl;
    mat1.print();

    std::cout << "Matrix 2 (1x3):" << std::endl;
    mat2.print();

    // Concatenate along axis 0 (rows)
    auto concat_axis0 = concatenate<0>(mat1, mat2);
    std::cout << "Concatenated along axis 0 (vertical):" << std::endl;
    concat_axis0.print();

    // Concatenate along axis 1 (columns)
    Tensor<float, 2, 2> mat3{10.0f, 11.0f,
                             12.0f, 13.0f};

    std::cout << "Matrix 3 (2x2):" << std::endl;
    mat3.print();

    auto concat_axis1 = concatenate<1>(mat1, mat3);
    std::cout << "Matrix 1 concatenated with Matrix 3 along axis 1 (horizontal):" << std::endl;
    concat_axis1.print();
    std::cout << std::endl;

    // ========== STACKING ==========
    std::cout << "3. TENSOR STACKING" << std::endl;
    std::cout << "==================" << std::endl << std::endl;

    Tensor<int, 4> stack_vec1{1, 2, 3, 4};
    Tensor<int, 4> stack_vec2{5, 6, 7, 8};
    Tensor<int, 4> stack_vec3{9, 10, 11, 12};

    std::cout << "Three vectors to stack:" << std::endl;
    std::cout << "Vec 1: ";
    stack_vec1.print();
    std::cout << "Vec 2: ";
    stack_vec2.print();
    std::cout << "Vec 3: ";
    stack_vec3.print();

    auto stacked = stack(stack_vec1, stack_vec2, stack_vec3);
    std::cout << "Stacked into 3x4 matrix:" << std::endl;
    stacked.print();
    std::cout << std::endl;

    // Stack 2D matrices to create 3D tensor
    Tensor<int, 2, 2> stack_mat1{1, 2,
                                  3, 4};
    Tensor<int, 2, 2> stack_mat2{5, 6,
                                  7, 8};

    std::cout << "Stack two 2x2 matrices:" << std::endl;
    std::cout << "Matrix 1:" << std::endl;
    stack_mat1.print();
    std::cout << "Matrix 2:" << std::endl;
    stack_mat2.print();

    auto stacked_3d = stack(stack_mat1, stack_mat2);
    std::cout << "Stacked into 2x2x2 tensor:" << std::endl;
    stacked_3d.print();
    std::cout << std::endl;

    // ========== CONVENIENCE FUNCTIONS ==========
    std::cout << "4. CONVENIENCE FUNCTIONS" << std::endl;
    std::cout << "========================" << std::endl << std::endl;

    // hstack (horizontal stack)
    std::cout << "4.1 Horizontal Stack (hstack):" << std::endl;
    std::cout << "-------------------------------" << std::endl;

    Tensor<float, 3> h_vec1{1.0f, 2.0f, 3.0f};
    Tensor<float, 2> h_vec2{4.0f, 5.0f};

    std::cout << "hstack of two 1D vectors:" << std::endl;
    std::cout << "Vec 1: ";
    h_vec1.print();
    std::cout << "Vec 2: ";
    h_vec2.print();

    auto hstacked = hstack(h_vec1, h_vec2);
    std::cout << "Result: ";
    hstacked.print();
    std::cout << std::endl;

    Tensor<float, 2, 2> h_mat1{1.0f, 2.0f,
                               3.0f, 4.0f};
    Tensor<float, 2, 3> h_mat2{5.0f, 6.0f, 7.0f,
                               8.0f, 9.0f, 10.0f};

    std::cout << "hstack of two 2D matrices:" << std::endl;
    std::cout << "Matrix 1 (2x2):" << std::endl;
    h_mat1.print();
    std::cout << "Matrix 2 (2x3):" << std::endl;
    h_mat2.print();

    auto hstacked_2d = hstack(h_mat1, h_mat2);
    std::cout << "Result (2x5):" << std::endl;
    hstacked_2d.print();
    std::cout << std::endl;

    // vstack (vertical stack)
    std::cout << "4.2 Vertical Stack (vstack):" << std::endl;
    std::cout << "-----------------------------" << std::endl;

    Tensor<int, 3> v_vec1{1, 2, 3};
    Tensor<int, 3> v_vec2{4, 5, 6};

    std::cout << "vstack of two 1D vectors (creates new axis):" << std::endl;
    std::cout << "Vec 1: ";
    v_vec1.print();
    std::cout << "Vec 2: ";
    v_vec2.print();

    auto vstacked = vstack(v_vec1, v_vec2);
    std::cout << "Result (2x3 matrix):" << std::endl;
    vstacked.print();
    std::cout << std::endl;

    Tensor<int, 2, 3> v_mat1{1, 2, 3,
                             4, 5, 6};
    Tensor<int, 1, 3> v_mat2{7, 8, 9};

    std::cout << "vstack of two 2D matrices:" << std::endl;
    std::cout << "Matrix 1 (2x3):" << std::endl;
    v_mat1.print();
    std::cout << "Matrix 2 (1x3):" << std::endl;
    v_mat2.print();

    auto vstacked_2d = vstack(v_mat1, v_mat2);
    std::cout << "Result (3x3):" << std::endl;
    vstacked_2d.print();
    std::cout << std::endl;

    // dstack (depth stack)
    std::cout << "4.3 Depth Stack (dstack):" << std::endl;
    std::cout << "-------------------------" << std::endl;

    Tensor<float, 2, 2> d_mat1{1.0f, 2.0f,
                               3.0f, 4.0f};
    Tensor<float, 2, 2> d_mat2{5.0f, 6.0f,
                               7.0f, 8.0f};

    std::cout << "dstack of two 2x2 matrices:" << std::endl;
    std::cout << "Matrix 1:" << std::endl;
    d_mat1.print();
    std::cout << "Matrix 2:" << std::endl;
    d_mat2.print();

    auto dstacked = dstack(d_mat1, d_mat2);
    std::cout << "Result (2x2x2 tensor):" << std::endl;
    dstacked.print();
    std::cout << std::endl;

    // ========== PRACTICAL EXAMPLE ==========
    std::cout << "5. PRACTICAL EXAMPLE: BATCH PROCESSING" << std::endl;
    std::cout << "=======================================" << std::endl << std::endl;

    // Simulate collecting data batches
    std::cout << "Collecting sensor data in batches..." << std::endl;

    Tensor<float, 3> batch1{1.1f, 2.2f, 3.3f};  // First batch of 3 readings
    Tensor<float, 3> batch2{4.4f, 5.5f, 6.6f};  // Second batch
    Tensor<float, 3> batch3{7.7f, 8.8f, 9.9f};  // Third batch

    std::cout << "Batch 1: ";
    batch1.print();
    std::cout << "Batch 2: ";
    batch2.print();
    std::cout << "Batch 3: ";
    batch3.print();

    // Stack batches to create a time series matrix
    auto time_series = stack(batch1, batch2, batch3);
    std::cout << "Combined time series (3 time steps x 3 sensors):" << std::endl;
    time_series.print();

    // Compute statistics across time (rows) for each sensor
    std::cout << "\nStatistics for each sensor:" << std::endl;
    for (size_t sensor = 0; sensor < 3; ++sensor) {
        float sum = 0;
        float min_val = time_series(0, sensor);
        float max_val = time_series(0, sensor);

        for (size_t time = 0; time < 3; ++time) {
            float val = time_series(time, sensor);
            sum += val;
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }

        std::cout << "Sensor " << sensor << ": "
                  << "mean=" << std::fixed << std::setprecision(2) << (sum / 3.0f)
                  << ", min=" << min_val
                  << ", max=" << max_val << std::endl;
    }

    std::cout << "\n=== Demo Complete ===" << std::endl;

    return 0;
}