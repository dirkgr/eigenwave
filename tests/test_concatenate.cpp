#include <gtest/gtest.h>
#include <eigenwave/tensor.hpp>
#include <eigenwave/concatenate.hpp>

using namespace eigenwave;

// Test fixture for concatenation tests
class ConcatenateTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

// ============================================================================
// 1D Tensor Concatenation Tests
// ============================================================================

TEST_F(ConcatenateTest, Concatenate1D_Basic) {
    Tensor<int, 3> t1{1, 2, 3};
    Tensor<int, 2> t2{4, 5};

    auto result = concatenate(t1, t2);
    static_assert(std::is_same_v<decltype(result), Tensor<int, 5>>);

    EXPECT_EQ(result(0), 1);
    EXPECT_EQ(result(1), 2);
    EXPECT_EQ(result(2), 3);
    EXPECT_EQ(result(3), 4);
    EXPECT_EQ(result(4), 5);
}

TEST_F(ConcatenateTest, Concatenate1D_Multiple) {
    Tensor<float, 2> t1{1.0f, 2.0f};
    Tensor<float, 3> t2{3.0f, 4.0f, 5.0f};
    Tensor<float, 2> t3{6.0f, 7.0f};

    auto result = concatenate(t1, t2, t3);
    static_assert(std::is_same_v<decltype(result), Tensor<float, 7>>);

    EXPECT_FLOAT_EQ(result(0), 1.0f);
    EXPECT_FLOAT_EQ(result(1), 2.0f);
    EXPECT_FLOAT_EQ(result(2), 3.0f);
    EXPECT_FLOAT_EQ(result(3), 4.0f);
    EXPECT_FLOAT_EQ(result(4), 5.0f);
    EXPECT_FLOAT_EQ(result(5), 6.0f);
    EXPECT_FLOAT_EQ(result(6), 7.0f);
}

TEST_F(ConcatenateTest, Concatenate1D_SingleElement) {
    Tensor<int, 1> t1{42};
    Tensor<int, 1> t2{99};

    auto result = concatenate(t1, t2);
    static_assert(std::is_same_v<decltype(result), Tensor<int, 2>>);

    EXPECT_EQ(result(0), 42);
    EXPECT_EQ(result(1), 99);
}

// ============================================================================
// 2D Tensor Concatenation Tests
// ============================================================================

TEST_F(ConcatenateTest, Concatenate2D_Axis0) {
    Tensor<int, 2, 3> t1{1, 2, 3,
                         4, 5, 6};
    Tensor<int, 1, 3> t2{7, 8, 9};

    auto result = concatenate<0>(t1, t2);
    static_assert(std::is_same_v<decltype(result), Tensor<int, 3, 3>>);

    // Check first matrix rows
    EXPECT_EQ(result(0, 0), 1);
    EXPECT_EQ(result(0, 1), 2);
    EXPECT_EQ(result(0, 2), 3);
    EXPECT_EQ(result(1, 0), 4);
    EXPECT_EQ(result(1, 1), 5);
    EXPECT_EQ(result(1, 2), 6);

    // Check appended row
    EXPECT_EQ(result(2, 0), 7);
    EXPECT_EQ(result(2, 1), 8);
    EXPECT_EQ(result(2, 2), 9);
}

TEST_F(ConcatenateTest, Concatenate2D_Axis1) {
    Tensor<int, 2, 2> t1{1, 2,
                         3, 4};
    Tensor<int, 2, 3> t2{5, 6, 7,
                         8, 9, 10};

    auto result = concatenate<1>(t1, t2);
    static_assert(std::is_same_v<decltype(result), Tensor<int, 2, 5>>);

    // Check first row
    EXPECT_EQ(result(0, 0), 1);
    EXPECT_EQ(result(0, 1), 2);
    EXPECT_EQ(result(0, 2), 5);
    EXPECT_EQ(result(0, 3), 6);
    EXPECT_EQ(result(0, 4), 7);

    // Check second row
    EXPECT_EQ(result(1, 0), 3);
    EXPECT_EQ(result(1, 1), 4);
    EXPECT_EQ(result(1, 2), 8);
    EXPECT_EQ(result(1, 3), 9);
    EXPECT_EQ(result(1, 4), 10);
}

// ============================================================================
// Stack Tests
// ============================================================================

TEST_F(ConcatenateTest, Stack1D_Basic) {
    Tensor<float, 3> t1{1.0f, 2.0f, 3.0f};
    Tensor<float, 3> t2{4.0f, 5.0f, 6.0f};

    auto result = stack(t1, t2);
    static_assert(std::is_same_v<decltype(result), Tensor<float, 2, 3>>);

    // Check first row (from t1)
    EXPECT_FLOAT_EQ(result(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(result(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(result(0, 2), 3.0f);

    // Check second row (from t2)
    EXPECT_FLOAT_EQ(result(1, 0), 4.0f);
    EXPECT_FLOAT_EQ(result(1, 1), 5.0f);
    EXPECT_FLOAT_EQ(result(1, 2), 6.0f);
}

TEST_F(ConcatenateTest, Stack1D_Multiple) {
    Tensor<int, 2> t1{1, 2};
    Tensor<int, 2> t2{3, 4};
    Tensor<int, 2> t3{5, 6};

    auto result = stack(t1, t2, t3);
    static_assert(std::is_same_v<decltype(result), Tensor<int, 3, 2>>);

    EXPECT_EQ(result(0, 0), 1);
    EXPECT_EQ(result(0, 1), 2);
    EXPECT_EQ(result(1, 0), 3);
    EXPECT_EQ(result(1, 1), 4);
    EXPECT_EQ(result(2, 0), 5);
    EXPECT_EQ(result(2, 1), 6);
}

TEST_F(ConcatenateTest, Stack2D_Basic) {
    Tensor<int, 2, 2> t1{1, 2,
                         3, 4};
    Tensor<int, 2, 2> t2{5, 6,
                         7, 8};

    auto result = stack(t1, t2);
    static_assert(std::is_same_v<decltype(result), Tensor<int, 2, 2, 2>>);

    // Check first matrix
    EXPECT_EQ(result(0, 0, 0), 1);
    EXPECT_EQ(result(0, 0, 1), 2);
    EXPECT_EQ(result(0, 1, 0), 3);
    EXPECT_EQ(result(0, 1, 1), 4);

    // Check second matrix
    EXPECT_EQ(result(1, 0, 0), 5);
    EXPECT_EQ(result(1, 0, 1), 6);
    EXPECT_EQ(result(1, 1, 0), 7);
    EXPECT_EQ(result(1, 1, 1), 8);
}

// ============================================================================
// Convenience Function Tests
// ============================================================================

TEST_F(ConcatenateTest, HStack1D) {
    Tensor<double, 3> t1{1.0, 2.0, 3.0};
    Tensor<double, 2> t2{4.0, 5.0};

    auto result = hstack(t1, t2);
    static_assert(std::is_same_v<decltype(result), Tensor<double, 5>>);

    EXPECT_DOUBLE_EQ(result(0), 1.0);
    EXPECT_DOUBLE_EQ(result(1), 2.0);
    EXPECT_DOUBLE_EQ(result(2), 3.0);
    EXPECT_DOUBLE_EQ(result(3), 4.0);
    EXPECT_DOUBLE_EQ(result(4), 5.0);
}

TEST_F(ConcatenateTest, HStack2D) {
    Tensor<int, 2, 2> t1{1, 2,
                         3, 4};
    Tensor<int, 2, 1> t2{5,
                         6};

    auto result = hstack(t1, t2);
    static_assert(std::is_same_v<decltype(result), Tensor<int, 2, 3>>);

    EXPECT_EQ(result(0, 0), 1);
    EXPECT_EQ(result(0, 1), 2);
    EXPECT_EQ(result(0, 2), 5);
    EXPECT_EQ(result(1, 0), 3);
    EXPECT_EQ(result(1, 1), 4);
    EXPECT_EQ(result(1, 2), 6);
}

TEST_F(ConcatenateTest, VStack1D) {
    Tensor<float, 3> t1{1.0f, 2.0f, 3.0f};
    Tensor<float, 3> t2{4.0f, 5.0f, 6.0f};

    auto result = vstack(t1, t2);
    static_assert(std::is_same_v<decltype(result), Tensor<float, 2, 3>>);

    EXPECT_FLOAT_EQ(result(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(result(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(result(0, 2), 3.0f);
    EXPECT_FLOAT_EQ(result(1, 0), 4.0f);
    EXPECT_FLOAT_EQ(result(1, 1), 5.0f);
    EXPECT_FLOAT_EQ(result(1, 2), 6.0f);
}

TEST_F(ConcatenateTest, VStack2D) {
    Tensor<int, 1, 3> t1{1, 2, 3};
    Tensor<int, 2, 3> t2{4, 5, 6,
                         7, 8, 9};

    auto result = vstack(t1, t2);
    static_assert(std::is_same_v<decltype(result), Tensor<int, 3, 3>>);

    EXPECT_EQ(result(0, 0), 1);
    EXPECT_EQ(result(0, 1), 2);
    EXPECT_EQ(result(0, 2), 3);
    EXPECT_EQ(result(1, 0), 4);
    EXPECT_EQ(result(1, 1), 5);
    EXPECT_EQ(result(1, 2), 6);
    EXPECT_EQ(result(2, 0), 7);
    EXPECT_EQ(result(2, 1), 8);
    EXPECT_EQ(result(2, 2), 9);
}

TEST_F(ConcatenateTest, DStack2D) {
    Tensor<int, 2, 2> t1{1, 2,
                         3, 4};
    Tensor<int, 2, 2> t2{5, 6,
                         7, 8};

    auto result = dstack(t1, t2);
    static_assert(std::is_same_v<decltype(result), Tensor<int, 2, 2, 2>>);

    // First depth slice
    EXPECT_EQ(result(0, 0, 0), 1);
    EXPECT_EQ(result(0, 1, 0), 2);
    EXPECT_EQ(result(1, 0, 0), 3);
    EXPECT_EQ(result(1, 1, 0), 4);

    // Second depth slice
    EXPECT_EQ(result(0, 0, 1), 5);
    EXPECT_EQ(result(0, 1, 1), 6);
    EXPECT_EQ(result(1, 0, 1), 7);
    EXPECT_EQ(result(1, 1, 1), 8);
}

TEST_F(ConcatenateTest, DStack1D) {
    Tensor<float, 3> t1{1.0f, 2.0f, 3.0f};
    Tensor<float, 3> t2{4.0f, 5.0f, 6.0f};

    auto result = dstack(t1, t2);
    static_assert(std::is_same_v<decltype(result), Tensor<float, 3, 1, 2>>);

    // Check reshaped structure
    EXPECT_FLOAT_EQ(result(0, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(result(0, 0, 1), 4.0f);
    EXPECT_FLOAT_EQ(result(1, 0, 0), 2.0f);
    EXPECT_FLOAT_EQ(result(1, 0, 1), 5.0f);
    EXPECT_FLOAT_EQ(result(2, 0, 0), 3.0f);
    EXPECT_FLOAT_EQ(result(2, 0, 1), 6.0f);
}

// ============================================================================
// Edge Cases and Error Conditions
// ============================================================================

TEST_F(ConcatenateTest, EmptyConcat1D) {
    Tensor<int, 0> empty;
    Tensor<int, 3> t{1, 2, 3};

    auto result = concatenate(empty, t);
    static_assert(std::is_same_v<decltype(result), Tensor<int, 3>>);

    EXPECT_EQ(result(0), 1);
    EXPECT_EQ(result(1), 2);
    EXPECT_EQ(result(2), 3);
}

TEST_F(ConcatenateTest, LargeConcatenation) {
    Tensor<int, 100> t1 = Tensor<int, 100>::zeros();
    Tensor<int, 200> t2 = Tensor<int, 200>::ones();

    auto result = concatenate(t1, t2);
    static_assert(std::is_same_v<decltype(result), Tensor<int, 300>>);

    // Check first part is zeros
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_EQ(result(i), 0);
    }

    // Check second part is ones
    for (size_t i = 100; i < 300; ++i) {
        EXPECT_EQ(result(i), 1);
    }
}

// Test compile-time dimension validation
// These should not compile (uncomment to test):
// TEST_F(ConcatenateTest, InvalidDimensions) {
//     Tensor<int, 2, 3> t1;
//     Tensor<int, 2, 4> t2;
//     // This should fail: cannot concatenate along axis 0 when axis 1 dimensions differ
//     // auto result = concatenate<0>(t1, t2);
// }