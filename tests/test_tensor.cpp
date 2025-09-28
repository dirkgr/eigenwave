#include <gtest/gtest.h>
#include <eigenwave/tensor.hpp>
#include <eigenwave/operations.hpp>
#include <type_traits>

using namespace eigenwave;

// Test fixture for Tensor tests
class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }
};

// Test tensor creation and basic properties
TEST_F(TensorTest, TensorCreation) {
    // Test default constructor
    Tensor<float, 2, 3> t1;
    EXPECT_EQ(t1.size, 6);
    EXPECT_EQ(t1.ndim, 2);
    EXPECT_EQ(t1.shape[0], 2);
    EXPECT_EQ(t1.shape[1], 3);

    // Test fill constructor
    Tensor<int, 3, 3> t2(5);
    for (const auto& val : t2) {
        EXPECT_EQ(val, 5);
    }

    // Test initializer list constructor
    Tensor<double, 2, 2> t3{1.0, 2.0, 3.0, 4.0};
    EXPECT_DOUBLE_EQ(t3(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(t3(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(t3(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(t3(1, 1), 4.0);
}

// Test element access
TEST_F(TensorTest, ElementAccess) {
    Tensor<int, 3, 4> tensor;

    // Fill with sequential values
    int value = 0;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            tensor(i, j) = value++;
        }
    }

    // Test operator()
    EXPECT_EQ(tensor(0, 0), 0);
    EXPECT_EQ(tensor(1, 2), 6);
    EXPECT_EQ(tensor(2, 3), 11);

    // Test at() with bounds checking
    EXPECT_EQ(tensor.at(0, 0), 0);
    EXPECT_EQ(tensor.at(2, 3), 11);

    // Test out of bounds access
    EXPECT_THROW(tensor.at(3, 0), std::out_of_range);
    EXPECT_THROW(tensor.at(0, 4), std::out_of_range);
}

// Test factory methods
TEST_F(TensorTest, FactoryMethods) {
    auto zeros = Tensor<float, 2, 3>::zeros();
    for (const auto& val : zeros) {
        EXPECT_FLOAT_EQ(val, 0.0f);
    }

    auto ones = Tensor<double, 3, 3>::ones();
    for (const auto& val : ones) {
        EXPECT_DOUBLE_EQ(val, 1.0);
    }

    auto full = Tensor<int, 2, 2>::full(42);
    for (const auto& val : full) {
        EXPECT_EQ(val, 42);
    }
}

// Test element-wise addition
TEST_F(TensorTest, Addition) {
    Tensor<int, 2, 3> a{1, 2, 3, 4, 5, 6};
    Tensor<int, 2, 3> b{10, 20, 30, 40, 50, 60};

    // Tensor + Tensor
    auto c = a + b;
    EXPECT_EQ(c(0, 0), 11);
    EXPECT_EQ(c(0, 1), 22);
    EXPECT_EQ(c(1, 2), 66);

    // Tensor + scalar
    auto d = a + 10;
    EXPECT_EQ(d(0, 0), 11);
    EXPECT_EQ(d(0, 1), 12);
    EXPECT_EQ(d(1, 2), 16);

    // scalar + Tensor
    auto e = 100 + a;
    EXPECT_EQ(e(0, 0), 101);
    EXPECT_EQ(e(1, 2), 106);
}

// Test element-wise subtraction
TEST_F(TensorTest, Subtraction) {
    Tensor<float, 2, 2> a{10.0f, 20.0f, 30.0f, 40.0f};
    Tensor<float, 2, 2> b{1.0f, 2.0f, 3.0f, 4.0f};

    auto c = a - b;
    EXPECT_FLOAT_EQ(c(0, 0), 9.0f);
    EXPECT_FLOAT_EQ(c(0, 1), 18.0f);
    EXPECT_FLOAT_EQ(c(1, 0), 27.0f);
    EXPECT_FLOAT_EQ(c(1, 1), 36.0f);

    auto d = a - 5.0f;
    EXPECT_FLOAT_EQ(d(0, 0), 5.0f);
    EXPECT_FLOAT_EQ(d(1, 1), 35.0f);

    auto e = 100.0f - a;
    EXPECT_FLOAT_EQ(e(0, 0), 90.0f);
    EXPECT_FLOAT_EQ(e(1, 1), 60.0f);
}

// Test element-wise multiplication
TEST_F(TensorTest, Multiplication) {
    Tensor<double, 3, 2> a{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    Tensor<double, 3, 2> b{2.0, 2.0, 2.0, 2.0, 2.0, 2.0};

    auto c = a * b;
    EXPECT_DOUBLE_EQ(c(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(c(1, 0), 6.0);
    EXPECT_DOUBLE_EQ(c(2, 1), 12.0);

    auto d = a * 3.0;
    EXPECT_DOUBLE_EQ(d(0, 0), 3.0);
    EXPECT_DOUBLE_EQ(d(2, 1), 18.0);
}

// Test element-wise division
TEST_F(TensorTest, Division) {
    Tensor<float, 2, 2> a{10.0f, 20.0f, 30.0f, 40.0f};
    Tensor<float, 2, 2> b{2.0f, 4.0f, 5.0f, 8.0f};

    auto c = a / b;
    EXPECT_FLOAT_EQ(c(0, 0), 5.0f);
    EXPECT_FLOAT_EQ(c(0, 1), 5.0f);
    EXPECT_FLOAT_EQ(c(1, 0), 6.0f);
    EXPECT_FLOAT_EQ(c(1, 1), 5.0f);

    auto d = a / 10.0f;
    EXPECT_FLOAT_EQ(d(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(d(1, 1), 4.0f);

    auto e = 100.0f / a;
    EXPECT_FLOAT_EQ(e(0, 0), 10.0f);
    EXPECT_FLOAT_EQ(e(1, 1), 2.5f);
}

// Test compound assignment operators
TEST_F(TensorTest, CompoundAssignment) {
    Tensor<int, 2, 3> a{1, 2, 3, 4, 5, 6};
    Tensor<int, 2, 3> b{10, 10, 10, 10, 10, 10};

    a += b;
    EXPECT_EQ(a(0, 0), 11);
    EXPECT_EQ(a(1, 2), 16);

    a -= 5;
    EXPECT_EQ(a(0, 0), 6);
    EXPECT_EQ(a(1, 2), 11);

    a *= 2;
    EXPECT_EQ(a(0, 0), 12);
    EXPECT_EQ(a(1, 2), 22);

    a /= 2;
    EXPECT_EQ(a(0, 0), 6);
    EXPECT_EQ(a(1, 2), 11);
}

// Test mathematical functions
TEST_F(TensorTest, MathFunctions) {
    Tensor<float, 2, 2> a{-1.0f, 2.0f, -3.0f, 4.0f};

    // Test abs
    auto abs_result = abs(a);
    EXPECT_FLOAT_EQ(abs_result(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(abs_result(1, 0), 3.0f);

    // Test sqrt
    Tensor<float, 2, 2> b{4.0f, 9.0f, 16.0f, 25.0f};
    auto sqrt_result = sqrt(b);
    EXPECT_FLOAT_EQ(sqrt_result(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(sqrt_result(0, 1), 3.0f);
    EXPECT_FLOAT_EQ(sqrt_result(1, 0), 4.0f);
    EXPECT_FLOAT_EQ(sqrt_result(1, 1), 5.0f);

    // Test exp and log
    Tensor<double, 2, 2> c{0.0, 1.0, 2.0, 3.0};
    auto exp_result = exp(c);
    EXPECT_NEAR(exp_result(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(exp_result(0, 1), std::exp(1.0), 1e-10);

    auto log_result = log(exp_result);
    EXPECT_NEAR(log_result(0, 0), 0.0, 1e-10);
    EXPECT_NEAR(log_result(0, 1), 1.0, 1e-10);

    // Test trigonometric functions
    Tensor<float, 2, 1> angles{0.0f, 3.14159265f / 2.0f};
    auto sin_result = sin(angles);
    auto cos_result = cos(angles);

    EXPECT_NEAR(sin_result(0, 0), 0.0f, 1e-6f);
    EXPECT_NEAR(sin_result(1, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(cos_result(0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(cos_result(1, 0), 0.0f, 1e-6f);

    // Test pow
    Tensor<double, 2, 2> base{2.0, 3.0, 4.0, 5.0};
    auto pow_result = pow(base, 2.0);
    EXPECT_DOUBLE_EQ(pow_result(0, 0), 4.0);
    EXPECT_DOUBLE_EQ(pow_result(0, 1), 9.0);
    EXPECT_DOUBLE_EQ(pow_result(1, 0), 16.0);
    EXPECT_DOUBLE_EQ(pow_result(1, 1), 25.0);
}

// Test comparison operations
TEST_F(TensorTest, ComparisonOperations) {
    Tensor<int, 2, 3> a{1, 2, 3, 4, 5, 6};
    Tensor<int, 2, 3> b{1, 3, 2, 4, 4, 7};

    auto eq = (a == b);
    EXPECT_TRUE(eq(0, 0));   // 1 == 1
    EXPECT_FALSE(eq(0, 1));  // 2 != 3
    EXPECT_TRUE(eq(1, 0));   // 4 == 4

    auto ne = (a != b);
    EXPECT_FALSE(ne(0, 0));  // 1 == 1
    EXPECT_TRUE(ne(0, 1));   // 2 != 3

    auto lt = (a < b);
    EXPECT_FALSE(lt(0, 0));  // 1 < 1 is false
    EXPECT_TRUE(lt(0, 1));   // 2 < 3 is true
    EXPECT_FALSE(lt(0, 2));  // 3 < 2 is false

    auto gt = (a > b);
    EXPECT_FALSE(gt(0, 0));  // 1 > 1 is false
    EXPECT_FALSE(gt(0, 1));  // 2 > 3 is false
    EXPECT_TRUE(gt(0, 2));   // 3 > 2 is true
    EXPECT_TRUE(gt(1, 1));   // 5 > 4 is true
}

// Test reduction operations
TEST_F(TensorTest, ReductionOperations) {
    Tensor<int, 2, 3> a{1, 2, 3, 4, 5, 6};

    EXPECT_EQ(sum(a), 21);
    EXPECT_EQ(product(a), 720);
    EXPECT_EQ(min(a), 1);
    EXPECT_EQ(max(a), 6);

    Tensor<float, 2, 2> b{1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_FLOAT_EQ(mean(b), 2.5f);
}

// Test unary negation
TEST_F(TensorTest, UnaryNegation) {
    Tensor<int, 2, 2> a{1, -2, 3, -4};
    auto neg = -a;

    EXPECT_EQ(neg(0, 0), -1);
    EXPECT_EQ(neg(0, 1), 2);
    EXPECT_EQ(neg(1, 0), -3);
    EXPECT_EQ(neg(1, 1), 4);
}

// Test compile-time dimension checking
TEST_F(TensorTest, CompileTimeDimensionChecking) {
    // These should compile
    Tensor<float, 2, 3> t1;
    Tensor<float, 2, 3> t2;
    auto t3 = t1 + t2;  // Same dimensions, should work

    // Test that dimensions are checked at compile time
    static_assert(t1.ndim == 2);
    static_assert(t1.size == 6);
    static_assert(t1.shape[0] == 2);
    static_assert(t1.shape[1] == 3);

    // Test type trait
    static_assert(is_tensor_v<decltype(t1)>);
    static_assert(!is_tensor_v<int>);
}

// Test 1D tensors (vectors)
TEST_F(TensorTest, OneDimensionalTensors) {
    Vector1D<float, 5> vec{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    EXPECT_EQ(vec.ndim, 1);
    EXPECT_EQ(vec.size, 5);
    EXPECT_FLOAT_EQ(vec(0), 1.0f);
    EXPECT_FLOAT_EQ(vec(4), 5.0f);

    auto vec2 = vec * 2.0f;
    EXPECT_FLOAT_EQ(vec2(0), 2.0f);
    EXPECT_FLOAT_EQ(vec2(4), 10.0f);

    EXPECT_FLOAT_EQ(sum(vec), 15.0f);
    EXPECT_FLOAT_EQ(mean(vec), 3.0f);
}

// Test 3D tensors
TEST_F(TensorTest, ThreeDimensionalTensors) {
    Tensor3D<int, 2, 2, 3> t3d;

    // Fill with sequential values
    int value = 0;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 3; ++k) {
                t3d(i, j, k) = value++;
            }
        }
    }

    EXPECT_EQ(t3d.ndim, 3);
    EXPECT_EQ(t3d.size, 12);
    EXPECT_EQ(t3d(0, 0, 0), 0);
    EXPECT_EQ(t3d(1, 1, 2), 11);

    auto t3d_plus_10 = t3d + 10;
    EXPECT_EQ(t3d_plus_10(0, 0, 0), 10);
    EXPECT_EQ(t3d_plus_10(1, 1, 2), 21);
}

// Test fill method
TEST_F(TensorTest, FillMethod) {
    Tensor<double, 3, 3> tensor;
    tensor.fill(3.14);

    for (const auto& val : tensor) {
        EXPECT_DOUBLE_EQ(val, 3.14);
    }
}

// Test copy and move semantics
TEST_F(TensorTest, CopyAndMove) {
    Tensor<int, 2, 2> original{1, 2, 3, 4};

    // Test copy constructor
    Tensor<int, 2, 2> copy(original);
    EXPECT_EQ(copy(0, 0), 1);
    EXPECT_EQ(copy(1, 1), 4);

    // Test copy assignment
    Tensor<int, 2, 2> copy_assign;
    copy_assign = original;
    EXPECT_EQ(copy_assign(0, 0), 1);
    EXPECT_EQ(copy_assign(1, 1), 4);

    // Test move constructor
    Tensor<int, 2, 2> moved(std::move(copy));
    EXPECT_EQ(moved(0, 0), 1);
    EXPECT_EQ(moved(1, 1), 4);

    // Test move assignment
    Tensor<int, 2, 2> move_assign;
    move_assign = std::move(copy_assign);
    EXPECT_EQ(move_assign(0, 0), 1);
    EXPECT_EQ(move_assign(1, 1), 4);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}