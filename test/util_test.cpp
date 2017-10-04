#include "cnn/cnn.hpp"
#include "cnn/cnn_weight.hpp"
#include "util/function.hpp"
#include "util/tensor.hpp"
#include "test_array.hpp"
#include "gtest/gtest.h"

int fmemcmp(float* a, float* b, size_t size) {
  for (size_t i = 0 ; i < size / sizeof(float); ++i) {
    if (std::abs(a[i] - b[i]) > 0.0001)
      return i + 1;
  }
  return 0;
}

TEST(MatmulTest, Matmul2D) {
  float x_raw[] = {1, 1, 1,
                   2, 2, 2};
  int x_dim[] = {3, 2};
  Tensor<6, 2, float> x(x_dim);
  x.set_v(x_raw);

  float y_raw[] = {1, 2,
                   2, 3,
                   3, 4};
  int y_dim[] = {2, 3};
  Tensor<6, 2, float> y(y_dim);
  y.set_v(y_raw);

  float expected_raw[] = {6, 9,
                          12, 18};
  int exp_dim[] = {2, 2};
  Tensor<4, 2, float> expected(exp_dim);
  expected.set_v(expected_raw);
  Tensor<4, 2, float> actual(exp_dim);

  Function::matmul(x, y, &actual);
  EXPECT_EQ(fmemcmp(expected.get_v(), actual.get_v(), actual.bytes()), 0);
}

TEST(MatmulTest, Matmul3D) {
  float x_raw[] = {1, 1, 1,
                   2, 2, 2,
                   3, 3, 3,
                   1, 1, 1,
                   2, 2, 2,
                   3, 3, 3};
  int x_dim[] = {3, 3, 2};
  Tensor<18, 3, float> x(x_dim);
  x.set_v(x_raw);

  float y_raw[] = {1, 2,
                   2, 3,
                   3, 4,
                   1, 1,
                   1, 1,
                   1, 1};
  int y_dim[] = {2, 3, 2};
  Tensor<12, 3, float> y(y_dim);
  y.set_v(y_raw);

  float expected_raw[] = {6, 9,
                          12, 18,
                          18, 27,
                          3, 3,
                          6, 6,
                          9, 9};
  int exp_dim[] = {2, 3, 2};
  Tensor<12, 3, float> expected(exp_dim);
  expected.set_v(expected_raw);
  Tensor<12, 3, float> actual(exp_dim);

  Function::matmul(x, y, &actual);
  EXPECT_EQ(fmemcmp(expected.get_v(), actual.get_v(), actual.bytes()), 0);
}

TEST(TransposeTest, Transpose2D) {
  float x_raw[] = {11, 12, 13,
                   21, 22, 23,
                   31, 32, 33};
  int x_dim[] = {3, 3};
  Tensor<9, 2, float> x(x_dim);
  x.set_v(x_raw);

  float expected_raw[] = {11, 21, 31,
                          12, 22, 32,
                          13, 23, 33};
  Tensor<9, 2, float> expected(x_dim);
  expected.set_v(expected_raw);
  Tensor<9, 2, float> actual = x.transpose();

  EXPECT_EQ(fmemcmp(expected.get_v(), actual.get_v(), actual.bytes()), 0);
}

TEST(SoftmaxTest, Softmax2D) {
  float x_raw[] = {1, 1, 1,
                   2, 2, 2,
                   3, 3, 3};
  int x_dim[] = {3, 3};
  Tensor<9, 2, float> x(x_dim);
  x.set_v(x_raw);

  float expected_raw[] = {0.33333333,  0.33333333,  0.33333333,
                          0.33333333,  0.33333333,  0.33333333,
                          0.33333333,  0.33333333,  0.33333333};
  int exp_dim[] = {3, 3};
  Tensor<9, 2, float> expected(exp_dim);
  expected.set_v(expected_raw);

  Function::softmax(&x);
  EXPECT_EQ(fmemcmp(x.get_v(), expected.get_v(), expected.bytes()), 0);
}

TEST(SoftmaxTest, Softmax3D) {
  float x_raw[] = {1, 1, 1,
                   2, 2, 2,
                   3, 3, 3,
                   1, 2, 3,
                   1, 2, 3,
                   1, 2, 3};
  int x_dim[] = {3, 3, 2};
  Tensor<18, 3, float> x(x_dim);
  x.set_v(x_raw);

  float expected_raw[] = {0.33333333, 0.33333333, 0.33333333,
                          0.33333333, 0.33333333, 0.33333333,
                          0.33333333, 0.33333333, 0.33333333,
                          0.09003057, 0.24472847, 0.66524096,
                          0.09003057, 0.24472847, 0.66524096,
                          0.09003057, 0.24472847, 0.66524096};
  Tensor<18, 3, float> expected(x_dim);
  expected.set_v(expected_raw);

  Function::softmax(&x);
  EXPECT_EQ(fmemcmp(x.get_v(), expected.get_v(), expected.bytes()), 0);
}

TEST(SoftmaxTest, DerivSoftmax) {
  float x_raw[] = {1, 1, 1,
                   2, 2, 2,
                   3, 3, 3};
  int x_dim[] = {3, 3};
  Tensor<9, 2, float> x(x_dim);
  x.set_v(x_raw);

  float expected_raw[] = {0.22222222, 0.22222222, 0.22222222,
                          0.22222222, 0.22222222, 0.22222222,
                          0.22222222, 0.22222222, 0.22222222};
  int exp_dim[] = {3, 3};
  Tensor<9, 2, float> expected(exp_dim);
  expected.set_v(expected_raw);

  Function::deriv_softmax(&x);
  EXPECT_EQ(fmemcmp(x.get_v(), expected.get_v(), expected.bytes()), 0);
}

TEST(SigmoidTest, Sigmoid) {
  float x_raw[] = {1, 1, 1,
                   2, 2, 2,
                   3, 3, 3};
  int x_dim[] = {3, 3};
  Tensor<9, 2, float> x(x_dim);
  x.set_v(x_raw);

  float expected_raw[] = { 0.7310586,  0.7310586, 0.7310586,
                           0.88079703, 0.88079703, 0.88079703,
                           0.95257413, 0.95257413, 0.95257413};
  int exp_dim[] = {3, 3};
  Tensor<9, 2, float> expected(exp_dim);
  expected.set_v(expected_raw);

  Function::sigmoid(&x);
  EXPECT_EQ(fmemcmp(x.get_v(), expected.get_v(), expected.bytes()), 0);
}

TEST(SigmoidTest, DerivSigmoid) {
  float x_raw[] = {1, 1, 1,
                   2, 2, 2,
                   3, 3, 3};
  int x_dim[] = {3, 3};
  Tensor<9, 2, float> x(x_dim);
  x.set_v(x_raw);

  float expected_raw[] = {0.19661193,  0.19661193,  0.19661193,
                          0.10499363,  0.10499363,  0.10499363,
                          0.04517666,  0.04517666,  0.04517666};
  int exp_dim[] = {3, 3};
  Tensor<9, 2, float> expected(exp_dim);
  expected.set_v(expected_raw);

  Function::deriv_sigmoid(&x);
  EXPECT_EQ(fmemcmp(x.get_v(), expected.get_v(), expected.bytes()), 0);
}

TEST(ReLUTest, ReLU) {
  float x_raw[] = {0.1, -0.1, 0.1,
                   -2, 2, -2,
                   3, -3, 3};
  int x_dim[] = {3, 3};
  Tensor<9, 2, float> x(x_dim);
  x.set_v(x_raw);

  float expected_raw[] = {0.1, 0, 0.1,
                          0, 2, 0,
                          3, 0, 3};
  int exp_dim[] = {3, 3};
  Tensor<9, 2, float> expected(exp_dim);
  expected.set_v(expected_raw);

  Function::ReLU(&x);
  EXPECT_EQ(fmemcmp(x.get_v(), expected.get_v(), expected.bytes()), 0);
}

TEST(Conv2dTest, Padding1) {
  float x_raw[] = {1, 1, 1, 0, 0,
                   0, 1, 1, 1, 0,
                   0, 0, 1, 1, 1,
                   0, 0, 1, 1, 0,
                   0, 1, 1, 0, 0};
  int x_dim[] = {5, 5, 1};
  Tensor<25, 3, float> x(x_dim);
  x.set_v(x_raw);

  float w_raw[] = {1, 0, 1,
                   0, 1, 0,
                   1, 0, 1};
  int w_dim[] = {3, 3, 1, 1};
  Tensor<9, 4, float> w(w_dim);
  w.set_v(w_raw);

  float expected_raw[] = {4, 3, 4,
                          2, 4, 3,
                          2, 3, 4};
  int exp_dim[] = {3, 3, 1};
  Tensor<9, 3, float> expected(exp_dim);
  expected.set_v(expected_raw);
  Tensor<9, 3, float> actual(exp_dim);

  Function::conv2d(x, w, &actual, 1);
  EXPECT_EQ(fmemcmp(expected.get_v(), actual.get_v(), expected.bytes()), 0);
}

TEST(Conv2dTest, Padding2) {
  float x_raw[] = {1, 1, 1, 0, 0,
                   0, 1, 1, 1, 0,
                   0, 0, 1, 1, 1,
                   0, 0, 1, 1, 0,
                   0, 1, 1, 0, 0};
  int x_dim[] = {5, 5, 1};
  Tensor<25, 3, float> x(x_dim);
  x.set_v(x_raw);

  float w_raw[] = {1, 0, 1,
                   0, 1, 0,
                   1, 0, 1};
  int w_dim[] = {3, 3, 1, 1};
  Tensor<9, 4, float> w(w_dim);
  w.set_v(w_raw);

  float expected_raw[] = {4, 4,
                          2, 4};
  int exp_dim[] = {2, 2, 1};
  Tensor<4, 3, float> expected(exp_dim);
  expected.set_v(expected_raw);
  Tensor<4, 3, float> actual(exp_dim);

  Function::conv2d(x, w, &actual, 2);
  EXPECT_EQ(fmemcmp(expected.get_v(), actual.get_v(), expected.bytes()), 0);
}

TEST(Conv2dTest, Out3D) {
  float x_raw[] = {1, 1, 1, 0, 0,
                   0, 1, 1, 1, 0,
                   0, 0, 1, 1, 1,
                   0, 0, 1, 1, 0,
                   0, 1, 1, 0, 0};
  int x_dim[] = {5, 5, 1};
  Tensor<25, 3, float> x(x_dim);
  x.set_v(x_raw);

  float w_raw[] = {1, 0, 1,
                   0, 1, 0,
                   1, 0, 1,
                   2, 1, 2,
                   1, 2, 1,
                   2, 1, 2,
                   0.2, 0.1, 0.2,
                   0.1, 0.2, 0.1,
                   0.2, 0.1, 0.2};
  int w_dim[] = {3, 3, 1, 3};
  Tensor<27, 4, float> w(w_dim);
  w.set_v(w_raw);

  float expected_raw[] = {4, 3, 4,
                          2, 4, 3,
                          2, 3, 4,
                          10, 10, 10,
                          6, 11, 10,
                          6, 9, 10,
                          1, 1, 1,
                          0.6, 1.1, 1,
                          0.6, 0.9, 1};
  int exp_dim[] = {3, 3, 3};
  Tensor<27, 3, float> expected(exp_dim);
  expected.set_v(expected_raw);
  Tensor<27, 3, float> actual(exp_dim);

  Function::conv2d(x, w, &actual, 1);
  EXPECT_EQ(fmemcmp(expected.get_v(), actual.get_v(), expected.bytes()), 0);
}

TEST(MaxPoolTest, Stride1) {
  float x_raw[] = {77, 80, 82, 78, 70,
                   83, 78, 80, 83, 82,
                   87, 82, 81, 80, 74,
                   87, 87, 85, 77, 66,
                   84, 79, 77, 78, 76};
  int x_dim[] = {5, 5, 1};
  Tensor<25, 3,float> x(x_dim);
  x.set_v(x_raw);

  float expected_raw[] = {83, 82, 83, 83,
                          87, 82, 83, 83,
                          87, 87, 85, 80,
                          87, 87, 85, 78};
  int exp_dim[] = {4, 4, 1};
  Tensor<16, 3, float> expected(exp_dim);
  expected.set_v(expected_raw);
  Tensor<16, 3, float> actual(exp_dim);

  Function::max_pool(x, 2, 2, &actual, 1);
  EXPECT_EQ(memcmp(expected.get_v(), actual.get_v(), expected.bytes()), 0);
}

TEST(MaxPoolTest, Stride2) {
  float x_raw[] = {77, 80, 82, 78, 70,
                   83, 78, 80, 83, 82,
                   87, 82, 81, 80, 74,
                   87, 87, 85, 77, 66,
                   84, 79, 77, 78, 76};
  int x_dim[] = {5, 5, 1};
  Tensor<25, 3, float> x(x_dim);
  x.set_v(x_raw);

  float expected_raw[] = {83, 83,
                          87, 85};
  int exp_dim[] = {2, 2, 1};
  Tensor<4, 3, float> expected(exp_dim);
  expected.set_v(expected_raw);
  Tensor<4, 3, float> actual(exp_dim);

  Function::max_pool(x, 2, 2, &actual, 2);
  EXPECT_EQ(fmemcmp(expected.get_v(), actual.get_v(), expected.bytes()), 0);
}

TEST(MaxPoolTest, Out3D) {
  float x_raw[] = {77, 80, 82, 78, 70,
                   83, 78, 80, 83, 82,
                   87, 82, 81, 80, 74,
                   87, 87, 85, 77, 66,
                   84, 79, 77, 78, 76,
                   77, 80, 82, 78, 70,
                   83, 78, 80, 83, 82,
                   87, 82, 81, 80, 74,
                   87, 87, 85, 77, 66,
                   84, 79, 77, 78, 7};
  int x_dim[] = {5, 5, 2};
  Tensor<50, 3, float> x(x_dim);
  x.set_v(x_raw);

  float expected_raw[] = {83, 83,
                          87, 85,
                          83, 83,
                          87, 85};
  int exp_dim[] = {2, 2, 2};
  Tensor<8, 3, float> expected(exp_dim);
  expected.set_v(expected_raw);
  Tensor<8, 3, float> actual(exp_dim);

  Function::max_pool(x, 2, 2, &actual, 2);
  EXPECT_EQ(fmemcmp(expected.get_v(), actual.get_v(), expected.bytes()), 0);
}

TEST(AddTest, AddBias) {
  float x_raw[] = {1, 1, 1, 1,
                   2, 2, 2, 2,
                   3, 3, 3, 3};
  int x_dim[] = {2, 2, 3};
  Tensor<12, 3, float> x(x_dim);
  x.set_v(x_raw);

  float b_raw[] = {3, 2, 1};
  int b_dim[] = {3};
  Tensor<3, 1, float> b(b_dim);
  b.set_v(b_raw);

  float expected_raw[] = {4, 4, 4, 4,
                          4, 4, 4, 4,
                          4, 4, 4, 4};
  Tensor<12, 3, float> expected(x_dim);
  expected.set_v(expected_raw);

  Function::add_bias(&x, b);
  EXPECT_EQ(fmemcmp(expected.get_v(), x.get_v(), expected.bytes()), 0);
}

TEST(OperatorTest, Add) {
    float x_raw[] = {1, 1, 1,
                     2, 2, 2};
  int x_dim[] = {3, 2};
  Tensor<6, 2, float> x(x_dim);
  x.set_v(x_raw);

  float y_raw[] = {1, 2, 3,
                   1, 2, 3};
  int y_dim[] = {3, 2};
  Tensor<6, 2, float> y(y_dim);
  y.set_v(y_raw);

  float expected_raw[] = {2, 3, 4,
                          3, 4, 5};
  int exp_dim[] = {2, 2};
  Tensor<6, 2, float> expected(exp_dim);
  expected.set_v(expected_raw);
  Tensor<6, 2, float> actual(exp_dim);

  actual = x + y;
  EXPECT_EQ(fmemcmp(expected.get_v(), actual.get_v(), actual.bytes()), 0);
}

TEST(CNNTest, ConvConv2D) {
  int dim[] = {28, 28, 1};
  Tensor<28*28, 3, float> x(dim);
  Tensor<len_w1, 4, float> w1(dim_w1);
  x.set_v(x_raw);
  w1.set_v(w1_raw);

  int ans_dim[] = {24, 24, 30};
  Tensor<24*24*30, 3, float> ans(ans_dim);
  Function::conv2d(x, w1, &ans, 1);
  EXPECT_EQ(fmemcmp(cnv_dot_raw, ans.get_v(), sizeof(cnv_dot_raw)), 0);
}

TEST(CNNTest, ConvAddBias) {
  int ans_dim[] = {24, 24, 30};
  Tensor<24*24*30, 3, float> ans(ans_dim);
  ans.set_v(cnv_dot_raw);

  int b_dim[] = {30};
  Tensor<30, 1, float> b(b_dim);
  b.set_v(b1_raw);

  Function::add_bias(&ans, b);
  EXPECT_EQ(fmemcmp(cnv_add_raw, ans.get_v(), sizeof(cnv_add_raw)), 0);
}

TEST(CNNTest, Conv1) {
  int dim[] = {28, 28, 1};
  Tensor<28*28, 3, float> x(dim);
  Tensor<len_w1, 4, float> w1(dim_w1);
  x.set_v(x_raw);
  w1.set_v(w1_raw);

  int b_dim[] = {30};
  Tensor<30, 1, float> b(b_dim);
  b.set_v(b1_raw);

  int ans_dim[] = {24, 24, 30};
  Tensor<24*24*30, 3, float> ans(ans_dim);
  Function::conv2d(x, w1, &ans, 1);
  Function::add_bias(&ans, b);
  EXPECT_EQ(fmemcmp(conv1_raw, ans.get_v(), sizeof(conv1_raw)), 0);
}

TEST(CNNTest, Relu1) {
  int dim[] = {24, 24, 30};
  Tensor<24*24*30, 3, float> cnv1(dim);
  cnv1.set_v(conv1_raw);

  Function::ReLU(&cnv1);
  EXPECT_EQ(fmemcmp(relu1_raw, cnv1.get_v(), sizeof(relu1_raw)), 0);
}

TEST(CNNTest, Pool1) {
  int dim[] = {24, 24, 30};
  Tensor<24*24*30, 3, float> relu1(dim);
  relu1.set_v(relu1_raw);

  int ans_dim[] = {12, 12, 30};
  Tensor<12*12*30, 3, float> ans(ans_dim);

  Function::max_pool(relu1, 2, 2, &ans, 2);
  EXPECT_EQ(fmemcmp(pool1_raw, ans.get_v(), sizeof(pool1_raw)), 0);
}

TEST(CNNTest, Matmul1) {
  int dim[] = {100, 12*12*30};
  Tensor<100*12*12*30, 2, float> w2(dim);
  w2.set_v(w2_raw);

  int pool1_dim[] = {12*12*30, 1};
  Tensor<12*12*30, 2, float> pool1(pool1_dim);
  pool1.set_v(pool1_raw);

  int ans_dim[] = {100, 1};
  Tensor<100, 2, float> ans(ans_dim);

  Function::matmul(pool1, w2, &ans);
  EXPECT_EQ(fmemcmp(matmul1_raw, ans.get_v(), sizeof(matmul1_raw)), 0);
}

TEST(CNNTest, AddBias1) {
  int b2_dim[] = {100, 1};
  Tensor<100, 2, float> b2(b2_dim);
  b2.set_v(b2_raw);

  int ans_dim[] = {100, 1};
  Tensor<100, 2, float> ans(ans_dim);
  ans.set_v(matmul1_raw);

  ans = ans + b2;
  EXPECT_EQ(fmemcmp(addbias1_raw, ans.get_v(), sizeof(addbias1_raw)), 0);
}

TEST(CNNTest, Affine1) {
  int dim[] = {100, 12*12*30};
  Tensor<100*12*12*30, 2, float> w2(dim);
  w2.set_v(w2_raw);

  int b2_dim[] = {100, 1};
  Tensor<100, 2, float> b2(b2_dim);
  b2.set_v(b2_raw);

  int pool1_dim[] = {12*12*30, 1};
  Tensor<12*12*30, 2, float> pool1(pool1_dim);
  pool1.set_v(pool1_raw);

  int ans_dim[] = {100, 1};
  Tensor<100, 2, float> ans(ans_dim);

  Function::matmul(pool1, w2, &ans);
  ans = ans + b2;
  EXPECT_EQ(fmemcmp(affine1_raw, ans.get_v(), sizeof(affine1_raw)), 0);
}

TEST(CNNTest, Relu2) {
  int dim[] = {100, 1};
  Tensor<100, 2, float> affine1(dim);
  affine1.set_v(affine1_raw);

  Function::ReLU(&affine1);
  EXPECT_EQ(fmemcmp(relu2_raw, affine1.get_v(), sizeof(relu2_raw)), 0);
}

TEST(CNNTest, Matmul2) {
  int w3_dim[] = {10, 100};
  Tensor<100*10, 2, float> w3(w3_dim);
  w3.set_v(w3_raw);

  int relu2_dim[] = {100, 1};
  Tensor<100, 2, float> relu2(relu2_dim);
  relu2.set_v(relu2_raw);

  int ans_dim[] = {10, 1};
  Tensor<10, 2, float> ans(ans_dim);

  Function::matmul(relu2, w3, &ans);
  EXPECT_EQ(fmemcmp(matmul2_raw, ans.get_v(), sizeof(matmul2_raw)), 0);
}

TEST(CNNTest, AddBias2) {
  int b3_dim[] = {10, 1};
  Tensor<10, 2, float> b3(b3_dim);
  b3.set_v(b3_raw);

  int ans_dim[] = {10, 1};
  Tensor<10, 2, float> ans(ans_dim);
  ans.set_v(matmul2_raw);

  ans = ans + b3;
  EXPECT_EQ(fmemcmp(addbias2_raw, ans.get_v(), sizeof(addbias2_raw)), 0);
}

TEST(CNNTest, Affine2) {
  int w3_dim[] = {10, 100};
  Tensor<100*10, 2, float> w3(w3_dim);
  w3.set_v(w3_raw);

  int b3_dim[] = {10, 1};
  Tensor<10, 2, float> b3(b3_dim);
  b3.set_v(b3_raw);

  int relu2_dim[] = {100, 1};
  Tensor<100, 2, float> relu2(relu2_dim);
  relu2.set_v(relu2_raw);

  int ans_dim[] = {10, 1};
  Tensor<10, 2, float> ans(ans_dim);

  Function::matmul(relu2, w3, &ans);
  ans = ans + b3;
  EXPECT_EQ(fmemcmp(affine2_raw, ans.get_v(), sizeof(affine2_raw)), 0);
}

TEST(CNNTest, Predict) {
  int dim[] = {28, 28, 1};
  Tensor<28*28, 3, float> x(dim);
  x.set_v(x_raw);
  CNN cnn;
  cnn.makeWeight();
  
  unsigned long y = cnn.predict(x);
  EXPECT_EQ(y, 7);
}
