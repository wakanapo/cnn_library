#pragma once

#include "util/tensor.hpp"
#include "util/function.hpp"
#include "mlp/mlp_weight.hpp"
#include "util/read_data.hpp"

class MLP {
private:
  Tensor<len_w1, 2, float> w1;
  Tensor<len_b1, 2, float> b1;
  Tensor<len_w2, 2, float> w2;
  Tensor<len_b2, 2, float> b2;
public:
  MLP() : w1((int*)dim_w1), b1((int*)dim_b1), w2((int*)dim_w2), b2((int*)dim_b2) {};
  void makeWeight();
  void randomWeight();
  void train();
  unsigned long predict(Tensor<784, 2, float>& x);
  void train(Tensor<784, 2, float>& x, Tensor<10, 2, float>& t, const float& eps);
  static void run(status st);
};

void MLP::makeWeight() {
  w1.set_v(w1_);
  b1.set_v(b1_);
  w2.set_v(w2_);
  b2.set_v(b2_);
}

void MLP::randomWeight() {
  w1.randomInit(-0.08, 0.08);
  b1.randomInit(-0.08, 0.08);
  w2.randomInit(-0.08, 0.08);
  b2.randomInit(-0.08, 0.08);
}

unsigned long MLP::predict(Tensor<784, 2, float>& x) {
  int dim_u1[] = {100, 1};
  Tensor<100, 2, float> u1(dim_u1);
  Function::matmul(x, w1, &u1);
  u1 = u1 + b1;
  Tensor<100, 2, float> z1 = u1;
  Function::sigmoid(&z1);

  int dim_u2[] = {10, 1};
  Tensor<10, 2, float>  u2(dim_u2);
  Function::matmul(z1, w2, &u2);
  u2 = u2 + b2;
  Tensor<10, 2, float> z2 = u2;
  Function::softmax(&z2);

  float max = 0;
  unsigned long argmax = 0;
  for (int i = 0; i < 10; ++i) {
    if (z2[i] > max) {
      max = z2[i];
      argmax = i;
    }
  }

  return argmax;
}

void MLP::train(Tensor<784, 2, float>& x, Tensor<10, 2, float>& t, const float& eps) {
  // Forward
  int dim_u1[] = {100, 1};
  Tensor<100, 2, float> u1(dim_u1);
  Function::matmul(x, w1, &u1);
  u1 = u1 + b1;
  Tensor<100, 2, float> z1 = u1;
  Function::sigmoid(&z1);

  int dim_u2[] = {10, 1};
  Tensor<10, 2, float>  u2(dim_u2);
  Function::matmul(z1, w2, &u2);
  u2 = u2 + b2;
  Tensor<10, 2, float> z2 = u2;
  Function::softmax(&z2);

  // Backward
  Tensor<10, 2, float> delta2 = z2 - t;
  
  Tensor<100, 2, float> delta1 = u1;
  Tensor<100, 2, float> temp(dim_u1);
  Tensor<len_w2, 2, float> w2_t = w2.transpose();
  Function::deriv_sigmoid(&delta1);
  Function::matmul(delta2, w2_t, &temp);
  delta1 = delta1 * temp;

  Tensor<len_w1, 2, float> dw1(dim_w1);
  Tensor<784, 2, float> x_t = x.transpose();
  Function::matmul(x_t, delta1, &dw1);
  Tensor<784, 2, float> x_ones(dim_w1);
  for (int i = 0; i < 784; ++i)
    x_ones[i] = 1;
  Tensor<len_b1, 2, float> db1(dim_b1);
  Function::matmul(x_ones, delta1, &db1);
  Tensor<len_w1, 2, float> dw1_n = dw1.times(eps);
  w1 = w1 - dw1_n;
  Tensor<len_b1, 2, float> db1_n = db1.times(eps);
  b1 = b1 - db1_n;

  Tensor<len_w2, 2, float> dw2(dim_w2);
  Tensor<100, 2, float> z1_t = z1.transpose();
  Function::matmul(z1_t, delta2, &dw2);
  Tensor<100, 2, float> z1_ones(dim_w1);
  for (int i = 0; i < 100; ++i)
    z1_ones[i] = 1;
  Tensor<len_b2, 2, float> db2(dim_b2);
  Function::matmul(z1_ones, delta2, &db2);
  Tensor<len_w2, 2, float> dw2_n = dw2.times(eps);
  w2 = w2 - dw2_n;
  Tensor<len_b2, 2, float> db2_n = db2.times(eps);
  b2 = b2 - db2_n;
}

void MLP::run(status st) {
  if (st == TEST) {
    const data test_X = readMnistImages(st);
    const data test_y = readMnistLabels(st);

    int dim[] = {784, 1};
    Tensor<784, 2, float> x(dim);
    MLP mlp;
    mlp.makeWeight();

    int cnt = 0;
    for (int i = 0; i < test_X.col; ++i) {
      x.set_v((float*)test_X.ptr + i * x.size(0));
      unsigned long y = mlp.predict(x);
      printf("predict: %lu, labels: %lu\n", y, ((unsigned long*)test_y.ptr)[i]);
      if (y == ((unsigned long*)test_y.ptr)[i])
        ++cnt;
    }
    std::cout << "Accuracy: " << (float)cnt / (float)test_X.col << std::endl;
    free(test_X.ptr);
    free(test_y.ptr);
  }
  else if (st == TRAIN) {
    const data train_X = readMnistImages(st);
    const data train_y = readMnistLabels(st);

    const data test_X = readMnistImages(TEST);
    const data test_y = readMnistLabels(TEST);

    int x_dim[] = {784, 1};
    Tensor<784, 2, float> x(x_dim);
    int t_dim[] = {10, 1};
    Tensor<10, 2, float> t(t_dim);
    MLP mlp;
    mlp.randomWeight();

    float eps = 0.02;
    int epoch = 25;
    for (int k = 0; k < epoch; ++k) {
      for (int i = 0; i < train_X.col; ++i) {
        x.set_v((float*)train_X.ptr + i * x.size(0));
        t.set_v(mnistOneHot(((unsigned long*) train_y.ptr)[i]));
        mlp.train(x, t, eps);
      }
      std::cout << "Finish Training!" << std::endl;

      int cnt = 0;
      for (int i = 0; i < test_X.col; ++i) {
        x.set_v((float*)test_X.ptr + i * x.size(0));
        unsigned long y = mlp.predict(x);
        if (y == ((unsigned long*)test_y.ptr)[i])
          ++cnt;
      }
      std::cout << "Epoc: " << k << std::endl;
      std::cout << "Accuracy: " << (float)cnt / (float)test_X.col << std::endl;
    }
    free(train_X.ptr);
    free(train_y.ptr);

    free(test_X.ptr);
    free(test_y.ptr);
  }
}
