#pragma once

#include "tensor.hpp"
#include "function.hpp"
#include "cnn_weight.hpp"
#include "read_data.hpp"

enum status {
  TRAIN,
  TEST
};

enum activation {
  RELU,
  SOFTMAX,
  NONE
};

class CNN {
private:
  Tensor<len_w1, 4, float> w1;
  Tensor<len_b1, 1, float> b1;
  Tensor<len_w2, 2, float> w2;
  Tensor<len_b2, 2, float> b2;
  Tensor<len_w3, 2, float> w3;
  Tensor<len_b3, 2, float> b3;
public:
  CNN() : w1((int*)dim_w1), b1((int*)dim_b1), w2((int*)dim_w2), b2((int*)dim_b2), w3((int*)dim_w3), b3((int*)dim_b3) {};
  void makeWeight();
  template <int N, int M, int L, int K>
  void conv_layer(Tensor<N, 3, float>& x, Tensor<M, 4, float>& w,
                  Tensor<L, 1, float>& b, Tensor<K, 3, float>* ans);
  template <int N, int M>
  void pool_layer(Tensor<N, 3, float>& x, Tensor<M, 3, float>* ans);
  template <int N, int M, int L>
  void fc_layer(Tensor<N, 2, float>& x, Tensor<M, 2, float>& w, Tensor<L, 2, float>& b,
                Tensor<L, 2, float>* ans, activation act);
  void train();
  unsigned long predict(Tensor<784, 3, float> x);
  static void run(status st);
};

void CNN::makeWeight() {
  w1.set_v(w1_raw);
  b1.set_v(b1_raw);
  w2.set_v(w2_raw);
  b2.set_v(b2_raw);
  w3.set_v(w3_raw);
  b3.set_v(b3_raw);
}

template <int N, int M, int L, int K>
void CNN::conv_layer(Tensor<N, 3, float>& x, Tensor<M, 4, float>& w,
                     Tensor<L, 1, float>& b, Tensor<K, 3, float>* ans) {
  Function::conv2d(x, w, ans, 1);
  Function::add_bias(ans, b);
  Function::ReLU(ans);
}

template <int N, int M>
void CNN::pool_layer(Tensor<N, 3, float>& x, Tensor<M, 3, float>* ans) {
  Function::max_pool(x, 2, 2, ans, 2);
}

template <int N, int M, int L>
void CNN::fc_layer(Tensor<N, 2, float> &x, Tensor<M, 2, float>& w,  Tensor<L, 2, float>& b,
                   Tensor<L, 2, float> *ans, activation act) {

  Function::matmul(x, w, ans);
  (*ans) = (*ans) + b;
  if (act == RELU)
    Function::ReLU(ans);
  else if (act == SOFTMAX)
    Function::softmax(ans);
}

unsigned long CNN::predict(Tensor<784, 3, float> x) {
  int dim[] = {24, 24, 30};
  Tensor<24*24*30, 3, float> cnv1(dim);
  conv_layer(x, w1, b1, &cnv1);

  dim[0] = 12; dim[1] = 12; dim[2] = 30;
  Tensor<12*12*30, 3, float> pool1(dim);
  pool_layer(cnv1, &pool1);

  Tensor<12*12*30, 2, float> dense1 = pool1.flatten();
  dim[0] = 100; dim[1] = 1;
  Tensor<100, 2, float> dense2(dim);
  fc_layer(dense1, w2, b2, &dense2, RELU);

  dim[0] = 10; dim[1] = 1;
  Tensor<10, 2, float> ans(dim);
  fc_layer(dense2, w3, b3, &ans, SOFTMAX);

  float max = 0;
  unsigned long argmax = 0;
  for (int i = 0; i < 10; ++i) {
    if (ans[i] > max) {
      max = ans[i];
      argmax = i;
    }
  }
  return argmax;
}

void CNN::run(status st) {
  // This method works on only CPU. It's not supported by Vivado HLS.
  if (st == TEST) {
    const data test_X = readMnistImages();
    const data test_y = readMnistLabels();

    int dim[] = {28, 28, 1};
    Tensor<28*28, 3, float> x(dim);
    CNN cnn;
    cnn.makeWeight();

    int cnt = 0;
    for (int i = 0; i < test_X.col; ++i) {
      x.set_v((float*)test_X.ptr + i * x.size(0));
      unsigned long y = cnn.predict(x);
      printf("predict: %lu, labels: %lu\n", y, ((unsigned long*)test_y.ptr)[i]);
      if (y == ((unsigned long*)test_y.ptr)[i])
        ++cnt;
    }
    std::cout << "Accuracy: " << (float)cnt / (float)test_X.col << std::endl;
    free(test_X.ptr);
    free(test_y.ptr);
  }
}

