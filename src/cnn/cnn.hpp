#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include "util/tensor.hpp"
#include "util/function.hpp"
#include "cnn/layers.hpp"
#include "util/read_data.hpp"
#include "protos/cnn_params.pb.h"

template <typename T>
class SimpleConvNet {
public:
  SimpleConvNet() : Conv1(Convolution<5, 5, 1, 30, 0, 1, T>((T)0, (T)0.001)),
                    Affine1(Affine<12*12*30, 100, T>((T)0, (T)0.001)),
                    Affine2(Affine<100, 10, T>((T)0, (T)0.001)) {};
  Convolution<5, 5, 1, 30, 0, 1, T> Conv1;
  Relu<T> Relu1;
  Pooling<2, 2, 0, 2, T> Pool1;
  Affine<12*12*30, 100, T> Affine1;
  Relu<T> Relu2;
  Affine<100, 10, T> Affine2;
  Sigmoid<T> Last;
};

template <typename T>
class DoubleConvNet {
public:
  DoubleConvNet() : Conv1(Convolution<5, 5, 1, 20, 0, 1, T>((T)-0.008, (T)0.008)),
                    Conv2(Convolution<5, 5, 20, 50, 0, 1, T>((T)-0.008, (T)0.008)),
                    Affine1(Affine<4*4*50, 10, T>((T)-0.008, (T)0.008)) {};
  Convolution<5, 5, 1, 20, 0, 1, T> Conv1;
  Relu<T> Relu1;
  Pooling<2, 2, 0, 2, T> Pool1;
  Convolution<5, 5, 20, 50, 0, 1, T> Conv2;
  Relu<T> Relu2;
  Pooling<2, 2, 0, 2, T> Pool2;
  Affine<4*4*50, 10, T> Affine1;
  Softmax<T> Last;
};

template <typename T>
class CNN {
public:
  void simple_train(Tensor2D<28, 28, T>& x, Tensor1D<10, T>& t, const T& eps);
  unsigned long simple_predict(Tensor2D<28, 28, T>& x);
  void dc_train(Tensor2D<28, 28, T>& x, Tensor1D<10, T>& t, const T& eps);
  unsigned long dc_predict(Tensor2D<28, 28, T>& x);
  static void run();
private:
  SimpleConvNet<T> simple;
  DoubleConvNet<T> dc;
};

template <typename T>
void CNN<T>::simple_train(Tensor2D<28, 28, T>& x, Tensor1D<10, T>& t, const T& eps) {
  // forward
  Tensor3D<24, 24, 30, T> conv1_ans;
  simple.Conv1.forward(x, &conv1_ans);

  simple.Relu1.forward(&conv1_ans);

  Tensor3D<12, 12, 30, T> pool1_ans;
  Tensor1D<12*12*30, int> idx;
  simple.Pool1.forward(conv1_ans, &pool1_ans, &idx);

  Tensor1D<12*12*30, T> dense1 = pool1_ans.flatten();
  Tensor1D<100, T> dense2;
  simple.Affine1.forward(dense1, &dense2);

  simple.Relu2.forward(&dense2);

  Tensor1D<10, T> ans;
  simple.Affine2.forward(dense2, &ans);

  simple.Last.forward(&ans);

  // Backward
  Tensor1D<10, T> delta3 = ans - t;
  Tensor1D<100, T> delta2;
  simple.Affine2.backward(delta3, dense2, &delta2, eps);
  simple.Relu2.backward(&delta2, dense2);

  Tensor1D<12*12*30, T> delta1;
  simple.Affine1.backward(delta2, dense1, &delta1, eps);

  Tensor3D<12, 12, 30, T> delta1_3D;
  delta1_3D.set_v(delta1.get_v());
  Tensor3D<24, 24, 30, T> delta_pool;
  simple.Pool1.backward(delta1_3D, idx, &delta_pool);

  simple.Relu1.backward(&delta_pool, conv1_ans);

  Tensor2D<28, 28, T> delta_conv;
  simple.Conv1.backward(delta_pool, x, &delta_conv, eps);
}

template <typename T>
unsigned long CNN<T>::simple_predict(Tensor2D<28, 28, T>& x) {
  Tensor3D<24, 24, 30, T> conv1_ans;
  simple.Conv1.forward(x, &conv1_ans);

  simple.Relu1.forward(&conv1_ans);

  Tensor3D<12, 12, 30, T> pool1_ans;
  Tensor1D<12*12*30, int> idx;
  simple.Pool1.forward(conv1_ans, &pool1_ans, &idx);

  Tensor1D<12*12*30, T> dense1 = pool1_ans.flatten();
  Tensor1D<100, T> dense2;
  simple.Affine1.forward(dense1, &dense2);

  simple.Relu2.forward(&dense2);

  Tensor1D<10, T> ans;
  simple.Affine2.forward(dense2, &ans);

  simple.Last.forward(&ans);

  T max = (T)0;
  unsigned long argmax = 0;
  for (int i = 0; i < 10; ++i) {
    if (ans[i] > max) {
      max = ans[i];
      argmax = i;
    }
  }
  return argmax;
}

template <typename T>
void CNN<T>::dc_train(Tensor2D<28, 28, T>& x, Tensor1D<10, T>& t, const T& eps) {
  // forward
  Tensor3D<24, 24, 20, T> conv1_ans;
  dc.Conv1.forward(x, &conv1_ans);

  dc.Relu1.forward(&conv1_ans);

  Tensor3D<12, 12, 20, T> pool1_ans;
  Tensor1D<12*12*20, int> idx1;
  dc.Pool1.forward(conv1_ans, &pool1_ans, &idx1);

  Tensor3D<8, 8, 50, T> conv2_ans;
  dc.Conv2.forward(pool1_ans, &conv2_ans);

  dc.Relu2.forward(&conv2_ans);

  Tensor3D<4, 4, 50, T> pool2_ans;
  Tensor1D<4*4*50, int> idx2;
  dc.Pool2.forward(conv2_ans, &pool2_ans, &idx2);

  Tensor1D<4*4*50, T> dense1 = pool2_ans.flatten();
  Tensor1D<10, T> ans;
  dc.Affine1.forward(dense1, &ans);

  dc.Last.forward(&ans);

  // backward
  Tensor1D<10, T> delta2 = ans - t;
  Tensor1D<4*4*50, T> delta1;
  dc.Affine1.backward(delta2, dense1, &delta1, eps);

  Tensor3D<4, 4, 50, T> delta1_3D;
  delta1_3D.set_v(delta1.get_v());
  Tensor3D<8, 8, 50, T> delta_pool2;
  dc.Pool2.backward(delta1_3D, idx2, &delta_pool2);

  dc.Relu2.backward(&delta_pool2, conv2_ans);

  Tensor3D<12, 12, 20, T> delta_conv2;
  dc.Conv2.backward(delta_pool2, pool1_ans, &delta_conv2, eps);

  Tensor3D<24, 24, 20, T> delta_pool1;
  dc.Pool1.backward(delta_conv2, idx1, &delta_pool1);

  dc.Relu1.backward(&delta_pool1, conv1_ans);

  Tensor2D<28, 28, T> delta_conv1;
  dc.Conv1.backward(delta_pool1, x, &delta_conv1, eps);
}

template <typename T>
unsigned long CNN<T>::dc_predict(Tensor2D<28, 28, T>& x) {
  Tensor3D<24, 24, 20, T> conv1_ans;
  dc.Conv1.forward(x, &conv1_ans);

  dc.Relu1.forward(&conv1_ans);

  Tensor3D<12, 12, 20, T> pool1_ans;
  Tensor1D<12*12*20, int> idx1;
  dc.Pool1.forward(conv1_ans, &pool1_ans, &idx1);

  Tensor3D<8, 8, 50, T> conv2_ans;
  dc.Conv2.forward(pool1_ans, &conv2_ans);

  dc.Relu2.forward(&conv2_ans);

  Tensor3D<4, 4, 50, T> pool2_ans;
  Tensor1D<4*4*50, int> idx2;
  dc.Pool2.forward(conv2_ans, &pool2_ans, &idx2);

  Tensor1D<4*4*50, T> dense1 = pool2_ans.flatten();
  Tensor1D<10, T> ans;
  dc.Affine1.forward(dense1, &ans);

  dc.Last.forward(&ans);

  T max = (T)0;
  unsigned long argmax = 0;
  for (int i = 0; i < 10; ++i) {
    if (ans[i] > max) {
      max = ans[i];
      argmax = i;
    }
  }
  return argmax;
}

template <typename T>
void CNN<T>::run() {
  const data train_X = readMnistImages(TRAIN);
  const data train_y = readMnistLabels(TRAIN);

  const data test_X = readMnistImages(TEST);
  const data test_y = readMnistLabels(TEST);

  Tensor2D<28, 28, T> x;
  Tensor1D<10, T> t;
  CNN<T> cnn;
 
  T eps = (T)0.01;
  int epoch = 1;
  for (int k = 0; k < epoch; ++k) {
    for (int i = 0; i < train_X.col; ++i) {
      x.set_v((float*)train_X.ptr + i * x.size(1) * x.size(0));
      t.set_v(mnistOneHot(((unsigned long*) train_y.ptr)[i]));
      cnn.dc_train(x, t, eps);
    }
    int cnt = 0;
    for (int i = 0; i < test_X.col; ++i) {
      x.set_v((float*)test_X.ptr + i * x.size(1) * x.size(0));
      unsigned long y = cnn.dc_predict(x);
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

