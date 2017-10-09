#pragma once

#include "util/tensor.hpp"
#include "util/function.hpp"
#include "cnn/cnn_weight.hpp"
#include "util/read_data.hpp"

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
  void randomWeight();
  template <int N, int M, int L, int K>
  void conv_layer(Tensor<N, 3, float>& x, Tensor<M, 4, float>& w,
                  Tensor<L, 1, float>& b, Tensor<K, 3, float>* ans);
  template <int N, int M>
  void pool_layer(Tensor<N, 3, float>& x,
                  Tensor<M, 3, float>* ans, Tensor<M, 1, int>* idx);
  template <int N, int M, int L>
  void fc_layer(Tensor<N, 2, float>& x, Tensor<M, 2, float>& w,
                Tensor<L, 2, float>& b, Tensor<L, 2, float>* ans, activation act);
  template <int N, int M, int L, int K, int U, int S>
  void deconv_layer(Tensor<N, 3, float>& delta, Tensor<M, 3, float>& x,
                    Tensor<L, 4, float>* w, Tensor<U, 1, float>* b,
                    Tensor<S, 3, float>* pad_conv, Tensor<K, 3, float>* ans,
                    const float& eps);
  template <int N, int M>
  void depool_layer(Tensor<N, 3, float>& delta, Tensor<N, 1, int>& idx,
                    Tensor<M, 3, float>* depool);
  template <int N, int M, int L, int S>
  void defc_layer(Tensor<N, 2, float>& delta, Tensor<S, 2, float>& x,
                  Tensor<M, 2, float>* w, Tensor<L, 2, float>* b,
                  Tensor<S, 2, float>* ans, const float& eps);
  void train(Tensor<784, 3, float>& x, Tensor<10, 2, float>& t, const float& eps);
  unsigned long predict(Tensor<784, 3, float>& x);
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


void CNN::randomWeight() {
  w1.randomInit(-0.08, 0.08);
  b1.randomInit(-0.08, 0.08);
  w2.randomInit(-0.08, 0.08);
  b2.randomInit(-0.08, 0.08);
  w3.randomInit(-0.08, 0.08);
  b3.randomInit(-0.08, 0.08);
}

template <int N, int M, int L, int K>
void CNN::conv_layer(Tensor<N, 3, float>& x, Tensor<M, 4, float>& w,
                     Tensor<L, 1, float>& b, Tensor<K, 3, float>* ans) {
  Function::conv2d(x, w, ans, 1);
  Function::add_bias(ans, b);
  Function::ReLU(ans);
}

template <int N, int M>
void CNN::pool_layer(Tensor<N, 3, float>& x,
                     Tensor<M, 3, float>* ans, Tensor<M, 1, int>* idx) {
  Function::max_pool(x, 2, 2, ans, idx, 2);
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

template <int N, int M>
void CNN::depool_layer(Tensor<N, 3, float>& delta, Tensor<N, 1, int>& idx,
                  Tensor<M, 3, float>* ans) {
  Function::depool(delta, idx, ans);
}

template <int N, int M, int L, int K, int U, int S>
void CNN::deconv_layer(Tensor<N, 3, float>& delta, Tensor<M, 3, float>& x,
                       Tensor<L, 4, float>* w, Tensor<U, 1, float>* b,
                       Tensor<S, 3, float>* pad_conv, Tensor<K, 3, float>* ans,
                       const float& eps) {
  Function::deconv2d(x, *w, pad_conv, ans, 1);

  Tensor<U, 1, float> delta_b(b->shape());
  delta_b.init();
  for (int i = 0; i < L; ++i)
    for (int j = 0; j < delta.size(1); ++j)
      for (int h = 0; h < delta.size(0); ++h)
        delta_b[i] = delta_b[i] + delta[i*delta.size(0) + h];

  Tensor<L, 4, float> delta_w(w->shape());
  delta_w.init();
  int* w_dim = w->shape();
  int* d_dim = delta.shape();
  int* x_dim = x.shape();
  for (int i = 0; i < w_dim[3]; ++i)
    for (int j = 0; j < w_dim[2]; ++j)
      for (int k = 0; k < w_dim[1]; ++k)
        for (int l = 0; l < w_dim[0]; ++l)

          for (int c = 0; c < d_dim[1]; ++c)
            for (int r = 0; r < d_dim[0]; ++r)
              delta_w[i*w_dim[0]*w_dim[1]*w_dim[2] +
                      j*w_dim[0]*w_dim[1] + k*w_dim[0] + l]
                += delta[i*d_dim[0]*d_dim[1] + c*d_dim[0] + r] *
                x[j*(x_dim[1]*x_dim[0]) + (k+c)*x_dim[0] + (l+r)];

  (*w) = (*w) - delta_w;
  (*b) = (*b) - delta_b;
}

template <int N, int M, int L, int S>
void CNN::defc_layer(Tensor<N, 2, float>& delta, Tensor<S, 2, float>& x,
                Tensor<M, 2, float>* w, Tensor<L, 2, float>* b,
                Tensor<S, 2, float>* ans, const float& eps) {
  Tensor<M, 2, float> dw(w->shape());
  Tensor<S, 2, float> x_t = x.transpose();
  Function::matmul(x_t, delta, &dw);
  Tensor<100, 2, float> x_ones(x.shape());
  for (int i = 0; i < 100; ++i)
    x_ones[i] = 1;
  Tensor<L, 2, float> db(b->shape());
  Function::matmul(x_ones, delta, &db);
  Tensor<M, 2, float> dw_n = dw.times(eps);
  (*w) = (*w) - dw_n;
  Tensor<L, 2, float> db_n = db.times(eps);
  (*b) = (*b) - db_n;

  Tensor<S, 2, float> temp(ans->shape());
  Tensor<M, 2, float> w_t = w->transpose();
  Function::deriv_sigmoid(&delta);
  Function::matmul(delta, w_t, &temp);
  (*ans) = (*ans) * temp;
}

void CNN::train(Tensor<784, 3, float>& x, Tensor<10, 2, float>& t, const float& eps) {
  // Forward
  int dim[] = {24, 24, 30};
  Tensor<24*24*30, 3, float> cnv1(dim);
  conv_layer(x, w1, b1, &cnv1);

  dim[0] = 12; dim[1] = 12; dim[2] = 30;
  int idx_dim[] = {12*12*30};
  Tensor<12*12*30, 3, float> pool1(dim);
  Tensor<12*12*30, 1, int> idx1(idx_dim);
  pool_layer(cnv1, &pool1, &idx1);

  Tensor<12*12*30, 2, float> dense1 = pool1.flatten();
  dim[0] = 100; dim[1] = 1;
  Tensor<100, 2, float> dense2(dim);
  fc_layer(dense1, w2, b2, &dense2, RELU);

  dim[0] = 10; dim[1] = 1;
  Tensor<10, 2, float> ans(dim);
  fc_layer(dense2, w3, b3, &ans, SOFTMAX);

  // Backward
  Tensor<10, 2, float> delta3 = ans - t;
  Tensor<100, 2, float> delta2(dense2.shape());
  defc_layer(delta3, dense2, &w3, &b3, &delta2, eps);

  Tensor<12*12*30, 2, float> delta1(dense1.shape());
  defc_layer(delta2, dense1, &w2, &b2, &delta1, eps);

  Tensor<12*12*30, 3, float> delta1_3D(pool1.shape());
  delta1_3D.set_v(delta1.get_v());
  Tensor<24*24*30, 3, float> delta_pool(cnv1.shape());
  depool_layer(delta1_3D, idx1, &delta_pool);

  Tensor<28*28, 3, float> delta_cnv(x.shape());
  int pad_dim[] = {32, 32, 1};
  Tensor<32*32, 3, float> pad_conv(pad_dim);
  deconv_layer(delta_pool, x, &w1, &b1, &pad_conv, &delta_cnv, eps);
}

unsigned long CNN::predict(Tensor<784, 3, float>& x) {
  int dim[] = {24, 24, 30};
  Tensor<24*24*30, 3, float> cnv1(dim);
  conv_layer(x, w1, b1, &cnv1);

  dim[0] = 12; dim[1] = 12; dim[2] = 30;
  int idx_dim[] = {12*12*30};
  Tensor<12*12*30, 3, float> pool1(dim);
  Tensor<12*12*30, 1, int> idx1(idx_dim);
  pool_layer(cnv1, &pool1, &idx1);

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
    const data test_X = readMnistImages(TEST);
    const data test_y = readMnistLabels(TEST);

    int dim[] = {28, 28, 1};
    Tensor<28*28, 3, float> x(dim);
    CNN cnn;
    cnn.makeWeight();

    int cnt = 0;
    for (int i = 0; i < test_X.col; ++i) {
      x.set_v((float*)test_X.ptr + i * 28 * x.size(0));
      unsigned long y = cnn.predict(x);
      printf("predict: %lu, labels: %lu\n", y, ((unsigned long*)test_y.ptr)[i]);
      if (y == ((unsigned long*)test_y.ptr)[i])
        ++cnt;
    }
    std::cout << "Accuracy: " << (float)cnt / (float)test_X.col << std::endl;
    free(test_X.ptr);
    free(test_y.ptr);
  }
  else if(st == TRAIN) {
    const data train_X = readMnistImages(st);
    const data train_y = readMnistLabels(st);

    const data test_X = readMnistImages(TEST);
    const data test_y = readMnistLabels(TEST);

    int x_dim[] = {28, 28, 1};
    Tensor<784, 3, float> x(x_dim);
    int t_dim[] = {10, 1};
    Tensor<10, 2, float> t(t_dim);
    CNN cnn;
    cnn.randomWeight();

    float eps = 10e-8;
    int epoch = 25;
    for (int k = 0; k < epoch; ++k) {
      for (int i = 0; i < train_X.col; ++i) {
        x.set_v((float*)train_X.ptr + i * x.size(0));
        t.set_v(mnistOneHot(((unsigned long*) train_y.ptr)[i]));
        cnn.train(x, t, eps);
      }
      int cnt = 0;
      for (int i = 0; i < test_X.col; ++i) {
        x.set_v((float*)test_X.ptr + i * x.size(0));
        unsigned long y = cnn.predict(x);
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

