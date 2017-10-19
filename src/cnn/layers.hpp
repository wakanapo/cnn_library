#pragma once

#include "util/function.hpp"
#include "util/tensor.hpp"
#include "protos/cnn_params.pb.h"

template<int w_row, int w_col, int input, int output, int P, int S, typename T>
class Convolution {
public:
  Convolution(T low, T high);
  void saveParams(CnnProto::Params* p);
  template<int x_row, int x_col, int a_row, int a_col>
  void forward(Tensor3D<x_row, x_col, input, T>& x,
               Tensor3D<a_row, a_col, output, T>* ans);
  template<int x_row, int x_col, int a_row, int a_col>
  void backward(Tensor3D<a_row, a_col, output, T>& delta,
                Tensor3D<x_row, x_col, input, T>& x,
                Tensor3D<x_row, x_col, input, T>* ans, const T& eps);
private:
  int stride_ = S;
  int padding_ = P;
  Tensor4D<w_row, w_col, input, output, T> w_;
  Tensor1D<output, T> b_;
  template<int x_row, int x_col>
  void update_w(Tensor3D<(x_row+2*P-w_row)/S+1, (x_col+2*P-w_col)/S+1, output, T>& delta,
                Tensor3D<x_row, x_col, input, T>& x, const T& eps);
  template<int x_row, int x_col>
  void update_b(Tensor3D<(x_row+2*P-w_row)/S+1, (x_col+2*P-w_col)/S+1, output, T>& delta,
                Tensor3D<x_row, x_col, input, T>& x, const T& eps);
};

template<int w_row, int w_col, int input, int output, int P, int S, typename T>
Convolution<w_row, w_col, input, output, P, S, T>::Convolution(T low, T high) {
  w_.randomInit(low, high);
  b_.init();
}

template<int w_row, int w_col, int input, int output, int P, int S, typename T>
void Convolution<w_row, w_col, input, output, P, S, T>
::saveParams(CnnProto::Params* p) {
  CnnProto::Weight* w = p->add_weights();
  CnnProto::Bias* b = p->add_biases();
  for (int i = 0; i < w_.size(); ++i)
    w->add_w((float)w_[i]);
  for (int i = 0; i < b_.size(); ++i)
    b->add_b((float)b_[i]);
}

template<int w_row, int w_col, int input, int output, int P, int S, typename T>
template<int x_row, int x_col, int a_row, int a_col>
void Convolution<w_row, w_col, input, output, P, S, T>
::forward(Tensor3D<x_row, x_col, input, T> &x,
          Tensor3D<a_row, a_col, output, T> *ans) {
  Function::conv2d(x, w_, ans, P, S);
  Function::add_bias(ans, b_);
}

template<int w_row, int w_col, int input, int output, int P, int S, typename T>
template<int x_row, int x_col, int a_row, int a_col>
void Convolution<w_row, w_col, input, output, P, S, T>
::backward(Tensor3D<a_row, a_col, output, T> &delta,
           Tensor3D<x_row, x_col, input, T> &x,
           Tensor3D<x_row, x_col, input, T> *ans, const T& eps) {
  Function::deconv2d(delta, w_, ans, P, S);
  update_w(delta, x, eps);
  update_b(delta, x, eps);
}

template<int w_row, int w_col, int input, int output, int P, int S, typename T>
template<int x_row, int x_col>
void Convolution<w_row, w_col, input, output, P, S, T>
::update_w(Tensor3D<(x_row+2*P-w_row)/S+1, (x_col+2*P-w_col)/S+1, output, T>& delta,
            Tensor3D<x_row, x_col, input, T>& x, const T& eps) {
  Tensor4D<w_row, w_col, input, output, T> delta_w;
  delta_w.init();
  const int* w_dim = w_.shape();
  const int* d_dim = delta.shape();
  const int* x_dim = x.shape();
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
  
  delta_w = delta_w.times(eps);
  w_ = w_ - delta_w;
}

template<int w_row, int w_col, int input, int output, int P, int S, typename T>
template<int x_row, int x_col>
void Convolution<w_row, w_col, input, output, P, S, T>
::update_b(Tensor3D<(x_row+2*P-w_row)/S+1, (x_col+2*P-w_col)/S+1, output, T>& delta,
            Tensor3D<x_row, x_col, input, T>& x, const T& eps) {
  Tensor1D<output, T> delta_b;
  delta_b.init();
  for (int i = 0; i < delta.size(2); ++i)
    for (int j = 0; j < delta.size(1); ++j)
      for (int h = 0; h < delta.size(0); ++h)
        delta_b[i] = delta_b[i] + delta[i*delta.size(0)*delta.size(1) + j*delta.size(0) + h];

  delta_b = delta_b.times(eps);
  b_ = b_ - delta_b;
}

template<int k_row, int k_col, int P, int S, typename T>
class Pooling {
public:
  template<int dim1, int dim2, int dim3>
  void forward(Tensor3D<dim1, dim2, dim3, T>& x,
               Tensor3D<(dim1+2*P-k_row)/S+1, (dim2+2*P-k_col)/S+1, dim3, T>* ans,
               Tensor1D<((dim1+2*P-k_row)/S+1)*((dim2+2*P-k_col)/S+1)*dim3, int>* idx);
  template<int dim1, int dim2, int dim3>
  void backward(Tensor3D<(dim1+2*P-k_row)/S+1, (dim2+2*P-k_col)/S+1, dim3, T>& delta,
                Tensor1D<((dim1+2*P-k_row)/S+1)*((dim2+2*P-k_col)/S+1)*dim3, int>& idx,
                Tensor3D<dim1, dim2, dim3, T>* ans);
private:
  int stride_ = S;
  int padding_ = P;
};

template<int k_row, int k_col, int P, int S, typename T>
template<int dim1, int dim2, int dim3>
void Pooling<k_row, k_col, P, S, T>
::forward(Tensor3D<dim1, dim2, dim3, T>& x,
          Tensor3D<(dim1+2*P-k_row)/S+1, (dim2+2*P-k_col)/S+1, dim3, T>* ans,
          Tensor1D<((dim1+2*P-k_row)/S+1)*((dim2+2*P-k_col)/S+1)*dim3, int>* idx) {
  Function::max_pool(x, k_row, k_col, ans, idx, P, S);
}

template<int k_row, int k_col, int P, int S, typename T>
template<int dim1, int dim2, int dim3>
void Pooling<k_row, k_col, P, S, T>
::backward(Tensor3D<(dim1+2*P-k_row)/S+1, (dim2+2*P-k_col)/S+1, dim3, T> &delta,
           Tensor1D<((dim1+2*P-k_row)/S+1) * ((dim2+2*P-k_col)/S+1) * dim3, int> &idx,
           Tensor3D<dim1, dim2, dim3, T> *ans) {
  Function::depool(delta, idx, ans);
}

template<int input, int output, typename T>
class Affine {
public:
  Affine(T low, T high);
  void saveParams(CnnProto::Params* p);
  void forward(Tensor1D<input, T>& x, Tensor1D<output, T>* ans);
  void backward(Tensor1D<output, T>& delta, Tensor1D<input, T>& x,
                Tensor1D<input, T>* ans, const T& eps);
private:
  Tensor2D<output, input, T> w_;
  Tensor1D<output, T> b_;
  void update_w(Tensor1D<output, T>& delta, Tensor1D<input, T>& x,
                Tensor1D<input, T>* ans, const T& eps);
  void update_b(Tensor1D<output, T>& delta, Tensor1D<input, T>& x,
                Tensor1D<input, T>* ans, const T& eps);
};

template<int input, int output, typename T>
Affine<input, output, T>::Affine(T low, T high) {
  w_.randomInit(low, high);
  b_.init();
}

template<int input, int output, typename T>
void Affine<input, output, T>::saveParams(CnnProto::Params *p) {
  CnnProto::Weight* w = p->add_weights();
  CnnProto::Bias* b = p->add_biases();
  for (int i = 0; i < w_.size(); ++i)
    w->add_w((float)w_[i]);
  for (int i = 0; i < b_.size(); ++i)
    b->add_b((float)b_[i]);
}

template<int input, int output, typename T>
void Affine<input, output, T>
::forward(Tensor1D<input, T> &x, Tensor1D<output, T> *ans) {
  Function::matmul(x, w_, ans);
  *ans = *ans + b_;
}

template<int input, int output, typename T>
void Affine<input, output, T>
::update_w(Tensor1D<output, T>& delta, Tensor1D<input, T>& x,
           Tensor1D<input, T>* ans, const T& eps) {
  Tensor2D<output, input, T> dw;
  Tensor2D<1, input, T> x_t = x.transpose();
  Function::matmul(x_t, delta, &dw);
  dw = dw.times(eps);
  w_ = w_ - dw;
}

template<int input, int output, typename T>
void Affine<input, output, T>
::update_b(Tensor1D<output, T>& delta, Tensor1D<input, T>& x,
           Tensor1D<input, T>* ans, const T& eps) {
  Tensor1D<1, T> x_ones;
  x_ones[0] = 1;
  Tensor1D<output, T> db;
  Function::matmul(x_ones, delta, &db);
  
  db = db.times(eps);
  b_ = b_ - db;
}

template<int input, int output, typename T>
void Affine<input, output, T>
::backward(Tensor1D<output, T> &delta, Tensor1D<input, T> &x, Tensor1D<input, T> *ans, const T& eps) {
  Tensor2D<input, output, T> w_t = w_.transpose();
  Function::matmul(delta, w_t, ans);
  update_w(delta, x, ans, eps);
  update_b(delta, x, ans, eps);
}

template<typename T>
class Relu {
public:
  template<int dim1, int dim2, int dim3>
  void forward(Tensor3D<dim1, dim2, dim3, T>* x);
  template<int dim1, int dim2, int dim3>
  void backward(Tensor3D<dim1, dim2, dim3, T>* delta,
                Tensor3D<dim1, dim2, dim3, T>& x);
};

template<typename T>
template<int dim1, int dim2, int dim3>
void Relu<T>::forward(Tensor3D<dim1, dim2, dim3, T>* x) {
      Function::ReLU(x);
};

template<typename T>
template<int dim1, int dim2, int dim3>
void Relu<T>::backward(Tensor3D<dim1, dim2, dim3, T> *delta,
                       Tensor3D<dim1, dim2, dim3, T> &x) {
  Tensor3D<dim1, dim2, dim3, T> tmp = x;
  Function::deriv_ReLU(&tmp);
  (*delta) = (*delta) * tmp;
}

template<typename T>
class Sigmoid {
public:
  template<int dim1, int dim2, int dim3>
  void forward(Tensor3D<dim1, dim2, dim3, T>* x);
  template<int dim1, int dim2, int dim3>
  void backward(Tensor3D<dim1, dim2, dim3, T>* delta,
                Tensor3D<dim1, dim2, dim3, T>& x);
};

template<typename T>
template<int dim1, int dim2, int dim3>
void Sigmoid<T>::forward(Tensor3D<dim1, dim2, dim3, T>* x) {
      Function::sigmoid(x);
};

template<typename T>
template<int dim1, int dim2, int dim3>
void Sigmoid<T>::backward(Tensor3D<dim1, dim2, dim3, T> *delta,
                       Tensor3D<dim1, dim2, dim3, T> &x) {
  Tensor3D<dim1, dim2, dim3, T> tmp = x;
  Function::deriv_sigmoid(&tmp);
  (*delta) = (*delta) * tmp;
}

template<typename T>
class Softmax {
public:
  template<int dim1, int dim2, int dim3>
  void forward(Tensor3D<dim1, dim2, dim3, T>* x);
  template<int dim1, int dim2, int dim3>
  void backward(Tensor3D<dim1, dim2, dim3, T>* delta,
                Tensor3D<dim1, dim2, dim3, T>& x);
};

template<typename T>
template<int dim1, int dim2, int dim3>
void Softmax<T>::forward(Tensor3D<dim1, dim2, dim3, T>* x) {
      Function::softmax(x);
};

template<typename T>
template<int dim1, int dim2, int dim3>
void Softmax<T>::backward(Tensor3D<dim1, dim2, dim3, T> *delta,
                       Tensor3D<dim1, dim2, dim3, T> &x) {
  Tensor3D<dim1, dim2, dim3, T> tmp = x;
  Function::deriv_softmax(&tmp);
  (*delta) = (*delta) * tmp;
}
