#pragma once

#include <cstdio>
#include <cstring>
#include <cfloat>
#include <cmath>

#include "util/tensor.hpp"

class Function {
public:
  template <int N, int M, typename T>
  static void ReLU(Tensor<N, M, T>* t);
  template <int N, int M, typename T>
  static void deriv_ReLU(Tensor<N, M, T>* t);
  template <int N, int M, typename T>
  static void sigmoid(Tensor<N, M, T>* t);
  template <int N, int M, typename T>
  static void deriv_sigmoid(Tensor<N, M, T>* t);
  template <int N, int M, typename T>
  static void softmax(Tensor<N, M, T>* t);
  template <int N, int M, typename T>
  static void deriv_softmax(Tensor<N, M, T>* t);
  template <int N, int M, typename T, int K, int L>
  static void matmul(Tensor<N, M, T>& t, Tensor<K, M, T>& m, Tensor<L, M, T>* ans);
  template <int N, int M, typename T, int K, int L>
  static void conv2d(Tensor<N, M, T>& t, Tensor<K, M+1, T>& w,
                     Tensor<L, M, T>* ans, int strides);
  template <int N, int M, typename T, int K, int L, int S>
  static void deconv2d(Tensor<N, M, T>& conv, Tensor<K, M+1, T>& w,
                       Tensor<S, M, T>* pad_conv, Tensor<L, M, T>* ans, int strides);
  template <int N, int M, typename T, int L>
  static void max_pool(Tensor<N, M, T>& t, int k_col, int k_row,
                       Tensor<L, M, T>* ans, Tensor<L, 1, int>* idx, int strides);
  template <int N, int M, typename T, int L>
  static void depool(Tensor<N, M, T>& pool, Tensor<N, 1, int>& idx,
                     Tensor<L, M, T>* ans);
  template <int N, int M, int L, typename T>
  static void add_bias(Tensor<N, M, T>* t, Tensor<L, 1, T>& b);
  template <int N, int M, int L, typename T>
  static void padding(Tensor<N, M, T>& before, int pad_size, Tensor<L, M, T>* ans);
};

template <int N, int M, typename T>
void Function::ReLU(Tensor<N, M, T>* t) {
  for (int i = 0; i < N; ++i)
    (*t)[i] = ((*t)[i] > 0) ? (*t)[i] : 0;
}

template <int N, int M, typename T>
void Function::deriv_ReLU(Tensor<N, M, T>* t) {
  for (int i = 0; i < N; ++i)
    (*t)[i] = ((*t)[i] > 0) ? 1 : 0;
}

template <int N, int M, typename T>
void Function::sigmoid(Tensor<N, M, T>* t) {
  for (int i = 0; i < N; ++i)
    (*t)[i] = 1 / (1 + exp(-1 * (*t)[i]));
}

template <typename T>
T uni_sigmoid(T v) {
  return 1 / (1 + exp(-1 * v));
}

template <int N, int M, typename T>
void Function::deriv_sigmoid(Tensor<N, M, T>* t) {
  for (int i = 0; i < N; ++i)
    (*t)[i] = uni_sigmoid((*t)[i]) * (1 - uni_sigmoid((*t)[i]));
}

template <int N, int M, typename T>
void Function::softmax(Tensor<N, M, T>* t) {
  int col = t->shape()[1];
  int row = t->shape()[0];
  T* v = t->get_v();
  for (int l = 0; l < N / (col * row); ++l) {
    for (int k = 0; k < col; ++k) {
      float sum = 0;
      for (int i = 0; i < row; ++i) {
        sum += exp(v[l * (row * col) + k * row + i]);
      }
      for (int j = 0; j < row; ++j) {
        v[l * (row * col) + k * row + j] = exp(v[l * (row * col) + k * row + j]) / sum;
      }
    }
  }
}

template <int N, int M, typename T>
void Function::deriv_softmax(Tensor<N, M, T> *t) {
  int col = t->shape()[1];
  int row = t->shape()[0];
  T* v = t->get_v();
  for (int l = 0; l < N / (col * row); ++l) {
    for (int k = 0; k < col; ++k) {
      float sum = 0;
      for (int i = 0; i < row; ++i) {
        sum += exp(v[l * (row * col) + k * row + i]);
      }
      for (int j = 0; j < row; ++j) {
        int idx = l*(row*col) + k*row + j;
        v[idx] = exp(v[idx]) / sum;
        v[idx] *= 1 - v[idx];
      }
    }
  }
}

template <int N, int M, typename T, int K, int L>
void Function::matmul(Tensor<N, M, T>& t, Tensor<K, M, T>& m,
                      Tensor<L, M, T>* ans) {
  int t_col = t.shape()[1];
  int t_row = t.shape()[0];
  int m_col = m.shape()[1];
  int m_row = m.shape()[0];
  for (int l = 0; l < N / (t_col * t_row); ++l)
    for (int i = 0; i < t_col; ++i)
      for (int k = 0; k < t_row; ++k)
        for (int j = 0; j < m_row; ++j)
          if (k == 0)
            (*ans)[l * (t_col * m_row) + i * m_row + j]
              = t[l * (t_col * t_row) + i * t_row + k]
              * m[l * (m_col * m_row) + k * m_row + j];
          else
            (*ans)[l * (t_col * m_row) + i * m_row + j]
              += t[l * (t_col * t_row) + i * t_row + k]
              * m[l * (m_col * m_row) + k * m_row + j];
}


template <int N, int M, typename T, int K, int L>
void Function::conv2d(Tensor<N, M, T>& t, Tensor<K, M+1, T>& w,
                      Tensor<L, M, T> *ans, int strides) {
  ans->init();
  int* ans_dim = ans->shape();
  int* w_dim = w.shape();
  int* dim = t.shape();
  for (int k = 0; k < ans_dim[2]; ++k)
    for (int i = 0; i < ans_dim[1]; ++i)
      for (int j = 0; j < ans_dim[0]; ++j)

        for (int ch = 0; ch < w_dim[2]; ++ch)
          for (int c = 0; c < w_dim[1]; ++c)
            for (int r = 0; r < w_dim[0]; ++r)
              (*ans)[k*(ans_dim[0]*ans_dim[1]) + i*ans_dim[0] + j] +=
                t[ch*(dim[1]*dim[0]) + (i*strides+c)*dim[0] + (j*strides+r)] *
                w[k*(w_dim[2]*w_dim[1]*w_dim[0]) + ch*(w_dim[1]*w_dim[0])
                  + c*w_dim[0] + r];
}

template <int N, int M, typename T, int K, int L, int S>
void Function::deconv2d(Tensor<N, M, T>& conv, Tensor<K, M+1, T>& w,
              Tensor<S, M, T>* pad_conv, Tensor<L, M, T>* ans, int strides) {
  int pad_size = w.size(0) - 1;
  Function::padding(conv, pad_size, pad_conv);

  ans->init();
  int* ans_dim = ans->shape();
  int* w_dim = w.shape();
  int* dim = pad_conv->shape();
  for (int k = 0; k < ans_dim[2]; ++k)
    for (int i = 0; i < ans_dim[1]; ++i)
      for (int j = 0; j < ans_dim[0]; ++j)

        for (int ch = 0; ch < w_dim[3]; ++ch)
          for (int c = 0; c < w_dim[1]; ++c)
            for (int r = 0; r < w_dim[0]; ++r)
              (*ans)[k*(ans_dim[0]*ans_dim[1]) + i*ans_dim[0] + j] +=
                (*pad_conv)[ch*(dim[1]*dim[0]) + (i*strides+c)*dim[0] + (j*strides+r)] *
                w[ch*(w_dim[2]*w_dim[1]*w_dim[0]) + k*(w_dim[1]*w_dim[0])
                    + (w_dim[1]-1-c)*w_dim[0] + (w_dim[0]-1-r)];
}

template <int N, int M, typename T, int L>
void Function::max_pool(Tensor<N, M, T>& t, int k_col, int k_row,
                        Tensor<L, M, T>* ans, Tensor<L, 1, int>* idx, int strides) {
  int* ans_dim = ans->shape();
  int* dim = t.shape();
  for (int k = 0; k < ans_dim[2]; ++k) {
    for (int i = 0; i < ans_dim[1]; ++i) {
      for (int j = 0; j < ans_dim[0]; ++j){

        // TODO(wakanapo): Make it possible to handle types other than float.
        float max = -FLT_MAX;
        for (int c = 0; c < k_col; ++c)
          for (int r = 0; r < k_row; ++r)
            if (max < t[k*(dim[1]*dim[0]) + (i*strides+c)*dim[0] + (j*strides+r)]) {
              max = (*ans)[k*(ans_dim[1]*ans_dim[0]) + i*ans_dim[0] + j] =
                t[k*(dim[1]*dim[0]) + (i*strides+c)*dim[0] + (j*strides+r)];
              (*idx)[k*(ans_dim[1]*ans_dim[0]) + i*ans_dim[0] + j] =
                k*(dim[1]*dim[0]) + (i*strides+c)*dim[0] + (j*strides+r);
            }

      }
    }
  }
}

template <int N, int M, typename T, int L>
void Function::depool(Tensor<N, M, T>& pool, Tensor<N, 1, int>& idx,
                   Tensor<L, M, T>* ans) {
  ans->init();
  for (int i = 0; i < N; ++i) {
    (*ans)[idx[i]] = pool[i];
  }
}

template <int N, int M, int L, typename T>
void Function::add_bias(Tensor<N, M ,T>* t, Tensor<L, 1, T>& b) {
  int* dim = t->shape();
  int len = N / dim[M-1];
  for (int j = 0; j < dim[M-1]; ++j)
    for (int i = 0; i < len; ++i)
      (*t)[j * len + i] = (*t)[j * len + i] + b[j];
}

template <int N, int M, int L, typename T>
void Function::padding(Tensor<N, M, T>& before, int pad_size, Tensor<L, M, T>* ans) {
  ans->init();
  int col = before.size(1);
  int row = before.size(0);
  for (int k = 0; k < N / (col*row); ++k) {
    for (int i = 0; i < col; ++i) {
      for (int j = 0; j < row; ++j) {
        (*ans)[k*(col+2*pad_size)*(row+2*pad_size)
               + (i+pad_size)*(row+2*pad_size) + (j+pad_size)]
          = before[k*col*row + i*row + j];
      }
    }
  }
}
