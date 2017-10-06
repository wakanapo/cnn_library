#pragma once

#include <iostream>
#include <random>

template <int N, int M, typename T>
class Tensor {
private:
  // These initialization is supported by only C++11.
  int dim[M] = {};
  T v[N] = {};
public:
  Tensor(int const* dim_in);
  int size(int axis) const;
  int bytes() const;
  int* shape();
  void set_v(T* v_in);
  void init();
  void randomInit(float low, float high);
  T* get_v();
  Tensor<N, 2, T> flatten();
  template <int L>
  Tensor<N, L, T> reshape(int* shape);
  T &operator[](int i);
  Tensor<N, M, T> operator+(Tensor<N, M, T>& other) const;
  Tensor<N, M, T> operator-(Tensor<N, M, T>& other) const;
  Tensor<N, M, T> operator*(Tensor<N, M, T>& other) const;
  Tensor<N, M, T> times(const float& other) const;
  Tensor<N, M, T> transpose() const;
};

template <int N, int M, typename T>
Tensor<N, M, T>::Tensor(int const* dim_in) {
  for (int i = 0; i < M; ++i) {
    dim[i] = dim_in[i];
  }
}

template <int N, int M, typename T>
int Tensor<N, M, T>::size(int axis) const {
  return dim[axis];
}

template< int N, int M, typename T>
int Tensor<N, M, T>::bytes() const {
  return N * sizeof(T);
}

template <int N, int M, typename T>
int* Tensor<N, M, T>::shape() {
  return dim;
}

template <int N, int M, typename T>
void Tensor<N, M, T>::set_v(T* v_in) {
  for (int i = 0; i < N; ++i)
    v[i] = v_in[i];
}

template <int N, int M, typename T>
void Tensor<N, M, T>::randomInit(float low, float high) {
  // std::random is supported by C++11.
  // I'm not sure whether it's supported Vivado HLS or not.
  std::random_device seed_gen;
  std::mt19937 engine(seed_gen());
  std::uniform_real_distribution<> dist(low, high);
  for (int i = 0; i < N; ++i) {
    v[i] = dist(engine);
  }
}

template <int N, int M, typename T>
void Tensor<N, M, T>::init() {
  for (int i = 0; i < N; ++i)
    v[i] = 0;
}

template <int N, int M, typename T>
T* Tensor<N, M, T>::get_v() {
  return v;
}

template <int N, int M, typename T>
T &Tensor<N, M, T>::operator[](int i) {
  if (i < 0 || i >= N) {
    std::cerr << i << " is out of range(" << N << ")." << std::endl;
    std::cerr << "Error!: Invalid index." << std::endl;
    abort(); // abort() is not supported by Vivado HLS.
  }
  return v[i];
}

template <int N, int M, typename T>
Tensor<N, M, T> Tensor<N, M, T>::operator+(Tensor<N, M, T> &y) const {
  Tensor<N, M, T> ans(dim);
  for (int i = 0; i < N; ++i) {
    ans[i] = v[i] + y[i];
  }
  return ans;
}

template <int N, int M, typename T>
Tensor<N, M, T> Tensor<N, M, T>::operator-(Tensor<N, M, T> &y) const {
  Tensor<N, M, T> ans(dim);
  for (int i = 0; i < N; ++i) {
    ans[i] = v[i] - y[i];
  }
  return ans;
}

template <int N, int M, typename T>
Tensor<N, M, T> Tensor<N, M, T>::operator*(Tensor<N, M, T> &y) const {
  Tensor<N, M, T> ans(dim);
  for (int i = 0; i < N; ++i) {
    ans[i] = v[i] * y[i];
  }
  return ans;
}

template <int N, int M, typename T>
Tensor<N, M, T> Tensor<N, M, T>::times(const float& y) const {
  Tensor<N, M, T> ans(dim);
  for (int i = 0; i < N; ++i) {
    ans[i] = v[i] * y;
  }
  return ans;
}

template <int N, int M, typename T>
Tensor<N, 2, T> Tensor<N, M, T>::flatten() {
  int ans_dim[] = {N, 1};
  Tensor<N, 2, T> ans(ans_dim);
  ans.set_v(v);
  return ans;
}

template <int N, int M, typename T>
template <int L>
Tensor<N, L, T> Tensor<N, M, T>::reshape(int* shape) {
  Tensor<N, L, T> ans(shape);
  ans.set_v(v);
  return ans;
}

template <int N, int M, typename T>
Tensor<N, M, T> Tensor<N, M, T>::transpose() const {
  // TODO(wakanapo): Be able to deal with the multi-dimensional array.
  Tensor<N, M, T> m(dim);
  std::swap(m.shape()[0], m.shape()[1]);
  for (int i = 0; i < dim[0]; ++i) {
    for (int j = 0 ; j < dim[1]; ++j) {
      m[i * dim[1] + j] = v[j * dim[0] + i];
    }
  }
  return m;
}
