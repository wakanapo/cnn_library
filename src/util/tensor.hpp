#pragma once

#include <iostream>
#include <typeinfo>
#include <random>

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
class Tensor {
private:
  static constexpr int size_ = dim1*dim2*dim3*dim4*dim5;
  static constexpr int shape_[] = {dim1, dim2, dim3, dim4, dim5};
  T v_[size_] = {};
public:
  int size(int axis) const;
  const int size() const;
  int bytes() const;
  const int* shape() const;
  template <typename T_prime>
  void set_v(T_prime* v_in);
  void init();
  void randomInit(T low, T high);
  T* get_v();
  Tensor<dim1*dim2*dim3*dim4*dim5, 1, 1, 1, 1, T> flatten();
  T &operator[](int i);
  Tensor<dim1, dim2, dim3, dim4, dim5, T>
  operator+(Tensor<dim1, dim2, dim3, dim4, dim5, T>& other) const;
  Tensor<dim1, dim2, dim3, dim4, dim5, T>
  operator-(Tensor<dim1, dim2, dim3, dim4, dim5, T>& other) const;
  Tensor<dim1, dim2, dim3, dim4, dim5, T>
  operator*(Tensor<dim1, dim2, dim3, dim4, dim5, T>& other) const;
  Tensor<dim1, dim2, dim3, dim4, dim5, T>
  times(const float& other) const;
  Tensor<dim2, dim1, dim3, dim4, dim5, T>
  transpose() const;
};

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
int Tensor<dim1, dim2, dim3, dim4, dim5, T>::size(int axis) const {
  return shape_[axis];
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
const int Tensor<dim1, dim2, dim3, dim4, dim5, T>::size() const {
  return size_;
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
int Tensor<dim1, dim2, dim3, dim4, dim5, T>::bytes() const {
  return size_ * sizeof(T);
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
const int* Tensor<dim1, dim2, dim3, dim4, dim5, T>::shape() const {
  return shape_;
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
template <typename T_prime>
void Tensor<dim1, dim2, dim3, dim4, dim5, T>::set_v(T_prime* v_in) {
  for (int i = 0; i < size_; ++i) {
    if (typeid(T) == typeid(T_prime)) {
      v_[i] = v_in[i];
    }
    else {
      v_[i] = (T)v_in[i];
    }
  }
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
void Tensor<dim1, dim2, dim3, dim4, dim5, T>::randomInit(T low, T high) {
  // std::random is supported by C++11.
  // I'm not sure whether it's supported Vivado HLS or not.
  std::random_device seed_gen;
  std::mt19937 engine(seed_gen());
  if (typeid(T) == typeid(int)) {
    std::uniform_int_distribution<> dist(low, high);
    for (int i = 0; i < size_; ++i) {
      v_[i] = dist(engine);
    }
  }
  else {
    std::uniform_real_distribution<> dist(low, high);
    for (int i = 0; i < size_; ++i) {
      v_[i] = (T)dist(engine);
    }
  }
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
void Tensor<dim1, dim2, dim3, dim4, dim5, T>::init() {
  for (int i = 0; i < size_; ++i)
    v_[i] = 0;
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
T* Tensor<dim1, dim2, dim3, dim4, dim5, T>::get_v() {
  return v_;
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
T &Tensor<dim1, dim2, dim3, dim4, dim5, T>::operator[](int i) {
  if (i < 0 || i >= size_) {
    std::cerr << i << " is out of range(" << size_ << ")." << std::endl;
    std::cerr << "Error!: Invalid index." << std::endl;
    abort(); // abort() is not supported by Vivado HLS.
  }
  return v_[i];
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
Tensor<dim1, dim2, dim3, dim4, dim5, T> Tensor<dim1, dim2, dim3, dim4, dim5, T>
::operator+(Tensor<dim1, dim2, dim3, dim4, dim5, T> &y) const {
  Tensor<dim1, dim2, dim3, dim4, dim5, T> ans;
  for (int i = 0; i < size_; ++i) {
    ans[i] = v_[i] + y[i];
  }
  return ans;
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
Tensor<dim1, dim2, dim3, dim4, dim5, T> Tensor<dim1, dim2, dim3, dim4, dim5, T>
::operator-(Tensor<dim1, dim2, dim3, dim4, dim5, T> &y) const {
  Tensor<dim1, dim2, dim3, dim4, dim5, T> ans;
  for (int i = 0; i < size_; ++i) {
    ans[i] = v_[i] - y[i];
  }
  return ans;
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
Tensor<dim1, dim2, dim3, dim4, dim5, T> Tensor<dim1, dim2, dim3, dim4, dim5, T>
::operator*(Tensor<dim1, dim2, dim3, dim4, dim5, T> &y) const {
  Tensor<dim1, dim2, dim3, dim4, dim5, T> ans;
  for (int i = 0; i < size_; ++i) {
    ans[i] = v_[i] * y[i];
  }
  return ans;
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
Tensor<dim1, dim2, dim3, dim4, dim5, T> Tensor<dim1, dim2, dim3, dim4, dim5, T>
::times(const float& y) const {
  Tensor<dim1, dim2, dim3, dim4, dim5, T> ans;
  for (int i = 0; i < size_; ++i) {
    ans[i] = v_[i] * y;
  }
  return ans;
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
Tensor<dim1*dim2*dim3*dim4*dim5, 1, 1, 1, 1, T>
Tensor<dim1, dim2, dim3, dim4, dim5, T>::flatten() {
  Tensor<size_, 1, 1, 1, 1, T> ans;
  ans.set_v(v_);
  return ans;
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
Tensor<dim2, dim1, dim3, dim4, dim5, T> Tensor<dim1, dim2, dim3, dim4, dim5, T>::transpose() const {
  // TODO(wakanapo): Be able to deal with the multi-dimensional array.
  Tensor<dim2, dim1, dim3, dim4, dim5, T> m;
  for (int i = 0; i < dim1; ++i) {
    for (int j = 0 ; j < dim2; ++j) {
      m[i * dim2 + j] = v_[j * dim1 + i];
    }
  }
  return m;
}

template<int dim1, typename T>
using Tensor1D = Tensor<dim1, 1, 1, 1, 1, T>;

template<int dim1, int dim2, typename T>
using Tensor2D = Tensor<dim1, dim2, 1, 1, 1, T>;

template<int dim1, int dim2, int dim3, typename T>
using Tensor3D = Tensor<dim1, dim2 ,dim3, 1, 1, T>;

template<int dim1, int dim2, int dim3, int dim4, typename T>
using Tensor4D = Tensor<dim1, dim2, dim3, dim4, 1, T>;

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
using Tensor5D = Tensor<dim1, dim2, dim3, dim4, dim5, T>;

