#include "util/converter.hpp"

// static
template<>
float Converter::ToFloat(const float& other) {
  return other;
}

template<>
float Converter::ToFloat(const half& other) {
  return (float) other;
}

template<>
double Converter::ToDouble(const double& other) {
  return other;
}

template<>
double Converter::ToDouble(const float& other) {
  return (double)other;
}

using half_float::half;
template<>
double Converter::ToDouble(const half& other) {
  return (double)other;
}
