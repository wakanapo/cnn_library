#include <cstdio>

#include "util/read_data.hpp"
#include "cnn/cnn.hpp"
// #include "util/types.hpp"
#include "half.hpp"

using half_float::half;
int main() {
  CNN<half>::run();
  // CNN<half>::inference();
  return 0;
}
  
