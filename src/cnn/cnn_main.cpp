#include <cstdio>

#include "util/read_data.hpp"
#include "cnn/cnn.hpp"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage error!\n");
    printf("====Usage====\n Train: ./cnn train\n Test : ./cnn test\n");
    return 1;
  }
  CNN cnn;
  if (argv[1][1] == 'r')
    cnn.run(TRAIN);
  else
    cnn.run(TEST);
  return 0;
}
  
