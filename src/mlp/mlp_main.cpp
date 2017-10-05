#include <cstdio>

#include "util/read_data.hpp"
#include "mlp/mlp.hpp"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage error!\n");
    printf("====Usage====\n Train: ./mlp train\n Test : ./mlp test\n");
    return 1;
  }
  MLP mlp;
  if (argv[1][1] == 'r')
    mlp.run(TRAIN);
  else
    mlp.run(TEST);
  return 0;
}
  
