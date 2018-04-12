#include "ga/genom.hpp"

int main() {
  GeneticAlgorithm ga(16, 50, 10, 0.001, 0.001, 20);
  ga.run();
}
