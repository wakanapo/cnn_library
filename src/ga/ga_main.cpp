#include "ga/genom.hpp"

int main() {
  GeneticAlgorithm ga(8, 10, 5, 0.001, 0.001, 10);
  ga.run();
}
