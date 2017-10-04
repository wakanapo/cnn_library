#pragma once

enum status {
  TRAIN,
  TEST
};


class data {
public:
  data(int c, int r, void* p) : col(c), row(r), ptr(p) {};
  const int col;
  const int row;
  void* ptr;
};
data readMnistImages(status st);
data readMnistLabels(status st);
float* mnistOneHot(unsigned long t);
