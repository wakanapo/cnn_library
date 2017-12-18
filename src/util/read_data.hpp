#pragma once

#include <stdio.h>
#include <stdlib.h>


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

int reverseInt(int i) {
  unsigned char c1, c2, c3, c4;
  c1 = i & 255;
  c2 = (i >> 8) & 255;
  c3 = (i >> 16) & 255;
  c4 = (i >> 24) & 255;
  return ((int)c1 << 24) + ((int) c2 << 16) + ((int)c3 << 8) + c4;
}

template<typename T>
data readMnistImages(status st) {
  FILE *fp = (st == TEST) ? fopen("data/t10k-images-idx3-ubyte", "rb") :
    fopen("data/train-images.idx3-ubyte", "rb");
  if (fp == NULL) {
    printf("file open error!!\n");
    exit(EXIT_FAILURE);
  }

  int magic_number = 0;
  int number_of_images = 0;
  int n_rows = 0;
  int n_cols = 0;

  size_t err = fread(&magic_number, sizeof(int), 1, fp);
  magic_number = reverseInt(magic_number);
  if (magic_number != 2051) {
    printf("Invalid MNIST image file!\n");
    exit(EXIT_FAILURE);
  }

  err = fread(&number_of_images, sizeof(int), 1, fp);
  number_of_images = reverseInt(number_of_images);
  err = fread(&n_rows, sizeof(int), 1, fp);
  n_rows = reverseInt(n_rows);
  err = fread(&n_cols, sizeof(int), 1, fp);
  n_cols = reverseInt(n_cols);
  int image_size = n_rows * n_cols;

  T* datasets =
    (T*)malloc(sizeof(T) * number_of_images * image_size);
  for (int n = 0; n < number_of_images; ++n) {
    for (int i = 0; i < n_rows; ++i) {
      for (int j = 0; j < n_cols; ++j) {
        unsigned char temp = 0;
        err = fread(&temp, sizeof(temp), 1, fp);
        if (err < 1)
          printf("File read error!\n");
        datasets[n * image_size + i * n_cols + j] = (T)temp / 255.0;
      }
    }
  }
  fclose(fp);
  data d(number_of_images, image_size, datasets);
  return d;
}

data readMnistLabels(status st) {
  FILE *fp = (st == TEST) ? fopen("data/t10k-labels-idx1-ubyte", "rb") :
    fopen("data/train-labels.idx1-ubyte", "rb");
  if (fp == NULL) {
    printf("file open error!!\n");
    exit(EXIT_FAILURE);
  }

  int magic_number = 0;
  int number_of_labels = 0;
  size_t err = fread(&magic_number, sizeof(int), 1, fp);
  magic_number = reverseInt(magic_number);
  if (magic_number != 2049) {
    printf("Invalid MNIST label file!\n");
    exit(EXIT_FAILURE);
  }
  err = fread(&number_of_labels, sizeof(int), 1, fp);
  number_of_labels = reverseInt(number_of_labels);

  unsigned long* datasets =
    (unsigned long*)malloc(sizeof(unsigned long) * number_of_labels);
  for (int i = 0; i < number_of_labels; ++i) {
    unsigned char temp = 0;
    err = fread(&temp, sizeof(temp), 1, fp);
    if (err < 1)
      printf("File read error!\n");
    datasets[i] = (unsigned long)temp;
  }
  fclose(fp);
  data l(1, number_of_labels, datasets);
  return l;
}

template<typename T>
T* mnistOneHot(unsigned long t) {
  T* onehot = (T*)malloc(10*sizeof(T));
  for (int i = 0; i < 10; ++i) {
    onehot[i] = (i == (int)t) ? 1 : 0;
  }
  return onehot;
}
