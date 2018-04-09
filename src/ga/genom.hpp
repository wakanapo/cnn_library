#pragma once

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "util/box_quant.hpp"
#include "cnn/cnn.hpp"
#include "ga/set_gene.hpp"
#include "ga/first_genoms.hpp"
#include "protos/genom.pb.h"

class Genom {
public:
  Genom(std::vector<float> genom_list, float evaluation):
    genom_list_(genom_list), evaluation_(evaluation) {};
  std::vector<float> getGenom() const { return genom_list_; };
  float getEvaluation() const { return evaluation_; };
  void setGenom(std::vector<float> genom_list) { genom_list_ = genom_list; };
  void evaluation();
private:
  std::vector<float> genom_list_;
  float evaluation_;
};

void Genom::evaluation() {
  const Data test_X = ReadMnistImages<float>(TEST);
  const Data test_y = ReadMnistLabels(TEST);
  CNN<float> cnn;
  cnn.simple_load("float_params.pb");
  int cnt = 0;

  std::vector<std::future<bool>> futures;
  for (int i = 0; i < 3000; ++i) {
    futures.push_back(std::async([&test_X, &test_y, &cnn, i] {
          Tensor2D<28, 28, float> x;
          x.set_v((float*)test_X.ptr_ + i * x.size(1) * x.size(0));
          unsigned long y = cnn.simple_predict(x);
          return y == ((unsigned long*)test_y.ptr_)[i];
        }));
  }
  for (auto& f : futures) {
    if (f.get())
      ++cnt;
  }
  evaluation_ = (float)cnt / (float)3000;
  free(test_X.ptr_);
  free(test_y.ptr_);
}


class GeneticAlgorithm {
public:
  GeneticAlgorithm(int genom_length, int genom_num, int elite_num,
                   float individual_mutation, float genom_mutation, int max_generation)
    : genom_length_(genom_length), genom_num_(genom_num), elite_num_(elite_num),
      individual_mutation_(individual_mutation), genom_mutation_(genom_mutation),
      max_generation_(max_generation) {
    for (auto genom: range) {
      genoms_.push_back(Genom(genom, 0));
    }
  };
  std::vector<Genom> selectElite() const;
  std::vector<Genom> crossover(std::vector<Genom> parents) const;
  void nextGenerationGeneCreate(std::vector<Genom>& progenies);
  void mutation();
  void run();
  void save();
  void print(int i);
private:
  int genom_length_;
  int genom_num_;
  int elite_num_;
  float individual_mutation_;
  float genom_mutation_;
  int max_generation_;
  std::vector<Genom> genoms_;
};

std::vector<Genom> GeneticAlgorithm::selectElite() const {
  std::vector<Genom> elites = genoms_;
  std::sort(elites.begin(), elites.end(),
            [](const Genom& a, const Genom& b) {
              return  a.getEvaluation() >  b.getEvaluation();
            });
  return std::vector<Genom>(elites.begin(), elites.begin() + elite_num_);
}

std::vector<Genom> GeneticAlgorithm::crossover(std::vector<Genom> parents) const {
  /*
    二点交叉を行う関数
   */
  std::vector<Genom> genoms;
  int cross_one = rand() % genom_length_;
  int cross_second = rand() % (genom_length_ - cross_one) + cross_one;

  std::vector<std::thread> threads;
  for (int i = 1; i < parents.size(); ++i) {
    std::vector<float> genom_one = parents[i-1].getGenom();
    std::vector<float> genom_two = parents[i].getGenom();
    threads.push_back(std::thread([&] {
          for (int j = cross_one; j < cross_second; ++j)
            std::swap(genom_one[j], genom_two[j]);
          std::sort(genom_one.begin(), genom_one.end());
          std::sort(genom_two.begin(), genom_two.end());
          genoms.push_back(Genom(genom_one, 0));
          genoms.push_back(Genom(genom_two, 0));
        }));
  }
  for (std::thread &th : threads) {
    th.join();
  }
  return genoms;
}

void GeneticAlgorithm::nextGenerationGeneCreate(std::vector<Genom>& progenies) {
  /*
    世代交代処理を行う関数
   */
  std::sort(genoms_.begin(), genoms_.end(),
            [](const Genom& a, const Genom& b) {
              return  a.getEvaluation() <  b.getEvaluation();
            });
  for (int i = 0; i < progenies.size(); ++i) {
    std::cout << genoms_[i].getEvaluation() << std::endl;
    genoms_[i] = progenies[i];
  }
}

void GeneticAlgorithm::mutation() {
  /*
    突然変異関数
   */
  for (auto& genom: genoms_) {
    std::random_device seed;
    std::mt19937 mt(seed());
    std::uniform_real_distribution<> rand(0.0, 1.0);
    if (individual_mutation_ > rand(mt))
      continue;
    std::vector<float> new_genom;
    float offset = 0.0;
    for (int i = 0; i < genom_length_; ++i) {
      int gene = genom.getGenom()[i];
      if (genom_mutation_ < rand(mt) / 100.0) {
        float random = (rand(mt) - 0.5) * 2;
        float diff;
        if (random > 0)
          diff = (i == genom.getGenom().size() -1) ? 0.1 : genom.getGenom()[i+1] - gene;
        else
          diff = (i == 0) ? 0.1 : gene - genom.getGenom()[i-1] - gene;
        offset = random * diff;
      }
      new_genom.push_back(gene + offset);
    }
    genom.setGenom(new_genom);
    std::cout << genom.getGenom().size() << std::endl;
  }
}

void GeneticAlgorithm::print(int i) {
  float min = 1.0;
  float max = 0;
  float sum = 0;
  
  for (auto& genom: genoms_) {
    float evaluation = genom.getEvaluation();
    sum += evaluation;
    if (evaluation < min)
      min = evaluation;
    if (evaluation > max)
      max = evaluation;
  }

  std::cout << "世代: " << i << std::endl;
  std::cout << "Min: " << min << std::endl;
  std::cout << "Max: " << max << std::endl;
  std::cout << "Ave: " << sum / genom_num_ << std::endl;
  std::cout << "-------------" << std::endl;
}

void GeneticAlgorithm::save() {
  std::string home = getenv("HOME");
  Gene::Genoms gs;
  for (auto genom : genoms_) {
    Gene::Gene* g = gs.add_genoms();
    for (auto gene : genom.getGenom()) {
      g->mutable_gene()->Add(gene);
    }
  }
  std::fstream output(home+"/utokyo-kudohlab/cnn_cpp/data/mutation0001.pb", std::ios::out | std::ios::trunc | std::ios::binary);
  if (!gs.SerializeToOstream(&output))
    std::cerr << "Failed to save genoms." << std::endl;
}

void GeneticAlgorithm::run() {
  for (int i = 0; i < max_generation_; ++i) {
    /* 各遺伝子の評価*/
    for (auto& genom: genoms_) {
      GlobalParams::setParams(genom.getGenom());
      genom.evaluation();
    }
    print(i);
    
    /* エリートの選出 */
    std::vector<Genom> elites = selectElite();
    /* エリート遺伝子を交叉させ、子孫を作る */
    std::vector<Genom> progenies = crossover(elites);
    /* 次世代集団の作成 */
    nextGenerationGeneCreate(progenies);
    /* 突然変異 */
    // mutation();
  }
  save();
}
