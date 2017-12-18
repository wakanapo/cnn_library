#pragma once

#include "protos/arithmatic.pb.h"
#include "util/float_type.hpp"

Arithmatic::One p;
int E = 8;
int M = 23;

float multiple(const float a, const float b, const char* file, int line) {
  // Arithmatic::Calculation* c = p.add_calc();
  // c->set_file(file);
  // c->set_line(line);
  // c->set_operator_("*");
  // c->set_a(a);
  // c->set_b(b);
  // c->set_ans(a*b);
  float a_ = BitConverter(E, M, a);
  float b_ = BitConverter(E, M, b);
  return a_ * b_;
}

float division(const float a, const float b, const char* file, int line) {
  // Arithmatic::Calculation* c = p.add_calc();
  // c->set_file(file);
  // c->set_line(line);
  // c->set_operator_("/");
  // c->set_a(a);
  // c->set_b(b);
  // c->set_ans(a/b);
  float a_ = BitConverter(E, M, a);
  float b_ = BitConverter(E, M, b);
  return a_ / b_;
}

float add(const float a, const float b, const char* file, int line) {
  // Arithmatic::Calculation* c = p.add_calc();
  // c->set_file(file);
  // c->set_line(line);
  // c->set_operator_("+");
  // c->set_a(a);
  // c->set_b(b);
  // c->set_ans(a+b);
  float a_ = BitConverter(E, M, a);
  float b_ = BitConverter(E, M, b);
  return a_ + b_ ;
}

float sub(const float a, const float b, const char* file, int line) {
  // Arithmatic::Calculation* c = p.add_calc();
  // c->set_file(file);
  // c->set_line(line);
  // c->set_operator_("-");
  // c->set_a(a);
  // c->set_b(b);
  // c->set_ans(a-b);
  float a_ = BitConverter(E, M, a);
  float b_ = BitConverter(E, M, b);
  return a_ - b_;
}

#define MUL(a, b) multiple(a, b, __FILE__, __LINE__)
#define DIV(a, b) division(a, b, __FILE__, __LINE__)
#define ADD(a, b) add(a, b, __FILE__, __LINE__)
#define SUB(a, b) sub(a, b, __FILE__, __LINE__)
