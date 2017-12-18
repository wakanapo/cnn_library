#pragma once

#include "protos/arithmatic.pb.h"
#include "util/float_type.hpp"

Arithmatic::One p;
int E = 8;
int M = 23;

template<typename T>
T multiple(const T a, const T b, const char* file, int line) {
  // Arithmatic::Calculation* c = p.add_calc();
  // c->set_file(file);
  // c->set_line(line);
  // c->set_operator_("*");
  // c->set_a(a);
  // c->set_b(b);
  // c->set_ans(a*b);
  // float a_ = BitConverter(E, M, a);
  // float b_ = BitConverter(E, M, b);
  return a * b;
}

template<typename T>
T division(const T a, const T b, const char* file, int line) {
  // Arithmatic::Calculation* c = p.add_calc();
  // c->set_file(file);
  // c->set_line(line);
  // c->set_operator_("/");
  // c->set_a(a);
  // c->set_b(b);
  // c->set_ans(a/b);
  // float a_ = BitConverter(E, M, a);
  // float b_ = BitConverter(E, M, b);
  return a / b;
}

template<typename T> 
T add(const T a, const T b, const char* file, int line) {
  // Arithmatic::Calculation* c = p.add_calc();
  // c->set_file(file);
  // c->set_line(line);
  // c->set_operator_("+");
  // c->set_a(a);
  // c->set_b(b);
  // c->set_ans(a+b);
  // float a_ = BitConverter(E, M, a);
  // float b_ = BitConverter(E, M, b);
  return a + b ;
}

template<typename T>
T sub(const T a, const T b, const char* file, int line) {
  // Arithmatic::Calculation* c = p.add_calc();
  // c->set_file(file);
  // c->set_line(line);
  // c->set_operator_("-");
  // c->set_a(a);
  // c->set_b(b);
  // c->set_ans(a-b);
  // float a_ = BitConverter(E, M, a);
  // float b_ = BitConverter(E, M, b);
  return a - b;
}

#define MUL(a, b) multiple(a, b, __FILE__, __LINE__)
#define DIV(a, b) division(a, b, __FILE__, __LINE__)
#define ADD(a, b) add(a, b, __FILE__, __LINE__)
#define SUB(a, b) sub(a, b, __FILE__, __LINE__)

