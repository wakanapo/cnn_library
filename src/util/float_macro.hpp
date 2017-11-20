#pragma once

#include "protos/arithmatic.pb.h"

Arithmatic::One p;

float multiple(const float a, const float b, const char* file, int line) {
  // Arithmatic::Calculation* c = p.add_calc();
  // c->set_file(file);
  // c->set_line(line);
  // c->set_operator_("*");
  // c->set_a(a);
  // c->set_b(b);
  // c->set_ans(a*b);
  return a*b;
}

float division(const float a, const float b, const char* file, int line) {
  // Arithmatic::Calculation* c = p.add_calc();
  // c->set_file(file);
  // c->set_line(line);
  // c->set_operator_("/");
  // c->set_a(a);
  // c->set_b(b);
  // c->set_ans(a/b);
  return a/b;
}

float add(const float a, const float b, const char* file, int line) {
  // Arithmatic::Calculation* c = p.add_calc();
  // c->set_file(file);
  // c->set_line(line);
  // c->set_operator_("+");
  // c->set_a(a);
  // c->set_b(b);
  // c->set_ans(a+b);
  return a+b;
}

float sub(const float a, const float b, const char* file, int line) {
  // Arithmatic::Calculation* c = p.add_calc();
  // c->set_file(file);
  // c->set_line(line);
  // c->set_operator_("-");
  // c->set_a(a);
  // c->set_b(b);
  // c->set_ans(a-b);
  return a-b;
}

#define MUL(a, b) multiple(a, b, __FILE__, __LINE__)
#define DIV(a, b) division(a, b, __FILE__, __LINE__)
#define ADD(a, b) add(a, b, __FILE__, __LINE__)
#define SUB(a, b) sub(a, b, __FILE__, __LINE__)
