#ifndef KAHANDOT_H
#define KAHANDOT_H
#include <stdlib.h>
void _kdot(double *A, double *B, double *C, size_t n, size_t m, size_t l);
double _ksum(double *X, size_t n, size_t m);
void _exprel(double *X, double *Y, size_t n);
void _exprel2(double *X, double *Y, size_t n);
#endif
