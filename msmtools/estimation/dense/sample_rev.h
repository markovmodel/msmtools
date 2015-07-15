/* * Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
 * Berlin, 14195 Berlin, Germany.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation and/or
 * other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* * moduleauthor:: F. Noe <frank DOT noe AT fu-berlin DOT de>
 */
#include <stdbool.h>

double _square(double x);
bool _is_positive(double x);
double _update_step(double v0, double v1, double v2, double c0, double c1, double c2, int random_walk_stepsize);
double _sum_all(double* X, int n);
void _normalize_all(double* X, int n);
void _normalize_all_sparse(double* X, int* I, int* J, int n, int n_idx);
double _sum_row(double* X, int n, int i);
void _update(double* C, double* sumC, double* X, int n, int n_step);
void _update_sparse_sparse(double* C, double* sumC, double* X, double* sumX, int* I, int* J, int n, int n_idx, int n_step);

void _print_matrix(double* X, int n);
void _update_sparse_speedtest(double* C, double* sumC, double* X, double* sumX, int* I, int* J, int n, int n_idx, int n_step);
