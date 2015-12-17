/* * This file is part of MSMTools.
 *
 * Copyright (c) 2015, 2014 Computational Molecular Biology Group
 *
 * MSMTools is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/* moduleauthor:: F. Paul <fabian DOT paul AT fu-berlin DOT de> */
#include <stdlib.h>
#include <math.h>
//#include <string.h>
#undef NDEBUG
#include <assert.h>
#include "../../util/sigint_handler.h"
#include "_mle_trev.h"

#ifdef _MSC_VER
#undef isnan
int isnan(double var)
{
    volatile double d = var;
    return d != d;
}
#endif

static double relative_error(const int n, const double *const a, const double *const b)
{
  double sum;
  double d;
  double max = 0.0;
  int i;
  for(i=0; i<n; i++) {
    sum = a[i]+b[i];
    if(sum>0) {
      d = fabs((a[i]-b[i])/sum);
      if(d>max) max=d;
    }
  }
  return max;
}

int _mle_trev_sparse(double * const T_data, const double * const CCt_data,
					const int * const i_indices, const int * const j_indices,
					const int len_CCt, const double * const sum_C,
					const int dim, const double maxerr, const int maxiter,
//					double * const mu,
					double eps_mu)
{
  double rel_err;
  int i, j, t, err, iteration;
  double *x, *x_new, *sum_x, *sum_x_old, *temp;
  double CCt_ij;
  double x_norm;

  sigint_on();

  err = 0;

  x = (double*)malloc(len_CCt*sizeof(double));
  x_new= (double*)malloc(len_CCt*sizeof(double));
  sum_x= (double*)malloc(dim*sizeof(double));
  sum_x_old= (double*)malloc(dim*sizeof(double));
  if(!(x && x_new && sum_x && sum_x_old)) { err=1; goto error; }

  /* ckeck sum_C */
  for(i = 0; i<dim; i++) if(sum_C[i]==0) { err=3; goto error; }

  /* initialize x */
  x_norm = 0;
  for(t = 0; t<len_CCt; t++) x_norm += CCt_data[t];
  for(t = 0; t<len_CCt; t++) x_new[t]= CCt_data[t]/x_norm;

  /* initialize sum_x */
  for(i = 0; i<dim; i++) sum_x[i] = 0;

  /* iterate */
  iteration = 0;
  do {
    /* swap buffers */
    temp = x;
    x = x_new;
    x_new = temp;

    temp = sum_x;
    sum_x = sum_x_old;
    sum_x_old = temp;

    /* update x_sum */
    for(i = 0; i<dim; i++) sum_x[i] = 0;
    for(t = 0; t<len_CCt; t++) { 
      j = j_indices[t];
      sum_x[j] += x[t];
    }
    for(i = 0; i<dim; i++) if(sum_x[i]==0 || isnan(sum_x[i])) { err=2; goto error; }

    /* update x */
    x_norm = 0;
    for(t=0; t<len_CCt; t++) {
      i = i_indices[t];
      j = j_indices[t];
      CCt_ij = CCt_data[t];
      x_new[t] = CCt_ij / (sum_C[i]/sum_x[i] + sum_C[j]/sum_x[j]);
      x_norm += x_new[t];
    }

    /* normalize x */
    for(t=0; t<len_CCt; t++) {
      x_new[t] /= x_norm;
      if (x_new[t] <= eps_mu) { err = 6; goto error; }
    }

    iteration += 1;
    rel_err = relative_error(dim, sum_x, sum_x_old);
  } while(rel_err > maxerr && iteration < maxiter && !interrupted);

  /* calculate T */
  for(t=0; t<len_CCt; t++) {
      i = i_indices[t];
      j = j_indices[t];
      T_data[t] = x[t] / sum_x[i];
  }

  if(iteration==maxiter) { err=5; goto error; }

  //memcpy(mu, x_new, len_CCt*sizeof(double));
  free(x);
  free(x_new);
  free(sum_x_old);
  free(sum_x);
  sigint_off();
  return 0;

error:
  //memcpy(mu, x_new, len_CCt*sizeof(double));
  free(x);
  free(x_new);
  free(sum_x_old);
  free(sum_x);
  sigint_off();
  return -err;
}
