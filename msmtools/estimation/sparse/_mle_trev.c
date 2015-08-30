/* * This file is part of PyEMMA.
 *
 * Copyright 2015, Martin K. Scherer, Benjamin Trendelkamp-Schroer, Frank Noé,
 * Fabian Paul, Guillermo Pérez-Hernández
 *
 * PyEMMA is free software: you can redistribute it and/or modify
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

static double distsq(const int n, const double *const a, const double *const b)
{
  double d = 0.0;
  int i;
  for(i=0; i<n; i++) {
    d += (a[i]-b[i])*(a[i]-b[i]);
  }
  return d;
}

int _mle_trev_sparse(double * const T_data, const double * const CCt_data, const int * const i_indices, const int * const j_indices, const int len_CCt, const double * const sum_C, const int dim, const double maxerr, const int maxiter)
{
  double d_sq;
  int i, j, t, err, iteration;
  double *x, *x_new, *sum_x, *temp;
  double CCt_ij;
  double x_norm;

  sigint_on();

  err = 0;

  x = (double*)malloc(len_CCt*sizeof(double));
  x_new= (double*)malloc(len_CCt*sizeof(double));
  sum_x= (double*)malloc(dim*sizeof(double));
  if(!(x && x_new && sum_x)) { err=1; goto error; }
   
  /* ckeck sum_C */
  for(i = 0; i<dim; i++) if(sum_C[i]==0) { err=3; goto error; }
  
  /* initialize x */
  x_norm = 0;
  for(t = 0; t<len_CCt; t++) x_norm += CCt_data[t];
  for(t = 0; t<len_CCt; t++) x_new[t]= CCt_data[t]/x_norm;

  /* iterate */
  iteration = 0;
  do {
    /* swap buffers */
    temp = x;
    x = x_new;
    x_new = temp;
    
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
    }

    iteration += 1;
    d_sq = distsq(len_CCt,x,x_new);
  } while(d_sq > maxerr*maxerr && iteration < maxiter && !interrupted);
  
  /* calculate T */
  for(t=0; t<len_CCt; t++) {
      i = i_indices[t];
      j = j_indices[t];
      T_data[t] = x[t] / sum_x[i];
  }

  if(iteration==maxiter) { err=5; goto error; }

  free(x);
  free(x_new);
  free(sum_x);
  sigint_off();
  return 0;

error:
  free(x);
  free(x_new);
  free(sum_x);
  sigint_off();
  return -err;
}
