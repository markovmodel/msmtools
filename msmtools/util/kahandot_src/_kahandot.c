#include "_kahandot.h"
#include <math.h>
#include <float.h>

#define A(i,j) (A[(i)*m+(j)])
#define B(i,j) (B[(i)*l+(j)])
#define C(i,j) (C[(i)*l+(j)])

#ifdef _MSC_VER
#define INFINITY (DBL_MAX+DBL_MAX)
#define portable_inline __inline
#else
#define portable_inline inline
#endif

void _kdot(double *A, double *B, double *C, size_t n, size_t m, size_t l)
{
    size_t i,j,k;
    double err, sum, t, y;
    
    for(i=0; i<n; ++i) {
        for(j=0; j<l; ++j) {
            err = 0.0;
            sum = 0.0;
            for(k=0; k<m; ++k) {
                y = A(i,k)*B(k,j) - err;
                t = sum + y;
                err = (t - sum) - y;
                sum = t;
            }
            C(i,j) = sum;
        }
    }
}

#undef A
#undef B
#undef C

#define X(i,j) (X[(i)*m+(j)])

double _ksum(double *X, size_t n, size_t m)
{
    size_t i,j;
    double err, sum, t, y;

    err = 0.0;
    sum = 0.0;
    for(i=0; i<n; ++i) {
        for(j=0; j<m; ++j) {
            y = X(i,j) - err;
            t = sum + y;
            err = (t - sum) - y;
            sum = t;
        }
    }
    return sum;
}

#undef X

/* specfunc/exp.c
 * 
 * Copyright (C) 1996, 1997, 1998, 1999, 2000 Gerard Jungman
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

/* gsl_machine.h: Author:  B. Gough and G. Jungman */
#define GSL_LOG_DBL_MIN   (-7.0839641853226408e+02)
#define GSL_LOG_DBL_MAX    7.0978271289338397e+02

static portable_inline double gsl_sf_exprel_e(const double x)
{
  const double cut = 0.002;

  if(x < GSL_LOG_DBL_MIN) {
    return -1.0/x;
  }
  else if(x < -cut) {
    return (exp(x) - 1.0)/x;
  }
  else if(x < cut) {
    return (1.0 + 0.5*x*(1.0 + x/3.0*(1.0 + 0.25*x*(1.0 + 0.2*x))));
  } 
  else if(x < GSL_LOG_DBL_MAX) {
    return (exp(x) - 1.0)/x;
  }
  else {
    return INFINITY;
  }
}


void _exprel(double *X, double *Y, size_t n)
{
  size_t i;

  for(i=0; i<n; ++i) {
    Y[i] = gsl_sf_exprel_e(X[i]);
  }
}

/*
http://www.wolfgang-ehrhardt.de/amath_functions.html

(*-------------------------------------------------------------------------
 (C) Copyright 2009-2015 Wolfgang Ehrhardt

 This software is provided 'as-is', without any express or implied warranty.
 In no event will the authors be held liable for any damages arising from
 the use of this software.

 Permission is granted to anyone to use this software for any purpose,
 including commercial applications, and to alter it and redistribute it
 freely, subject to the following restrictions:

 1. The origin of this software must not be misrepresented; you must not
    claim that you wrote the original software. If you use this software in
    a product, an acknowledgment in the product documentation would be
    appreciated but is not required.

 2. Altered source versions must be plainly marked as such, and must not be
    misrepresented as being the original software.

 3. This notice may not be removed or altered from any source distribution.
----------------------------------------------------------------------------*)

[9] N.J. Higham, Accuracy and Stability of Numerical Algorithms,
                      2nd ed., Philadelphia, 2002
                      http://www.maths.manchester.ac.uk/~higham/asna/
*/

static portable_inline long double amath_exprel(long double x)
{
  const long double xsmall = 1.0e-6;
  const long double ln2 = 0.69314718055994530941;
  long double z;

  z = fabs(x);
  if(z < ln2) {
    if(z < xsmall) {
      /*exprel = 1 + 1/2*x + 1/6*x^2 + 1/24*x^3 + O(x^4)*/
      return 1.0 + x*(0.5 + x/6.0);
    }
    else {
      /*See Higham [9], 1.14.1 Computing (e^x-1)/x, Algorithm 2*/
      z = exp(x);
      x = log(z);
      if(x==0.0) return 1.0;
      else return (z-1.0)/x;
    }
  }
  else {
    if(x<-45.0) return -1.0/x;
    else return (exp(x) - 1.0)/x;
  }
}

void _exprel2(double *X, double *Y, size_t n)
{
  size_t i;

  for(i=0; i<n; ++i) {
    Y[i] = (double)amath_exprel((long double)X[i]);
  }
}
