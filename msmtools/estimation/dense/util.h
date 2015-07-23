#ifndef __util_h_
#define __util_h_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>


int my_isinf(double x) {
#if _MSC_VER && !__INTEL_COMPILER
	return ! _finite(x);
#else
	return isinf(x);
#endif
}

int my_isnan(double x) {
#if _MSC_VER && !__INTEL_COMPILER
	return _isnan(x);
#else
	return isnan(x);
#endif
}

/**
    Helper function, tests if x is numerically positive

    :param x:
    :return:
*/
int is_positive(double x)
{
    double eps = 1e-8;
    if (x >= eps && !my_isinf(x) && !my_isnan(x))
		return 1;
    else
    	return 0;
}

double my_fmin(double a, double b) {
#if _MSC_VER && !__INTEL_COMPILER
	return __min(a,b);
#else
	return fmin(a,b);
#endif
}


#endif
