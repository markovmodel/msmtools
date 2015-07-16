#ifndef _RANLIB_
#define _RANLIB_

extern char ch_cap ( char ch );
extern float genbet ( float aa, float bb );
extern float genchi ( float df );
extern float genexp ( float av );
extern float genf ( float dfn, float dfd );
extern float gengam ( float a, float r );
extern float *genmn ( float parm[] );
extern int *genmul ( int n, float p[], int ncat );
extern float gennch ( float df, float xnonc );
extern float gennf ( float dfn, float dfd, float xnonc );
extern float gennor ( float av, float sd );
extern void genprm ( int iarray[], int n );
extern float genunf ( float low, float high );
extern int i4_max ( int i1, int i2 );
extern int i4_min ( int i1, int i2 );
extern int ignbin ( int n, float pp );
extern int ignnbn ( int n, float p );
extern int ignpoi ( float mu );
extern int ignuin ( int low, int high );
extern int lennob ( char *s );
extern void phrtsd ( char *phrase, int *seed1, int *seed2 );
extern void prcomp ( int maxobs, int p, float mean[], float xcovar[], float answer[] );
extern float r4_exp ( float x );
extern float r4_exponential_sample ( float lambda );
extern float r4_max ( float x, float y );
extern float r4_min ( float x, float y );
extern float r4vec_covar ( int n, float x[], float y[] );
extern int s_eqi ( char *s1, char *s2 );
extern float sdot ( int n, float dx[], int incx, float dy[], int incy );
extern float *setcov ( int p, float var[], float corr );
extern void setgmn ( float meanv[], float covm[], int p, float parm[] );
extern float sexpo (void);
extern float sgamma ( float a );
extern float snorm (void);
extern int spofa ( float a[], int lda, int n );
extern void stats ( float x[], int n, float *av, float *var, float *xmin, float *xmax );
extern void trstat ( char *pdf, float parin[], float *av, float *var );

#endif /* _RANLIB_ */
