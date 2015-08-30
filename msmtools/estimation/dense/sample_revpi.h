/* * moduleauthor:: B. Trendelkamp-Schroer 
 * <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>  
 */
#ifndef _TMATRIX_SAMPLING_REVPI_
#define _TMATRIX_SAMPLING_REVPI_

extern double sample_quad(double xkl, double xkk, double xll,
			  double ckl, double clk, double ckk, double cll,
			  double bk, double bl);

extern double sample_quad_rw(double xkl, double xkk, double xll,
			  double ckl, double clk, double ckk, double cll,
			  double bk, double bl);

extern void update(double *X, double *C, double *b, size_t n);

extern void update_sparse(double *X, double *C, double *b, size_t n,
			  size_t * I, size_t * J, size_t n_idx);

#endif /* _TMATRIX_SAMPLING_REVPI_ */
