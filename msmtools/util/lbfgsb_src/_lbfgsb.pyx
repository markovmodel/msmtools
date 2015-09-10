import atexit

ctypedef int integer
ctypedef double doublereal
ctypedef int logical
ctypedef int ftnlen

cdef extern from "lbfgsb.h":
    int setulb_(integer *n,
                integer *m,
                doublereal *x,
                doublereal *l,
                doublereal *u,
                integer *nbd,
                doublereal *f,
                doublereal *g,
                doublereal *factr,
                doublereal *pgtol,
                doublereal *wa,
                integer *iwa,
                char *task,
                integer *iprint,
                char *csave,
                logical *lsave,
                integer *isave,
                doublereal *dsave,
                integer *maxls,
                ftnlen task_len,
                ftnlen csave_len)
    void f_exit()
    void f_init()

f_init()
atexit.register(f_exit)

def setulb(int m, double[:] x, double[:] low_bnd, double[:] upper_bnd, int[:] nbd,
           double f, double[:] g, double factr, double pgtol,
           double[:] wa, int[:] iwa, char[:] task, int iprint,
           char[:] csave, int[:] lsave, int[:] isave, double[:] dsave,
           int maxls):

    cdef int n = len(x)

    assert len(low_bnd) == n
    assert len(upper_bnd) == n
    assert len(nbd) == n
    assert len(g) == n

    setulb_(&n, &m,
            &x[0],
            &low_bnd[0], &upper_bnd[0], &nbd[0],
            &f, &g[0],
            &factr, &pgtol,
            &wa[0], &iwa[0],
            &task[0], &iprint,
            &csave[0], &lsave[0], &isave[0], &dsave[0],
            &maxls,
            60, 60)
