#ifndef LBFGSB_H
#define LBFGSB_H

#include "f2c.h"

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
            ftnlen csave_len);

void f_init();
void f_exit();

#endif
