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

#ifndef _RNGLIB_
#define _RNGLIB_

extern void advance_state ( int k );
extern int antithetic_get (void );
extern void antithetic_memory ( int i, int *value );
extern void antithetic_set (  int value );
extern void cg_get ( int g, int *cg1, int *cg2 );
extern void cg_memory ( int i, int g, int *cg1, int *cg2 );
extern void cg_set ( int g, int cg1, int cg2 );
extern int cgn_get (void);
extern void cgn_memory ( int i, int *g );
extern void cgn_set ( int g );
extern void get_state ( int *cg1, int *cg2 );
extern int i4_uni (void);
extern void ig_get ( int g, int *ig1, int *ig2 );
extern void ig_memory ( int i, int g, int *ig1, int *ig2 );
extern void ig_set ( int g, int ig1, int ig2 );
extern void init_generator ( int t );
extern void initialize (void);
extern int initialized_get (void);
extern void initialized_memory ( int i, int *initialized );
extern void initialized_set (void);
extern void lg_get ( int g, int *lg1, int *lg2 );
extern void lg_memory ( int i, int g, int *lg1, int *lg2 );
extern void lg_set ( int g, int lg1, int lg2 );
extern int multmod ( int a, int s, int m );
extern float r4_uni_01 (void);
extern double r8_uni_01 (void);
extern void set_initial_seed ( int ig1, int ig2 );
extern void set_seed ( int cg1, int cg2 );
extern void timestamp (void);

#endif /* _RNGLIB_ */
