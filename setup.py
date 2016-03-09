#!/usr/bin/env python

# This file is part of MSMTools.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# MSMTools is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" MSMTools

MSMTools contains an API to estimate and analyze Markov state models.
"""
DOCLINES = __doc__.split("\n")

import sys
import os
import versioneer
import warnings

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Environment :: Console
Environment :: MacOS X
Intended Audience :: Science/Research
License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)
Natural Language :: English
Operating System :: MacOS :: MacOS X
Operating System :: POSIX
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3
Topic :: Scientific/Engineering :: Bio-Informatics
Topic :: Scientific/Engineering :: Chemistry
Topic :: Scientific/Engineering :: Mathematics
Topic :: Scientific/Engineering :: Physics

"""
from setup_util import getSetuptoolsError, lazy_cythonize
try:
    from setuptools import setup, Extension, find_packages
    from pkg_resources import VersionConflict
except ImportError as ie:
    print(getSetuptoolsError())
    sys.exit(23)

###############################################################################
# Extensions
###############################################################################
def extensions():
    """How do we handle cython:
    1. when on git, require cython during setup time (do not distribute
    generated .c files via git)
     a) cython present -> fine
     b) no cython present -> install it on the fly. Extensions have to have .pyx suffix
    This is solved via a lazy evaluation of the extension list. This is needed,
    because build_ext is being called before cython will be available.
    https://bitbucket.org/pypa/setuptools/issue/288/cannot-specify-cython-under-setup_requires

    2. src dist install (have pre-converted c files and pyx files)
     a) cython present -> fine
     b) no cython -> use .c files
    """
    USE_CYTHON = False
    try:
        from Cython.Build import cythonize
        USE_CYTHON = True
    except ImportError:
        warnings.warn('Cython not found. Using pre cythonized files.')

    # setup OpenMP support
    from setup_util import detect_openmp
    openmp_enabled, needs_gomp = detect_openmp()

    from numpy import get_include as _np_inc
    np_inc = _np_inc()

    exts = []

    mle_trev_given_pi_dense_module = \
        Extension('msmtools.estimation.dense.mle_trev_given_pi',
                  sources=['msmtools/estimation/dense/mle_trev_given_pi.pyx',
                           'msmtools/estimation/dense/_mle_trev_given_pi.c'],
                  depends=['msmtools/util/sigint_handler.h'],
                  include_dirs=['msmtools/estimation/dense', np_inc])

    mle_trev_given_pi_sparse_module = \
        Extension('msmtools.estimation.sparse.mle_trev_given_pi',
                  sources=['msmtools/estimation/sparse/mle_trev_given_pi.pyx',
                           'msmtools/estimation/sparse/_mle_trev_given_pi.c'],
                  depends=['msmtools/util/sigint_handler.h'],
                  include_dirs=['msmtools/estimation/dense', np_inc])

    mle_trev_dense_module = \
        Extension('msmtools.estimation.dense.mle_trev',
                  sources=['msmtools/estimation/dense/mle_trev.pyx',
                           'msmtools/estimation/dense/_mle_trev.c'],
                  depends=['msmtools/util/sigint_handler.h'],
                  include_dirs=[np_inc])

    mle_trev_sparse_module = \
        Extension('msmtools.estimation.sparse.mle_trev',
                  sources=['msmtools/estimation/sparse/mle_trev.pyx',
                           'msmtools/estimation/sparse/_mle_trev.c'],
                  depends=['msmtools/util/sigint_handler.h'],
                  include_dirs=[np_inc,
                                ])

    sampler_rev = \
        Extension('msmtools.estimation.dense.sampler_rev',
                  sources=['msmtools/estimation/dense/sampler_rev.pyx',
                           'msmtools/estimation/dense/sample_rev.c',
                           'msmtools/estimation/dense/_rnglib.c',
                           'msmtools/estimation/dense/_ranlib.c'],
                  include_dirs=[np_inc,
                                ])

    sampler_revpi = \
        Extension('msmtools.estimation.dense.sampler_revpi',
                  sources=['msmtools/estimation/dense/sampler_revpi.pyx',
                           'msmtools/estimation/dense/sample_revpi.c',
                           'msmtools/estimation/dense/_rnglib.c',
                           'msmtools/estimation/dense/_ranlib.c'],
                  include_dirs=[np_inc,
                                ])

    exts += [mle_trev_given_pi_dense_module,
             mle_trev_given_pi_sparse_module,
             mle_trev_dense_module,
             mle_trev_sparse_module,
             sampler_rev,
             sampler_revpi,
             ]

    if USE_CYTHON: # if we have cython available now, cythonize module
        exts = cythonize(exts)
    else:
        # replace pyx files by their pre generated c code.
        for e in exts:
            new_src = []
            for s in e.sources:
                new_src.append(s.replace('.pyx', '.c'))
            e.sources = new_src

    if openmp_enabled:
        warnings.warn('enabled openmp')
        omp_compiler_args = ['-fopenmp']
        omp_libraries = ['-lgomp'] if needs_gomp else []
        omp_defines = [('USE_OPENMP', None)]
        for e in exts:
            e.extra_compile_args += omp_compiler_args
            e.extra_link_args += omp_libraries
            e.define_macros += omp_defines

    return exts


def get_cmdclass():
    versioneer_cmds = versioneer.get_cmdclass()

    sdist_class = versioneer_cmds['sdist']
    class sdist(sdist_class):
        """ensure cython files are compiled to c, when distributing"""

        def run(self):
            # only run if .git is present
            if not os.path.exists('.git'):
                return

            try:
                from Cython.Build import cythonize
                print("cythonizing sources")
                cythonize(extensions())
            except ImportError:
                warnings.warn('sdist cythonize failed')
            return sdist_class.run(self)

    versioneer_cmds['sdist'] = sdist
    return versioneer_cmds


metadata = dict(
    name='msmtools',
    maintainer='Martin K. Scherer',
    maintainer_email='m.scherer@fu-berlin.de',
    author='Benjamin Trendelkamp-Schroer',
    author_email='benjamin.trendelkamp-schroer@fu-berlin.de',
    url='http://github.com/markovmodel/msmtools',
    license='LGPLv3+',
    description=DOCLINES[0],
    long_description=open('README.rst').read(),
    version=versioneer.get_version(),
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    classifiers=[c for c in CLASSIFIERS.split('\n') if c],
    keywords='Markov State Model Algorithms',
    # packages are found if their folder contains an __init__.py,
    packages=find_packages(),
    cmdclass=get_cmdclass(),
    # runtime dependencies
    install_requires=['numpy>=1.6.0',
                      'scipy>=0.11',
                      'six',
                      'decorator',
                      ],
    zip_safe=False,
)

# include testing data
metadata['package_data'] = {'msmtools.util.matrix': ['testfiles/*'],
                            'msmtools.analysis': ['tests/*'],
                            'msmtools.estimation': ['test/testfiles/*'],
                            'msmtools.estimation.sparse': ['testfiles/*'],
                            'msmtools.estimation.dense': ['testfiles/*'],
                            }


# this is only metadata and not used by setuptools
metadata['requires'] = ['numpy', 'scipy']

# not installing?
if len(sys.argv) == 1 or (len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
                          sys.argv[1] in ('--help-commands',
                                          '--version',
                                          'clean'))):
    pass
else:
    # setuptools>=2.2 can handle setup_requires
    metadata['setup_requires'] = ['numpy>=1.6.0',
                                  ]

    # when on git, we require cython
    if os.path.exists('.git'):
        warnings.warn('using git, require cython')
        metadata['setup_requires'] += ['cython>=0.22']

    # only require numpy and extensions in case of building/installing
    metadata['ext_modules'] = lazy_cythonize(extensions)

    # add argparse to runtime deps if python version is 2.6
    if sys.version_info[:2] == (2, 6):
        metadata['install_requires'] += ['argparse']

    # include ipython notebooks. Will be installed directly in site-packages
    #metadata['packages'] += ['pyemma-ipython']
    #metadata['include_package_data'] = True

setup(**metadata)
