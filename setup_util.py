
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

"""
utility functions for python setup
"""
import contextlib
import os
import shutil
import sys
import tempfile

import setuptools


@contextlib.contextmanager
def TemporaryDirectory():
    n = tempfile.mkdtemp()
    yield n
    shutil.rmtree(n)


@contextlib.contextmanager
def stdchannel_redirected(stdchannel, dest_filename, fake=False):
    """
    A context manager to temporarily redirect stdout or stderr

    e.g.:

    with stdchannel_redirected(sys.stderr, os.devnull):
        if compiler.has_function('clock_gettime', libraries=['rt']):
            libraries.append('rt')
    """
    if fake:
        yield
        return
    oldstdchannel = dest_file = None
    try:
        oldstdchannel = os.dup(stdchannel.fileno())
        dest_file = open(dest_filename, 'w')
        os.dup2(dest_file.fileno(), stdchannel.fileno())

        yield
    finally:
        if oldstdchannel is not None:
            os.dup2(oldstdchannel, stdchannel.fileno())
        if dest_file is not None:
            dest_file.close()

# From http://stackoverflow.com/questions/
# 7018879/disabling-output-when-compiling-with-distutils
def has_function(compiler, funcname, headers):
    if not isinstance(headers, (tuple, list)):
        headers = [headers]
    with TemporaryDirectory() as tmpdir, stdchannel_redirected(sys.stderr, os.devnull), \
             stdchannel_redirected(sys.stdout, os.devnull):
        try:
            fname = os.path.join(tmpdir, 'funcname.c')
            f = open(fname, 'w')
            for h in headers:
                f.write('#include <%s>\n' % h)
            f.write('int main(void) {\n')
            f.write(' %s();\n' % funcname)
            f.write('return 0;}')
            f.close()
            objects = compiler.compile([fname], output_dir=tmpdir)
            compiler.link_executable(objects, os.path.join(tmpdir, 'a.out'))
        except (setuptools.distutils.errors.CompileError, setuptools.distutils.errors.LinkError):
            return False
        except:
            import traceback
            traceback.print_last()
            return False
        return True

def detect_openmp(compiler):
    from distutils.log import debug
    from copy import deepcopy
    compiler = deepcopy(compiler) # avoid side-effects
    has_openmp = has_function(compiler, 'omp_get_num_threads', headers='omp.h')
    debug('[OpenMP] compiler %s has builtin support', compiler)
    additional_libs = []
    if not has_openmp:
        debug('[OpenMP] compiler %s needs library support', compiler)
        if sys.platform == 'darwin':
            compiler.add_library('iomp5')
        elif sys.platform == 'linux':
            compiler.add_library('gomp')
        has_openmp = has_function(compiler, 'omp_get_num_threads', headers='omp.h')
        if has_openmp:
            additional_libs = [compiler.libraries[-1]]
            debug('[OpenMP] added library %s', additional_libs)
    return has_openmp, additional_libs


def getSetuptoolsError():
    bootstrap_setuptools = """\
python2.7 -c "import urllib2;
url=\'https://bootstrap.pypa.io/ez_setup.py\';\n
exec urllib2.urlopen(url).read()\""""
    cmd = ((80 * '=') + '\n' + bootstrap_setuptools + '\n' + (80 * '='))
    s = 'You can use the following command to upgrade/install it:\n%s' % cmd
    return s


class lazy_cythonize(list):
    """evaluates extension list lazyly.
    pattern taken from http://tinyurl.com/qb8478q"""
    def __init__(self, callback):
        self._list, self.callback = None, callback
    def c_list(self):
        if self._list is None: self._list = self.callback()
        return self._list
    def __iter__(self):
        for e in self.c_list(): yield e
    def __getitem__(self, ii): return self.c_list()[ii]
    def __len__(self): return len(self.c_list())
