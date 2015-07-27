# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

r"""

=================
 dtraj API
=================

"""
from __future__ import absolute_import
from msmtools.util.annotators import shortcut
from msmtools.dtraj import discrete_trajectory

__docformat__ = "restructuredtext en"

__author__ = "Benjamin Trendelkamp-Schroer, Martin Scherer, Frank Noe"
__copyright__ = "Copyright 2014, Computational Molecular Biology Group, FU-Berlin"
__credits__ = ["Benjamin Trendelkamp-Schroer", "Martin Scherer", "Frank Noe"]
__license__ = "FreeBSD"
__version__ = "2.0.0"
__maintainer__ = "Martin Scherer"
__email__ = "m.scherer AT fu-berlin DOT de"

__all__ = ['read_discrete_trajectory',
           'write_discrete_trajectory',
           'load_discrete_trajectory',
           'save_discrete_trajectory']

################################################################################
# Discrete trajectory IO
################################################################################

################################################################################
# ascii
################################################################################

@shortcut('read_dtraj')
def read_discrete_trajectory(filename):
    r"""Read discrete trajectory from ascii file.

    Parameters
    ----------
    filename : str
        The filename of the discretized trajectory file.
        The filename can either contain the full or the
        relative path to the file.

    Returns
    -------
    dtraj : (M, ) ndarray of int
        Discrete state trajectory.

    See also
    --------
    write_discrete_trajectory

    Notes
    -----
    The discrete trajectory file contains a single column with
    integer entries.

    Examples
    --------

    >>> import numpy as np
    >>> from tempfile import NamedTemporaryFile
    >>> from msmtools.dtraj import write_discrete_trajectory, read_discrete_trajectory

    Use temporary file

    >>> tmpfile = NamedTemporaryFile()

    Discrete trajectory

    >>> dtraj = np.array([0, 1, 0, 0, 1, 1, 0])

    Write to disk (as ascii file)

    >>> write_discrete_trajectory(tmpfile.name, dtraj)

    Read from disk

    >>> X = read_discrete_trajectory(tmpfile.name)
    >>> X
    array([0, 1, 0, 0, 1, 1, 0])

    """
    return discrete_trajectory.read_discrete_trajectory(filename)


@shortcut('write_dtraj')
def write_discrete_trajectory(filename, dtraj):
    r"""Write discrete trajectory to ascii file.

    Parameters
    ----------
    filename : str
        The filename of the discrete state trajectory file.
        The filename can either contain the full or the
        relative path to the file.
    dtraj : array-like of int
        Discrete state trajectory

    See also
    --------
    read_discrete_trajectory

    Notes
    -----
    The discrete trajectory is written to a
    single column ascii file with integer entries.

    Examples
    --------

    >>> import numpy as np
    >>> from tempfile import NamedTemporaryFile
    >>> from msmtools.dtraj import write_discrete_trajectory, read_discrete_trajectory

    Use temporary file

    >>> tmpfile = NamedTemporaryFile()

    Discrete trajectory

    >>> dtraj = np.array([0, 1, 0, 0, 1, 1, 0])

    Write to disk (as ascii file)

    >>> write_discrete_trajectory(tmpfile.name, dtraj)

    Read from disk

    >>> X = read_discrete_trajectory(tmpfile.name)
    >>> X
    array([0, 1, 0, 0, 1, 1, 0])

    """
    discrete_trajectory.write_discrete_trajectory(filename, dtraj)


################################################################################
# binary
################################################################################

@shortcut('load_dtraj')
def load_discrete_trajectory(filename):
    r"""Read discrete trajectory form binary file.

    Parameters
    ----------
    filename : str
        The filename of the discrete state trajectory file.
        The filename can either contain the full or the
        relative path to the file.

    Returns
    -------
    dtraj : (M,) ndarray of int
        Discrete state trajectory

    See also
    --------
    save_discrete_trajectory

    Notes
    -----
    The binary file is a one dimensional numpy array
    of integers stored in numpy .npy format.

    Examples
    --------

    >>> import numpy as np
    >>> from tempfile import NamedTemporaryFile
    >>> from msmtools.dtraj import load_discrete_trajectory, save_discrete_trajectory

    Use temporary file
    
    >>> tmpfile = NamedTemporaryFile(suffix='.npy')

    Discrete trajectory

    >>> dtraj = np.array([0, 1, 0, 0, 1, 1, 0])

    Write to disk (as npy file)

    >>> save_discrete_trajectory(tmpfile.name, dtraj)

    Read from disk

    >>> X = load_discrete_trajectory(tmpfile.name)
    >>> X
    array([0, 1, 0, 0, 1, 1, 0])

    """
    return discrete_trajectory.load_discrete_trajectory(filename)


@shortcut('save_dtraj')
def save_discrete_trajectory(filename, dtraj):
    r"""Write discrete trajectory to binary file.

    Parameters
    ----------
    filename : str
        The filename of the discrete state trajectory file.
        The filename can either contain the full or the
        relative path to the file.
    dtraj : array-like of int
        Discrete state trajectory

    See also
    --------
    load_discrete_trajectory

    Notes
    -----
    The discrete trajectory is stored as ndarray of integers
    in numpy .npy format.

    Examples
    --------

    >>> import numpy as np
    >>> from tempfile import NamedTemporaryFile
    >>> from msmtools.dtraj import load_discrete_trajectory, save_discrete_trajectory

    Use temporary file

    >>> tmpfile = NamedTemporaryFile(suffix='.npy')

    Discrete trajectory

    >>> dtraj = np.array([0, 1, 0, 0, 1, 1, 0])

    Write to disk (as npy file)

    >>> save_discrete_trajectory(tmpfile.name, dtraj)

    Read from disk

    >>> X = load_discrete_trajectory(tmpfile.name)
    >>> X
    array([0, 1, 0, 0, 1, 1, 0])

    """
    discrete_trajectory.save_discrete_trajectory(filename, dtraj)
