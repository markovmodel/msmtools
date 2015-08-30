r"""

.. currentmodule:: msmtools


MSM functions
=============
Low-level functions for estimation and analysis of transition matrices and io.

.. toctree::
   :maxdepth: 1

   dtraj
   generation
   estimation
   analysis
   flux


"""
from __future__ import absolute_import

from . import analysis
from . import estimation
from . import generation
from . import dtraj
from . import flux

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
