
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
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
