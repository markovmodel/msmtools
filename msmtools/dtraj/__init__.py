r"""

===============================================================
dtraj - Discrete trajectories functions (:mod:`msmtools.dtraj`)
===============================================================

.. currentmodule:: msmtools.dtraj

Discrete trajectory io
======================

.. autosummary::
   :toctree: generated/

   read_discrete_trajectory - read microstate trajectoryfrom ascii file
   read_dtraj
   write_discrete_trajectory - write microstate trajectory to ascii file
   write_dtraj
   load_discrete_trajectory - read microstate trajectoryfrom biqqnary file
   load_dtraj
   save_discrete_trajectory -  write microstate trajectory to binary file
   save_dtraj

Simple statistics
=================

.. autosummary::
   :toctree: generated/

   count_states
   visited_set
   number_of_states
   index_states

Sampling trajectory indexes
===========================

.. autosummary::
   :toctree: generated/

   sample_indexes_by_distribution
   sample_indexes_by_state
   sample_indexes_by_sequence

"""
from __future__ import absolute_import
from .api import *
