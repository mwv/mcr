#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: cfg_isolated.py
# date: Fri June 19 15:07 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""cfg_isolated:

"""

from __future__ import division

spec_cfg = dict(
    stacksize=[50,75,100],
    n_noise_fr=[0, 50],
    medfilt_t=[0, 1, 3, 5, 7, 9, 11],
    medfilt_s=[0, 1, 3, 5, 7, 9, 11],
    deltas=[True, False],
    dct=[True, False]
)

clf_cfg = dict(
    C=[0.00001, 10],
    gamma=[0.00001, 1.0],
    alpha=[0.00001, 1.0],
    # kernel=['rbf'], # or 'linear'
    # use when doing approximate stuff:
    # n_components=[10, 500]
)
