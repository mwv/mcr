#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: util.py
# date: Fri June 05 14:58 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""util:

"""

from __future__ import division

import os.path as path
from time import time
import sys
from contextlib import contextmanager

import toml

@contextmanager
def verb_print(msg, verbose=False):
    """Helper for verbose printing with timing around pieces of code.
    """
    if verbose:
        t0 = time()
        msg = msg + '...'
        print msg,
        sys.stdout.flush()
    try:
        yield
    finally:
        if verbose:
            print 'done. time: {0:.3f}s'.format(time() - t0)
            sys.stdout.flush()

def load_config(filename):
    if not path.exists(filename):
        print 'no such file: {0}'.format(filename)
        exit()
    with open(filename) as fid:
        config = toml.loads(fid.read())
    return config
