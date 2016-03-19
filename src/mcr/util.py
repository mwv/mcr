"""util: miscellaneous helper functions
"""
from __future__ import division, print_function

from time import time
import sys
from contextlib import contextmanager
import string
from itertools import product, tee
from math import ceil, log

import numpy as np
import scipy.io.wavfile
import toml
import sklearn.metrics


def make_f1_score(average):
    """Return sklearn-style scorer object which measures f1-score with the
    specified averaging method.

    Returns
    -------
    Scorer object

    """
    func = partial(sklearn.metrics.f1_score, average=average)
    func.__name__ = 'f1_score'
    func.__doc__ = sklearn.metrics.f1_score.__doc__
    return sklearn.metrics.make_scorer(func)


def resample(X_train, y_train, minority_factor=2):
    """Resample the majority class.

    Reduce the size of the majority class to `minority_factor` * the size of the
    second largest class.

    """
    y_counts = np.bincount(y_train)
    minority_class = np.argsort(-y_counts)[1]  # second biggest class
    minority_size = y_counts[minority_class]
    ixs = []
    for label in np.unique(y_train):
        ixs_for_label = np.nonzero(y_train == label)[0]
        ixs.extend(
            list(np.random.choice(
                ixs_for_label,
                min(len(ixs_for_label), minority_size * minority_factor),
                replace=False
            ))
        )
    return X_train[ixs, :], y_train[ixs]


def wavread(filename):
    """Read wave file

    Returns
    -------
    sig : ndarray
        signal
    fs : int
        samplerate
    """
    fs, sig = scipy.io.wavfile.read(filename)
    return sig, fs


@contextmanager
def verb_print(msg, verbose=False):
    """Helper for verbose printing with timing around pieces of code.
    """
    if verbose:
        t0 = time()
        msg = msg + '...'
        print(msg, end='')
        sys.stdout.flush()
    try:
        yield
    finally:
        if verbose:
            print('done. time: {0:.3f}s'.format(time() - t0))
            sys.stdout.flush()


def load_config(filename):
    """Load configuration from file.
    """
    with open(filename) as fid:
        config = toml.loads(fid.read())
    return config


def pretty_cm(cm, labels, hide_zeros=False, offset=''):
    """Pretty print for confusion matrices.
    """
    valuewidth = int(np.log10(np.clip(cm, 1, np.inf)).max()) + 1
    columnwidth = max(map(len, labels)+[valuewidth]) + 1
    empty_cell = " " * columnwidth
    s = ''
    # header
    s += offset + empty_cell
    for label in labels:
        s += "{1:>{0}s}".format(columnwidth, label)
    s += '\n\n'
    # rows
    for i, label1 in enumerate(labels):
        s += offset + '{1:{0}s}'.format(columnwidth, label1)
        for j in range(len(labels)):
            cell = '{1:{0}d}'.format(columnwidth, cm[i, j])
            if hide_zeros:
                cell = cell if cm[i, j] != 0 else empty_cell
            s += cell
        s += '\n'
    return s


def string_to_bool(s):
    """yeah.
    """
    if s == 'True':
        return True
    elif s == 'False':
        return False
    raise ValueError('not parsable')


def roll_array(arr, stacksize):
    arr = np.vstack((
        np.zeros((stacksize//2, arr.shape[1])),
        arr,
        np.zeros((stacksize//2, arr.shape[1]))
    ))
    return np.hstack(
        np.roll(arr, -i, 0)
        for i in range(stacksize)
    )[:arr.shape[0] - stacksize + 1]


def encode_symbol_range(high,
                        symbols=string.ascii_lowercase,
                        join=lambda s: ''.join(s)):
    return dict(
        enumerate(
            map(
                join,
                product(
                    *tee(symbols,
                         int(ceil(log(high, len(symbols)))))
                )
            )
        )
    )
