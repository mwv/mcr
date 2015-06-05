#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: make_dataset.py
# date: Fri April 10 12:04 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""make_dataset:

"""

from __future__ import division
import numpy as np
import operator
from collections import Counter

import pandas as pd
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

def sample_combinations(n, minlength=1, maxlength=None, seed=None):
    """
    Sample combinations of the integers from 1 to n.
    """
    np.random.seed(seed)
    if maxlength is None:
        maxlength = n - 1
    a = np.arange(0, n, dtype=np.int)
    while True:
        yield np.random.choice(a,
                               size=np.random.random_integers(minlength, maxlength),
                               replace=False)


def train_test_split_files(df, test_size=0.25, max_iter=2e6, tolerance=0.05,
                           seed=13, verbose=False):
    """
    Split the files in file_counts into train and test sets, preserving the distribution
    of call classes.

    Parameters
    ----------
    df : DataFrame
    test_size : float, optional
        ratio of test set
    max_iter : int, optional
        maximum number of iterations in sampling
    tolerance : float, optional
        proportion of allowed errors
    seed : int, optional
        seed for the prng

    Returns
    -------
    train, test : list of string
        list of filenames in training and test sets respectively

    """
    df_ = df[df['call'] != 'SIL']
    file_counts = {}
    for filename in df_['filename'].unique():
        calls = Counter(df_[(df_['filename'] == filename) &
                            (df_['call'] != 'SIL')]['call'])
        file_counts[filename] = calls

    filenames = file_counts.keys()
    counts = zip(*sorted(file_counts.items(), key=lambda x: x[0]))[1]
    total = reduce(operator.add, counts)
    target = {k: int(total[k] * test_size) for k in total}

    mincost = np.inf
    bestsol = None
    target_distr = np.fromiter((target[k] for k in sorted(target.keys())), dtype=np.double)
    cutoff = target_distr.sum() * 0.05

    n = len(counts)
    minlength = max(1, len(counts) * test_size - 10)
    maxlength = min(len(counts), len(counts) * test_size + 10)
    if verbose:
        import time
        t0 = time.time()
    for ix, indices in enumerate(sample_combinations(n, minlength, maxlength, seed)):
        if ix > max_iter:
            break
        if verbose and ix % 1e5 == 0:
            print ix, mincost, len(indices), '{0:.3f}s'.format(time.time() - t0)
            t0 = time.time()
        counter = reduce(operator.add, [counts[i] for i in indices])
        found_distr = np.fromiter((counter[k] for k in sorted(target.keys())), dtype=np.double)
        cost = np.abs(found_distr - target_distr).sum()
        if cost < mincost:
            mincost = cost
            bestsol = indices
            if verbose:
                print ix, mincost, len(indices), sorted(counter.items())
        if mincost < cutoff:
            break
    if verbose:
        print ix, mincost, len(indices), sorted(counter.items())
    train = [f for ix, f in enumerate(filenames) if not ix in bestsol]
    test = [filenames[i] for i in bestsol]
    return train, test


def print_dataset(X, output):
    with open(output, 'w') as fid:
        fid.write('filename,start,end,call\n')
        for filename, start, end, call in sorted(X, key=lambda x: (x[0], x[1])):
            fid.write('{0},{1:.3f},{2:.3f},{3}\n'.format(
                filename, start, end, call))


if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(
            prog='make_dataset.py',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='Make datasets for monkeys')
        parser.add_argument('master', metavar='MASTERFILE',
                            nargs=1,
                            help='master annotation file')
        parser.add_argument('mode', # metavar='MODE',
                            nargs=1,
                            choices=['isolated', 'continuous'],
                            help='type of dataset to make')
        parser.add_argument('monkey', # metavar='MONKEY',
                            nargs=1,
                            choices=['blue', 'colobus', 'titi', 'all'],
                            help='monkey')
        parser.add_argument('output', metavar='OUTPUT',
                            nargs=1,
                            help='output destination')
        parser.add_argument('--size',
                            action='store',
                            dest='size',
                            default=False,
                            help='construct a data set of this size')
        parser.add_argument('--split',
                            action='store',
                            dest='split',
                            default=False,
                            help='split into training and test sets according to ratio')
        return vars(parser.parse_args())
    args = parse_args()

    master = args['master'][0]
    monkey = args['monkey'][0]
    mode = args['mode'][0]
    output = args['output'][0]
    split = float(args['split'])
    size = int(args['size'])
    print 'generating dataset'
    print 'master: {0}'.format(master)
    print 'monkey: {0}'.format(monkey)
    print 'mode:   {0}'.format(mode)
    print 'split:  {0:.3f}'.format(split)
    print 'size:   {0:d}'.format(size)
    print 'output: {0}'.format(output)

    df = pd.read_csv(master)

    # select monkey
    if monkey != 'all':
        df = df[df['monkey'] == monkey]

    # select calls based on mode
    if mode == 'isolated':
        # in isolated mode, we ignore the silences and select isolated
        # calls for the data sets without regard for their distribution over the files
        df = df[df['call'] != 'SIL']

        X = df[['filename', 'start', 'end', 'call']].values
        y = LabelEncoder().fit_transform(df['call'].values)

        if size:
            train_ix, _ = iter(StratifiedShuffleSplit(y, 1, train_size=size,
                                                      random_state=42)).next()
            X = X[train_ix]
            y = y[train_ix]

        if split:
            train_ix, test_ix = iter(StratifiedShuffleSplit(y,
                                                            1,
                                                            test_size=split,
                                                            random_state=42)).next()
            X_train = X[train_ix]
            X_test = X[test_ix]
            print_dataset(X_train, output + '.train')
            print_dataset(X_test, output + '.test')
        else:
            print_dataset(X, output)
    else:
        if size:
            # we're ignoring size restrictions in this mode
            pass
        if split:
            train_files, test_files = train_test_split_files(df, test_size=split)
            X_train = df[df['filename'].isin(train_files)]\
                [['filename', 'start', 'end', 'call']].values
            X_test = df[df['filename'].isin(test_files)]\
                [['filename', 'start', 'end', 'call']].values
            print_dataset(X_train, output + '.train')
            print_dataset(X_test, output + '.test')
        else:
            print_dataset(X, output)
