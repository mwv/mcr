#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: opt_isolated.py
# date: Fri June 19 10:10 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""opt_isolated: smac wrapper

"""

from __future__ import division

import numpy as np
import pandas as pd
import joblib

from sklearn.cross_validation import train_test_split

import classifier

APPROXIMATE = False
MONKEY = 'all'
VERBOSE = True
NJOBS = 4
NFOLDS = 3
CACHE_FILE = 'resources/cache.joblib.pkl'


def string_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    raise ValueError('not parsable')


def load_monkey(monkey):
    train_file = 'resources/{}.isolated.train'.format(monkey)
    # test_file = 'resources/{}.isolated.test'.format(monkey)

    df_train = pd.read_csv(train_file)
    X_train_text = df_train[['filename', 'start', 'end']].values
    calls_train = df_train['call'].values
    label2ix = {k: i for i, k in enumerate(np.unique(calls_train))}
    y_train = np.array([label2ix[call] for call in calls_train])

    # df_test = pd.read_csv(test_file)
    # X_test_text = df_test[['filename', 'start', 'end']].values
    # calls_test = df_test['call'].values
    # y_test = np.array([label2ix[call] for call in calls_test])

    return X_train_text, y_train, label2ix

if __name__ == '__main__':
    params = [
        ('stacksize', 50, int),
        ('normalize', 'mvn', str),
        ('n_noise_fr', 50, int),
        ('remove_dc', False, string_to_bool),
        ('medfilt_t', 0, int),
        ('medfilt_s', 0, int),
        ('fs', 16000, int),
        ('nfft', 2048, int),
        ('scale', 'mel', str),
        ('nfilt', 40, int),
        ('lowerf', 133.3333, float),
        ('upperf', 6855.4976, float),
        ('compression', 'log', str),
        ('deltas', False, string_to_bool),
        ('dct', False, string_to_bool),

        ('C', 1.0, float),
        ('gamma', 1.0, float),
        ('alpha', 1.0, float),
        ('kernel', 'rbf', str),
        ('n_components', 500, int)
    ]
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('STUFF', nargs='*')
        for name, default, typ in params:
            parser.add_argument(
                '-{}'.format(name),
                action='store',
                dest=name,
                default=default,
                type=typ,
                help=name
            )
        return vars(parser.parse_args())
    args = parse_args()

    CACHE_FILE = 'smac/cache.joblib.pkl'
    cache = joblib.load(CACHE_FILE)
    cache_keys = set([name for key in cache.keys() for name, value in key])
    param_dict = {k: args[k] for k in sorted(cache_keys)}
    # if 'medfilt_s' in param_dict:
    #     param_dict['medfilt_s'] = (param_dict['medfilt_s'],
    #                                param_dict['medfilt_s'])
    cache_key = tuple([(k, param_dict[k]) for k in sorted(cache_keys)])
    try:
        cache = cache[cache_key]
    except KeyError as exc:
        import sys
        print >>sys.stderr, cache.keys()
        print >>sys.stderr, cache_key
        raise exc

    X_text, y, label2ix = \
        load_monkey(MONKEY)
    X = np.vstack([cache['data'][(filename, start, end)]
                   for filename, start, end in X_text])

    if APPROXIMATE:
        clf_param_names = ['C', 'gamma', 'alpha', 'kernel', 'n_components']
    else:
        clf_param_names = ['C', 'gamma', 'alpha', 'kernel']
    clf_params = {
        name: args[name]
        for name in clf_param_names
    }

    score = 0
    for state in xrange(NFOLDS):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=state)
        if APPROXIMATE:
            clf = classifier.ApproximateKernelClassifier(
                n_jobs=NJOBS, verbose=VERBOSE,
                **clf_params
            )
        else:
            clf = classifier.SparseExactKernelClassifier(
                n_jobs=NJOBS, verbose=VERBOSE,
                **clf_params
            )
        # clf = SparseApproximateKernelClassifier(
        clf.fit(
            X_train, y_train
        )
        score += clf.score(X_test, y_test)
    print 'Result for SMAC: SUCCESS, 0, 0, {:.5f}, 0'.format(1-score/NFOLDS)
