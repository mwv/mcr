#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: train_isolated.py
# date: Mon April 13 16:40 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""train_isolated:

# train acoustic model (i.e. svm)

# save model file:
# configuration
# trained optimal svm
# grid scores (if any)
# normalization stats for filters
# (isolated call classification does not use vad)


"""

from __future__ import division

# and now for an ugly hack around the incomprehensible path manipulations
# on puck1 and puck2
import sys
try:
    sys.path.remove('/cm/shared/apps/python-anaconda/lib/python2.7/'
                    'site-packages/spectral-0.1.5-py2.7-linux-x86_64.egg')
except ValueError:
    pass

import operator
from pprint import pformat

import pandas as pd
import numpy as np
np.seterr(all='raise')
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import joblib

from util import load_config, verb_print, make_f1_score
import load_isolated


if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(
            prog='train_isolated.py',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='train classifier for isolated calls')
        parser.add_argument('datafile', metavar='DATAFILE',
                            nargs=1,
                            help='file with training stimuli')
        parser.add_argument('config', metavar='CONFIG',
                            nargs=1,
                            help='configuration file')
        parser.add_argument('output', metavar='OUTPUT',
                            nargs=1,
                            help='output file')
        parser.add_argument('-j', '--n-jobs',
                            action='store',
                            dest='n_jobs',
                            default=1,
                            help='number of jobs')
        parser.add_argument('-v', '--verbose',
                            action='store_true',
                            dest='verbose',
                            default=False,
                            help='talk more')
        return vars(parser.parse_args())

    args = parse_args()

    data_file = args['datafile'][0]
    config_file = args['config'][0]
    output_file = args['output'][0]
    n_jobs = int(args['n_jobs'])
    verbose = args['verbose']

    with verb_print('reading stimuli from {}'.format(data_file),
                    verbose=verbose):
        df = pd.read_csv(data_file)
        X = df[['filename', 'start', 'end']].values
        calls = df['call'].values
        label2ix = {k:i for i, k in enumerate(np.unique(calls))}
        y = np.array([label2ix[call] for call in calls])

    with verb_print('loading configuration from {}'.format(config_file),
                    verbose=verbose):
        config = load_config(config_file)

        features_params = load_isolated.ensure_list(config['features'])
        clf_params = load_isolated.ensure_list(config['svm'])

        param_grid = {}
        for k, v in features_params.iteritems():
            param_grid['features__{}'.format(k)] = v
        for k, v in clf_params.iteritems():
            param_grid['clf__{}'.format(k)] = v

    with verb_print('preloading audio', verbose=verbose):
        n_iter = reduce(operator.mul, map(len, features_params.values()))
        fl = load_isolated.FeatureLoader()
        for fname in X[:, 0]:
            fl._load_wav(fname)
        wav_cache = fl.wav_cache

        feat_cache = {}
        noise_cache = {}
        for ix, params in enumerate(ParameterGrid(features_params)):
            print 'combination {}/{}'.format(ix, n_iter)
            print params
            fl = load_isolated.FeatureLoader()
            fl.set_params(**params)
            fl._fill_noise_cache(X)
            noise_cache.update(fl.noise_cache)
            fl.get_specs(X)
            feat_cache.update(fl.feat_cache)

    if verbose:
        print 'PARAMETERGRID:'
        print pformat(param_grid)

    n_grid_values = reduce(operator.mul, map(len, param_grid.values()))
    average_method = 'binary' if len(label2ix) == 2 else 'micro'
    scorer = make_f1_score(average_method)
    with verb_print('preparing pipeline', verbose=verbose):
        pipeline = Pipeline(
            [('features', load_isolated.FeatureLoader(
                verbose=False,
                wav_cache=wav_cache,
                noise_cache=noise_cache,
                feat_cache=feat_cache)),
             ('clf', SVC(verbose=False))])
        if n_grid_values == 1:
            clf = pipeline
            clf.set_params(**iter(ParameterGrid(param_grid)).next())
        else:
            clf = GridSearchCV(pipeline, param_grid=param_grid,
                               scoring=scorer,
                               n_jobs=n_jobs,
                               refit=False,
                               verbose=0 if verbose else 0)

    with verb_print('training classifier', verbose=verbose):
        clf.fit(X, y)
    if n_grid_values == 1:
        y_pred = clf.predict(X)
        print clf.get_params()
        print scorer._score_func(y, y_pred)
        joblib.dump((clf, None, label2ix), output_file, compress=9)
    else:
        print 'BEST PARAMETERS at {}:'.format(clf.best_score_)
        print pformat(clf.best_params_)
        clf_ = Pipeline(
            [('features', load_isolated.FeatureLoader(
                verbose=False,
                n_jobs=n_jobs)),
             ('clf', SVC(verbose=False))])
        clf_.set_params(**clf.best_params_)
        clf_.fit(X, y)
        print classification_report(y, clf_.predict(X))
        clf_.steps[0][1].clear_cache()
        joblib.dump((clf_, clf.grid_scores_, label2ix),
                    output_file, compress=9)
