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

import pandas as pd
import numpy as np
np.seterr(all='raise')
from sklearn.grid_search import GridSearchCV, ParameterGrid
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

        stacksize = config['features']['nframes']
        normalize = config['features']['normalize']

        param_grid = {}
        if isinstance(stacksize, list):
            stacksize0 = stacksize[0]
            param_grid['features__stacksize'] = stacksize
        else:
            stacksize0 = stacksize
        if isinstance(normalize, list):
            normalize0 = normalize[0]
            param_grid['features__normalize'] = normalize
        else:
            normalize0 = normalize

        svm_kwargs = config['svm']
        svm_dynamic, svm_static = load_isolated.split_config(svm_kwargs)
        for k, v in svm_dynamic.iteritems():
            param_grid['clf__{}'.format(k)] = v

        spec_kwargs = config['features']['spectral']
        spec_kwargs.update(config['features']['preprocessing'])

        noise_fr = spec_kwargs.get('noise_fr', 0)
        del spec_kwargs['noise_fr']
        if isinstance(noise_fr, list):
            param_grid['features__noise_fr'] = noise_fr

        spec_dynamic, spec_static = load_isolated.split_config(spec_kwargs)
        if isinstance(noise_fr, list):
            spec_dynamic['noise_fr'] = noise_fr
        for k, v in spec_dynamic.iteritems():
            param_grid['features__{}'.format(k)] = v

    # with verb_print('preloading audio', verbose=verbose):
    #     n_iter = reduce(operator.mul, map(len, spec_dynamic.values()))
    #     for ix, params in enumerate(ParameterGrid(spec_dynamic)):
    #         print 'combination {}/{}'.format(ix, n_iter)
    #         print params
    #         print n_jobs
    #         fl = load_isolated.FeatureLoader(
    #             stacksize=stacksize0,
    #             normalize=normalize0,
    #             noise_fr=noise_fr,
    #             n_jobs=n_jobs,
    #             verbose=True,
    #             **spec_static
    #         )
    #         fl.set_params(**params)

    #         for fname in X[:,0]:
    #             load_isolated._load_wav(fname, fs=fl.encoder.fs)
    #             load_isolated._extract_noise(
    #                 fname, fl.encoder.fs, params.get('noise_fr', 0),
    #                 fl.encoder
    #             )
    #         X_ = fl.get_specs(X)
    #         assert (X_.shape[0] == X.shape[0])
    #         key = fl.get_key()
    #         load_isolated._feat_cache[key] = {
    #             (X[ix, 0], float(X[ix, 1])): X_[ix]
    #             for ix in xrange(X.shape[0])
    #         }

    n_grid_values = reduce(operator.mul, map(len, param_grid.values()))
    average_method = 'binary' if len(label2ix) == 2 else 'micro'
    scorer = make_f1_score(average_method)
    with verb_print('preparing pipeline', verbose=verbose):
        pipeline = Pipeline(
            [('features', load_isolated.FeatureLoader(
                stacksize=stacksize0,
                normalize=normalize0,
                verbose=False,
                **spec_static)),
             ('clf', SVC(verbose=False, **svm_static))])

        if n_grid_values == 1:
            clf = pipeline
            clf.set_params(**iter(ParameterGrid(param_grid)).next())
        else:
            clf = GridSearchCV(pipeline, param_grid=param_grid,
                               scoring=scorer,
                               n_jobs=n_jobs,
                               verbose=10 if verbose else 0)

    with verb_print('training classifier', verbose=verbose):
        clf.fit(X, y)
    with verb_print('saving output to {}'.format(output_file),
                    verbose=verbose):
        joblib.dump((clf, label2ix), output_file, compress=9)
    if n_grid_values == 1:
        y_pred = clf.predict(X)
        print clf.get_params()
        print scorer._score_func(y, y_pred)
    else:
        print clf.best_params_
        print clf.best_score_
