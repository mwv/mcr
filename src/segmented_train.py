"""segmented_train: train a classifier for segmented call recognition.

"""

from __future__ import division

import operator
from pprint import pformat

import pandas as pd
import numpy as np
np.seterr(all='raise')
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
import joblib

import mcr.classifier
from mcr.util import load_config, verb_print, make_f1_score
import mcr.load_segmented


if __name__ == '__main__':
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser(
            prog='segmented_train.py',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='train classifier for segmented calls')
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
        labels = df['label'].values
        label2ix = {k: i for i, k in enumerate(np.unique(labels))}
        y = np.array([label2ix[label] for label in labels])

    with verb_print('loading configuration from {}'.format(config_file),
                    verbose=verbose):
        config = load_config(config_file)

        features_params = mcr.load_segmented.ensure_list(config['features'])
        clf_params = mcr.load_segmented.ensure_list(config['svm'])

        CLASS_WEIGHT = clf_params['class_weight'][0]
        if not isinstance(CLASS_WEIGHT, bool):
            raise ValueError(
                'invalid value for class_weight: {}'.format(CLASS_WEIGHT)
            )
        del clf_params['class_weight']

        APPROXIMATE = clf_params['approximate'][0]
        if not isinstance(APPROXIMATE, bool):
            raise ValueError(
                'invalid value for approximation: {}'.format(APPROXIMATE)
            )
        del clf_params['approximate']


        param_grid = {}
        for k, v in features_params.iteritems():
            param_grid['features__{}'.format(k)] = v
        for k, v in clf_params.iteritems():
            param_grid['clf__{}'.format(k)] = v

    with verb_print('preloading audio', verbose=verbose):
        n_iter = reduce(operator.mul, map(len, features_params.values()))
        fl = mcr.load_segmented.FeatureLoader()
        for fname in X[:, 0]:
            fl._load_wav(fname)
        wav_cache = fl.wav_cache

        feat_cache = {}
        noise_cache = {}
        for ix, params in enumerate(ParameterGrid(features_params)):
            # print 'combination {}/{}'.format(ix, n_iter)
            # print params
            fl = mcr.load_segmented.FeatureLoader(verbose=5, n_jobs=n_jobs,
                                             wav_cache=wav_cache, **params)
            fl._fill_noise_cache(X)
            noise_cache.update(fl.noise_cache)
            fl.get_specs(X)
            feat_cache.update(fl.feat_cache)

    if verbose and n_iter > 1:
        print 'PARAMETERGRID:'
        print pformat(param_grid)

    n_grid_values = reduce(operator.mul, map(len, param_grid.values()))
    average_method = 'binary' if len(label2ix) == 2 else 'macro'
    scorer = make_f1_score(average_method)
    with verb_print('preparing pipeline', verbose=verbose):
        feature_loader = mcr.load_segmented.FeatureLoader(
            verbose=False,
            wav_cache=wav_cache,
            noise_cache=noise_cache,
            feat_cache=feat_cache
        )
        if CLASS_WEIGHT:
            svm = mcr.classifier.WeightedSparseKernelClassifier(
                mode='approximate' if APPROXIMATE else 'exact',
                verbose=False
            )
        else:
            svm = mcr.classifier.SparseKernelClassifier(
                mode='approximate' if APPROXIMATE else 'exact',
                verbose=False
            )

        pipeline = Pipeline(
            [('features', feature_loader), ('clf', svm)]
        )
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
        clf.steps[0][1].wav_cache = {}
        clf.steps[0][1].noise_cache = {}
        clf.steps[0][1].feat_cache = {}
        joblib.dump((clf, None, label2ix), output_file, compress=9)
    else:
        feature_loader = mcr.load_segmented.FeatureLoader(
            verbose=False,
            n_jobs=n_jobs
        )
        if CLASS_WEIGHT:
            svm = mcr.classifier.WeightedSparseKernelClassifier(
                mode='approximate' if APPROXIMATE else 'exact',
                verbose=False, n_jobs=n_jobs
            )
        else:
            svm = mcr.classifier.SparseKernelClassifier(
                mode='approximate' if APPROXIMATE else 'exact',
                verbose=False, n_jobs=n_jobs
            )

        clf_ = Pipeline(
            [('features', feature_loader), ('clf', svm)]
        )
        clf_.set_params(**clf.best_params_)
        clf_.fit(X, y)
        clf_.steps[0][1].clear_cache()
        joblib.dump((clf_, clf.grid_scores_, label2ix),
                    output_file, compress=9)
