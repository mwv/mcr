"""transcription_train:

"""

from __future__ import division

import numpy as np
import pandas as pd
import joblib
from pystruct.models import ChainCRF
from pystruct.learners import OneSlackSSVM

import mcr.classifier
from mcr.util import verb_print, load_config, resample

from mcr.load_transcription import extract_features, posteriors


def train_seq(X, y, crf_params):
    X_ = [X[k] for k in sorted(X.keys())]
    y_ = [y[k] for k in sorted(y.keys())]
    class_sizes = np.bincount(np.hstack(y_))
    cw = 1./class_sizes
    cw = cw / cw.sum()
    return OneSlackSSVM(
        model=ChainCRF(inference_method='max-product',
                       class_weight=cw),
        max_iter=100000, verbose=False, **crf_params
    ).fit(X_, y_)


def train_am(X, y, svm_params, class_weight, approximate):
    X_ = np.vstack(
        X[fname]
        for fname in sorted(X.keys())
    )
    y_ = np.hstack(
        y[fname]
        for fname in sorted(y.keys())
    )
    X_, y_ = resample(X_, y_)
    if class_weight:
        svm = mcr.classifier.WeightedSparseKernelClassifier(
            mode='approximate' if approximate else 'exact',
            verbose=False
        )
    else:
        svm = mcr.classifier.SparseKernelClassifier(
            mode='approximate' if approximate else 'exact',
            verbose=False
        )
    return svm.fit(X_, y_)


if __name__ == '__main__':
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser(
            prog='transcription_train.py',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='train model for transcriptions')
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

    with verb_print('loading configuration from {}'.format(config_file),
                    verbose=verbose):
        config = load_config(config_file)
        spec_config = config['features']['spectral']
        del spec_config['normalize']
        spec_config['noise_fr'] = spec_config['n_noise_fr']
        del spec_config['n_noise_fr']

        vad_config = config['features']['vad']

        stacksize = config['features']['stacksize']
        frate = int(1./spec_config['window_shift'])

        if spec_config['fs'] != vad_config['fs']:
            raise ValueError(
                'samplerates in spectral and vad configuration'
                'should be the same ({} and {})'.format(
                    spec_config['fs'], vad_config['fs']
                )
            )
        svm_params = config['model']['svm']

        CLASS_WEIGHT = svm_params['class_weight']
        if not isinstance(CLASS_WEIGHT, bool):
            raise ValueError(
                'invalid value for class_weight: {}'.format(CLASS_WEIGHT)
            )
        del svm_params['class_weight']
        APPROXIMATE = svm_params['approximate']
        if not isinstance(APPROXIMATE, bool):
            raise ValueError(
                'invalid value for approximation: {}'.format(APPROXIMATE)
            )
        del svm_params['approximate']

        crf_params = config['model']['crf']

        params_to_serialize = dict(
            spec_config=spec_config,
            vad_config=vad_config,
            stacksize=stacksize,
            frate=frate,
            smoothing=config['model']['smoothing']
        )

    with verb_print('reading stimuli from {}'.format(data_file),
                    verbose=verbose):
        df = pd.read_csv(data_file)
        ix2label = dict(enumerate(sorted(set(df.label))))
        label2ix = {label: ix for ix, label in ix2label.items()}

    with verb_print('extracting features', verbose=verbose):
        X, y = extract_features(
            df, label2ix, spec_config, vad_config, stacksize, frate
        )

    with verb_print('training acoustic model', verbose=verbose):
        am = train_am(X, y, svm_params, CLASS_WEIGHT, APPROXIMATE)

    with verb_print('svm posteriors', verbose=verbose):
        X_post = posteriors(am, X)

    with verb_print('training sequential model', verbose=verbose):
        crf = train_seq(X_post, y, crf_params)

    joblib.dump(
        (am, crf, label2ix, params_to_serialize),
        output_file, compress=9
    )
