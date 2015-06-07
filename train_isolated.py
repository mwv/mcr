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


import inspect
import warnings
from functools import partial

import pandas as pd
import numpy as np
from scikits.audiolab import wavread
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.metrics import make_scorer, f1_score
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import joblib
from joblib import Parallel, delayed

from spectral import Spectral
from zca import ZCA

from util import load_config, verb_print

USE_AUDIO_CACHE = True
USE_SPEC_CACHE = True
USE_DISK_CACHE = True

_wav_cache = {}
def _load_wav(fname, fs=16000):
    """
    Optionally memoized audio loader.
    """
    key = fname
    if not key in _wav_cache:
        sig, fs_, _ = wavread(fname)
        if fs != fs_:
            raise ValueError('sampling rate should be {0}, not {1}. '
                             'please resample.'.format(fs, fs_))
        if len(sig.shape) > 1:
            warnings.warn('stereo audio: merging channels')
            sig = (sig[:, 0] + sig[:, 1]) / 2
        if USE_AUDIO_CACHE:
            _wav_cache[key] = sig
        else:
            return sig
    return _wav_cache[key]

_noise_cache = {}
def _extract_noise(fname, fs, n_noise_fr, encoder):
    cfg = (('fs', encoder.fs),
           ('window_length', encoder.window_length),
           ('window_shift', encoder.window_shift),
           ('nfft', encoder.nfft),
           ('remove_dc', encoder.remove_dc),
           ('medfilt_t', encoder.medfilt_t),
           ('medfilt_s', encoder.medfilt_s),
           ('pre_emph', encoder.pre_emph))
    key = (fname, cfg)
    if not key in _noise_cache:
        sig = _load_wav(fname, fs=fs)
        nsamples = (n_noise_fr + 2) * encoder.fshift
        spec = encoder.get_spectrogram(sig[:nsamples])[2:, :]
        noise = spec.mean(axis=0)
        _noise_cache[key] = noise
    return _noise_cache[key]

_spec_cache = {}
def _extract_features(fname, fs, encoder):
    key = (fname, tuple(sorted(encoder.config.items())))
    # key = (fname, encoder.config)
    if not key in _spec_cache:
        sig = _load_wav(fname, fs=fs)
        spec = encoder.transform(sig)
        if USE_SPEC_CACHE:
            _spec_cache[key] = spec
        else:
            return spec
    return _spec_cache[key]

# def extract_spec_at(fname, fs, start, stacksize, encoder):
#     spec = _extract_spec(fname, fs, encoder)
#     part = spec[start: start + stacksize]
#     part = np.pad(part,
#                   ((0, stacksize-part.shape[0]),
#                    (0,0)),
#                   'constant')
#     return part.flatten()

_feat_cache = {}
def extract_features_at(fname, fs, start, stacksize, encoder, n_noise_fr=0,
                        buffer_length=0.5):
    """Extract features at a certain point in time.

    Parameters
    ----------
    fname : string
        filename
    fs : int
        samplerate
    start : float
        start position in seconds
    stacksize : int
        number of feature frames to extract starting from start
    encoder : Spectral object
        feature extractor
    n_noise_fr : int
        number of noise frames
    buffer_length : float
        pre- and post-padding time in seconds

    Returns
    -------
    ndarray
        vector of size stacksize * encoder.n_features
    """
    key = (fname, fs, start, stacksize, tuple(sorted(encoder.config.items())),
           n_noise_fr, buffer_length)
    if not key in _feat_cache:
        # load signal and pad for buffer size
        sig = _load_wav(fname, fs=fs)
        sig = np.pad(sig,
                     (int(buffer_length*fs), int(buffer_length*fs)),
                     'constant')

        # get noise from start of file
        noise = _extract_noise(fname, fs, n_noise_fr, encoder)

        # determine buffer and call start and end points in smp and fr
        buffer_len_smp = int(buffer_length * fs)
        buffer_len_fr = int(buffer_len_smp / encoder.fshift)

        stacksize_smp = int(stacksize * encoder.fshift)
        call_start_smp = int(start * fs) + buffer_len_smp
        call_end_smp = call_start_smp + stacksize_smp

        # the part we're gonna cut out: [buffer + call + buffer]
        slice_start_smp = call_start_smp - buffer_len_smp
        slice_end_smp = call_end_smp + buffer_len_smp
        sig_slice = sig[slice_start_smp: slice_end_smp]

        # extract features and cut out call
        feat = encoder.transform(sig_slice, noise_profile=noise)
        _feat_cache[key] = \
            feat[buffer_len_fr: buffer_len_fr + stacksize].flatten()
    return _feat_cache[key]


class IdentityTransform(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


class FeatureLoader(TransformerMixin, BaseEstimator):
    """
    Wrapper to load audio and extract features in a single transformer

    Parameters
    ----------
    fs : int
        sampling rate of the audio in Hz
    stacksize : int
        number of consecutive frames to stack in a single stimulus
    normalize : {None, 'minmax', 'mvn', 'zca'}
        normalization method to use
    """

    normalization_methods = [None, 'minmax', 'mvn', 'zca']

    # arguments to spectral constructor, if any of these are changed
    # the spectral encoder needs to be rebuilt
    spec_arg_names = inspect.getargspec(Spectral.__init__).args[1:]

    def __init__(self, stacksize=40, normalize='mvn', n_jobs=1, **spec_kwargs):
        self.spec_kwargs = spec_kwargs
        if not 'fs' in self.spec_kwargs:
            self.spec_kwargs['fs'] = 16000
        self.noise_fr = self.spec_kwargs.get('noise_fr', 0)
        self.spec_kwargs['noise_fr'] = 0

        self.fs = spec_kwargs['fs']

        self.n_jobs = n_jobs

        self.set_encoder()

        self.stacksize = stacksize
        self.normalize = normalize

    def set_encoder(self):
        self.encoder = Spectral(**self.spec_kwargs)
        self.n_features = self.encoder.n_features

    @property
    def normalize(self):
        return self._normalize

    @normalize.setter
    def normalize(self, v):
        if not v in self.normalization_methods:
            raise ValueError(
                'normalization method must be one of {}, not {}'
                .format('[{}]'.format(
                    ','.join(map(str, self.normalization_methods))),
                        v))
        self._normalize = v
        if v == 'minmax':
            self.normalizer = MinMaxScaler(feature_range=(0,1))
        elif v == 'mvn':
            self.normalizer = StandardScaler()
        elif v == 'zca':
            self.normalizer = ZCA()
        else:
            self.normalizer = IdentityTransform()

    def get_params(self, deep=True):
        p = {k: getattr(self, k) for k in self.spec_arg_names}
        p['normalize'] = self.normalize
        p['fs'] = self.fs
        p['stacksize'] = self.stacksize
        return p

    def __getattr__(self, attr):
        if attr in self.spec_arg_names:
            return getattr(self.encoder, attr)
        else:
            raise AttributeError('{0!r} object has no attribute {1!r}'
                                 .format(self.__class__, attr))

    def __setattr__(self, attr, value):
        if attr in self.spec_arg_names:
            self.spec_kwargs[attr] = value
            self.set_encoder()
        else:
            super(FeatureLoader, self).__setattr__(attr, value)

    def get_specs(self, X):
        r = np.empty((X.shape[0], self.n_features*self.stacksize))
        r = Parallel(n_jobs=self.n_jobs, verbose=verbose)(
            delayed(extract_features_at)(X[ix][0], self.fs,
                                         float(X[ix][1]),
                                         self.stacksize, self.encoder,
                                         self.noise_fr)
            for ix in xrange(X.shape[0])
        )
        # for ix in xrange(X.shape[0]):
        #     fname, start = X[ix][0], int(X[ix][1]*self.fs/self.encoder.fshift)
        #     spec = _extract_spec(fname, self.fs, self.encoder)
        #     part = spec[start: start+self.stacksize]
        #     # pad if too short
        #     part = np.pad(part,
        #                   ((0, self.stacksize-part.shape[0]),
        #                    (0, 0)),
        #                   'constant')
        #     r[ix] = part.flatten()
        return r

    def fit(self, X, y=None):
        """Load audio and optionally estimate mean and covar

        Parameters
        ----------
        X : ndarray with columns
            filename, start, end
        y :
        """
        r = self.get_specs(X)
        self.normalizer.fit(r)
        return self

    def transform(self, X, y=None):
        """Load audio and perform feature extraction.

        Parameters
        ----------
        X : ndarray
        """
        r = self.get_specs(X)
        return self.normalizer.transform(r)

    def fit_transform(self, X, y=None):
        r = self.get_specs(X)
        return self.normalizer.fit_transform(r)


def split_config(kwargs):
    """ split configuration into dynamic (multiple values) and static
    (single value) dicts
    """
    dynamic = {}
    static = {}
    for k, v  in kwargs.iteritems():
        if k == 'medfilt_s':
            if isinstance(v[0], list):
                dynamic[k] = map(tuple, v)
                static[k] = tuple(v[0])
            else:
                static[k] = tuple(v)
        else:
            if isinstance(v, list):
                dynamic[k] = v
                static[k] = v[0]
            else:
                static[k] = v
    return dynamic, static


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

    with verb_print('reading data file', verbose=verbose):
        df = pd.read_csv(data_file)
        X = df[['filename', 'start', 'end']].values
        calls = df['call']
        label2ix = {k:i for i, k in enumerate(np.unique(calls))}
        y = np.array([label2ix[call] for call in calls])

    with verb_print('loading configuration', verbose=verbose):
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
        svm_dynamic, svm_static = split_config(svm_kwargs)
        for k, v in svm_dynamic.iteritems():
            param_grid['clf__{}'.format(k)] = v

        spec_kwargs = config['features']['spectral']
        spec_kwargs.update(config['features']['preprocessing'])
        spec_dynamic, spec_static = split_config(spec_kwargs)
        for k, v in spec_dynamic.iteritems():
            param_grid['features__{}'.format(k)] = v

    with verb_print('preloading spectral features', verbose=verbose):
        for params in ParameterGrid(spec_dynamic):
            fl = FeatureLoader(stacksize=stacksize0,
                               normalize=normalize0,
                               n_jobs=n_jobs,
                               **spec_static)
            fl.set_params(**params)
            fl.fit(X)

    with verb_print('preparing pipeline', verbose=verbose):
        pipeline = Pipeline([('features', FeatureLoader(stacksize=stacksize0,
                                                        normalize=normalize0,
                                                        n_jobs=n_jobs,
                                                        **spec_static)),
                             ('clf', SVC(**svm_static))])
        average = 'binary' if len(label2ix) == 2 else 'micro'
        clf = GridSearchCV(pipeline, param_grid=param_grid,
                           scoring=make_scorer(partial(f1_score,
                                                       average=average)),
                           n_jobs=n_jobs,
                           verbose=0 if verbose else 0)
    with verb_print('training classifier', verbose=verbose):
        clf.fit(X, y)
    with verb_print('saving output to {}'.format(output_file),
                    verbose=verbose):
        joblib.dump(clf, output_file, compress=9)
    print clf.best_params_
    print clf.best_score_
