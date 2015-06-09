#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: load_isolated.py
# date: Tue June 09 12:59 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""load_isolated:

"""

from __future__ import division

import numpy as np
import inspect
import warnings

from scikits.audiolab import wavread
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator
from joblib import Parallel, delayed
from spectral import Spectral
from zca import ZCA

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
def _extract_noise(fname, fs, noise_fr, encoder):
    cfg = (('fs', encoder.fs),
           ('window_length', encoder.window_length),
           ('window_shift', encoder.window_shift),
           ('nfft', encoder.nfft),
           ('remove_dc', encoder.remove_dc),
           ('medfilt_t', encoder.medfilt_t),
           ('medfilt_s', encoder.medfilt_s),
           ('pre_emph', encoder.pre_emph))
    key = (fname, cfg)
    if noise_fr == 0:
        _noise_cache[key] = None
    elif not key in _noise_cache:
        sig = _load_wav(fname, fs=fs)
        nsamples = (noise_fr + 2) * encoder.fshift
        spec = encoder.get_spectrogram(sig[:nsamples])[2:, :]
        noise = spec.mean(axis=0)
        noise = np.clip(noise, 1e-4, np.inf)
        _noise_cache[key] = noise
    return _noise_cache[key]


def extract_features_at(fname, fs, start, stacksize, encoder, noise_fr=0,
                        buffer_length=0.1):
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
    noise_fr : int
        number of noise frames
    buffer_length : float
        pre- and post-padding time in seconds

    Returns
    -------
    ndarray
        vector of size stacksize * encoder.n_features
    """
    # load signal and pad for buffer size
    sig = _load_wav(fname, fs=fs)
    sig = np.pad(sig,
                 (int(buffer_length*fs), int(buffer_length*fs)),
                 'constant')

    # get noise from start of file
    noise = _extract_noise(fname, fs, noise_fr, encoder)

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
    feat = feat[buffer_len_fr: buffer_len_fr + stacksize]

    # pad at the end
    feat = np.pad(feat, ((0, stacksize - feat.shape[0]), (0, 0)), 'constant')

    # flatten the array
    feat = feat.flatten()
    return feat

class IdentityTransform(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

_feat_cache = {}
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

    def __init__(self, stacksize=40, normalize='mvn', noise_fr=0,
                 n_jobs=1, verbose=False, **spec_kwargs):
        self.spec_kwargs = spec_kwargs
        if not 'fs' in self.spec_kwargs:
            self.spec_kwargs['fs'] = 16000
        self.fs = spec_kwargs['fs']
        self.noise_fr = noise_fr

        self.set_encoder()

        self.stacksize = stacksize
        self.normalize = normalize

        self.n_jobs = n_jobs
        self.verbose = verbose

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

    def get_key(self):
        return tuple(sorted(self.get_params().items()))

    def get_specs(self, X):
        key = self.get_key()
        if key in _feat_cache:
            r = np.vstack((_feat_cache[key][(X[ix, 0], X[ix, 1])]
                           for ix in xrange(X.shape[0])))
            # return _feat_cache[key]
        else:
            if self.n_jobs == 1:
                r = np.empty((X.shape[0], self.n_features * self.stacksize),
                             dtype=np.double)
                for ix in xrange(X.shape[0]):
                    fname = X[ix][0]
                    start = float(X[ix][1])
                    feats = extract_features_at(
                        fname, self.fs, start, self.stacksize,
                        self.encoder, self.noise_fr)
                    r[ix] = feats
            else:
                r = np.vstack(
                    Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                        delayed(extract_features_at)(
                            X[ix][0], self.fs,
                            float(X[ix][1]),
                            self.stacksize,
                            self.encoder,
                            self.noise_fr
                        )
                        for ix in xrange(X.shape[0])
                    )
                )
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
                dynamic[k] = (tuple(v),)
                static[k] = tuple(v)
        else:
            if isinstance(v, list):
                dynamic[k] = v
                static[k] = v[0]
            else:
                dynamic[k] = (v,)
                static[k] = v
    return dynamic, static
