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

import os.path as path
import inspect
import warnings
from itertools import izip
from pprint import pformat
import copy_reg
import types

import numpy as np
from numpy.lib.stride_tricks import as_strided

from scikits.audiolab import wavread
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator
from joblib import Parallel, delayed
from spectral import Spectral
from zca import ZCA


# these two methods are here to work around pickle's inability to deal
# with instancemethod objects. below, we call joblib's Parallel on
# FeatureLoader.extract_features_at. joblib uses multiprocessing, which
# pickles the methods.
# def _pickle_method(method):
#     func_name = method.im_func.__name__
#     obj = method.im_self
#     cls = method.im_class
#     return _unpickle_method, (func_name, obj, cls)

# def _unpickle_method(func_name, obj, cls):
#     for cls in cls.mro():
#         try:
#             func = cls.__dict__[func_name]
#         except KeyError:
#             pass
#         else:
#             break
#     return func.__get__(obj, cls)

# copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


# this function goes with FeatureLoader but is defined outside it,
# because I cannot get the parallellization to work on instance methods
def extract_features_at(sig, noise, start, stacksize, encoder,
                        buffer_length=0.1):
    # determine buffer and call start and end points in smp and fr
    buffer_len_smp = int(buffer_length * encoder.fs)
    buffer_len_fr = int(buffer_len_smp / encoder.fshift)

    stacksize_smp = int(stacksize * encoder.fshift)
    call_start_smp = int(start * encoder.fs) + buffer_len_smp
    call_end_smp = call_start_smp + stacksize_smp

    # the part we're gonna cut out: [buffer + call + buffer]
    slice_start_smp = call_start_smp - buffer_len_smp
    slice_end_smp = call_end_smp + buffer_len_smp

    # pad signal
    sig = np.pad(sig,
                 (buffer_len_smp,
                  buffer_len_smp),
                 'constant')
    sig_slice = sig[slice_start_smp: slice_end_smp]

    # extract features and cut out call
    feat = encoder.transform(sig_slice, noise_profile=noise)
    feat = feat[buffer_len_fr: buffer_len_fr + stacksize]

    # pad at the end
    feat = np.pad(feat,
                  ((0, stacksize - feat.shape[0]), (0, 0)),
                  'constant')
    return feat


class IdentityTransform(TransformerMixin, BaseEstimator):
    """Dummy Transformer that implements the identity transform
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

class FeatureLoader(TransformerMixin, BaseEstimator):
    def __init__(self, stacksize=40, normalize='mvn', n_noise_fr=0,
                 fs=16000,
                 window_length=0.050,
                 window_shift=0.010,
                 nfft=1024,
                 scale='mel',
                 lowerf=120,
                 upperf=7000,
                 nfilt=40,
                 taper_filt=True,
                 compression='log',
                 dct=False,
                 nceps=13,
                 log_e=True,
                 lifter=22,
                 deltas=False,
                 remove_dc=False,
                 medfilt_t=0,
                 medfilt_s=(0,0),
                 noise_fr=0,
                 pre_emph=0.97,
                 feat_cache=None, noise_cache=None, wav_cache=None,
                 n_jobs=1, verbose=False):
        self.stacksize = stacksize
        self.normalize = normalize
        if self.normalize == 'mvn':
            self.normalizer = StandardScaler()
        elif self.normalize == 'zca':
            self.normalizer = ZCA()
        elif self.normalize == 'minmax':
            self.normalizer = MinMaxScaler()
        else:
            self.normalizer = IdentityTransform()
        self.n_noise_fr = n_noise_fr
        self.fs = fs
        self.window_length = window_length
        self.window_shift = window_shift
        self.nfft = nfft
        self.scale = scale
        self.lowerf = lowerf
        self.upperf = upperf
        self.nfilt = nfilt
        self.taper_filt = taper_filt
        self.compression = compression
        self.dct = dct
        self.nceps = nceps
        self.log_e = log_e
        self.lifter = lifter
        self.deltas = deltas
        self.remove_dc = remove_dc
        self.medfilt_t = medfilt_t
        self.medfilt_s = medfilt_s
        self.noise_fr = noise_fr
        self.pre_emph = pre_emph

        self.n_jobs = n_jobs
        self.verbose = verbose

        self.encoder = Spectral(
            fs=fs,
            window_length=window_length,
            window_shift=window_shift,
            nfft=nfft,
            scale=scale,
            lowerf=lowerf,
            upperf=upperf,
            nfilt=nfilt,
            taper_filt=taper_filt,
            compression=compression,
            dct=dct,
            nceps=nceps,
            log_e=log_e,
            lifter=lifter,
            deltas=deltas,
            remove_dc=remove_dc,
            medfilt_t=medfilt_t,
            medfilt_s=medfilt_s,
            noise_fr=noise_fr,
            pre_emph=pre_emph
        )
        self.D = self.encoder.n_features * self.stacksize
        self.wav_cache = wav_cache if wav_cache else {}
        self.noise_cache = noise_cache if noise_cache else {}
        self.feat_cache = feat_cache if feat_cache else {}

        # print '  FeatureLoader: self.wav_cache: {}'\
        #     .format(len(self.wav_cache))
        # print '  FeatureLoader: self.noise_cache: {}'\
        #     .format(len(self.noise_cache))
        # print '  FeatureLoader: self.feat_cache: {}'\
        #     .format(len(self.feat_cache))

    def clear_cache(self):
        self.wav_cache = {}
        self.noise_cache = {}
        self.feat_cache = {}

    def get_params(self, deep=True):
        p = super(FeatureLoader, self).get_params()
        del p['n_jobs']
        del p['verbose']
        return p

    def get_key(self):
        """'Frozen' dictionary representation of this object's parameters.
        Used as key in caching.
        """
        p = self.get_params()
        del p['wav_cache']
        del p['noise_cache']
        del p['feat_cache']
        return tuple(sorted(p.items()))

    def _load_wav(self, fname):
        """
        Memoized audio loader.
        """
        key = fname
        if not key in self.wav_cache:
            # print '  FeatureLoader: _load_wav: key not in cache: {}'.format(
            #     path.basename(key)
            # )
            sig, fs_, _ = wavread(fname)
            if self.fs != fs_:
                raise ValueError('sampling rate should be {0}, not {1}. '
                                 'please resample.'.format(self.fs, fs_))
            if len(sig.shape) > 1:
                warnings.warn('stereo audio: merging channels')
                sig = (sig[:, 0] + sig[:, 1]) / 2
            self.wav_cache[key] = sig
        return self.wav_cache[key]

    def _fill_noise_cache(self, X):
        for fname in X[:, 0]:
            self._extract_noise(fname)

    def _extract_noise(self, fname):
        cfg = (
            ('fs', self.fs),
            ('window_length', self.window_length),
            ('window_shift', self.window_shift),
            ('nfft', self.nfft),
            ('remove_dc', self.remove_dc),
            ('medfilt_t', self.medfilt_t),
            ('medfilt_s', self.medfilt_s),
            ('pre_emph', self.pre_emph)
        )
        key = (fname, cfg)
        if not key in self.noise_cache:
            if self.n_noise_fr == 0:
                self.noise_cache[key] = None
            else:
                sig = self._load_wav(fname)
                nsamples = (self.n_noise_fr + 2) * self.encoder.fshift
                spec = self.encoder.get_spectrogram(sig[:nsamples])[2:, :]
                noise = spec.mean(axis=0)
                noise = np.clip(noise, 1e-4, np.inf)
                self.noise_cache[key] = noise
        return self.noise_cache[key]

    def _fill_feat_cache(self, X_keys):
        sigs = [self._load_wav(fname) for fname, _ in X_keys]
        noises = [self._extract_noise(fname) for fname, _ in X_keys]
        p = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(extract_features_at)(
                sig, noise, start, self.stacksize, self.encoder)
            for (fname, start), sig, noise in izip(X_keys, sigs, noises)
        )
        r = {x_key: feat
             for x_key, feat in izip(X_keys, p)}
        key = self.get_key()
        self.feat_cache[key].update(r)

    def get_specs(self, X):
        key = self.get_key()
        # list of [(filename, start)]
        X_keys = [(X[ix, 0], X[ix, 1]) for ix in xrange(X.shape[0])]
        if key in self.feat_cache:
            # print '  FeatureLoader: get_specs: key in cache'
            # check for missing keys
            missing_X_keys = [
                x_key
                for x_key in X_keys
                if not x_key in self.feat_cache[key]
            ]
            self._fill_feat_cache(missing_X_keys)
        else:
            # print '  FeatureLoader: get_specs: key not in cache ({}): {}'\
            #     .format(
            #         len(self.feat_cache),
            #         key,
            # )
            # print '  FeatureLoader: _feat_cache.keys(): {}'\
            #     .format(pformat(self._feat_cache.keys()))
            self.feat_cache[key] = {}
            self._fill_feat_cache(X_keys)
        return np.vstack((self.feat_cache[key][x_key] for x_key in X_keys))

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
        r = self.normalizer.transform(r)
        return as_strided(
            r,
            shape=(r.shape[0]//self.stacksize, r.shape[1]*self.stacksize),
            strides=(r.strides[0]*self.stacksize, r.strides[1])
        )

    def fit_transform(self, X, y=None):
        r = self.get_specs(X)
        r = self.normalizer.fit_transform(r)
        return as_strided(
            r,
            shape=(r.shape[0]//self.stacksize, r.shape[1]*self.stacksize),
            strides=(r.strides[0]*self.stacksize, r.strides[1])
        )


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

def ensure_list(kwargs):
    r = {}
    for k, v in kwargs.iteritems():
        if k == 'medfilt_s':
            if isinstance(v[0], list):
                r[k] = map(tuple, v)
            else:
                r[k] = (tuple(v), )
        else:
            if isinstance(v, list):
                r[k] = v
            else:
                r[k] = (v,)
    return r
