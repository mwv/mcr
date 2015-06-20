#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: prep_data_isolated.py
# date: Fri June 19 15:12 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""prep_data_isolated: load audio and write .pcs file for smac

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

import os
import os.path as path
import operator
from itertools import imap
import numpy as np
import joblib
import pandas as pd

from sklearn.grid_search import ParameterGrid

from load_isolated import FeatureLoader
import cfg_isolated

try:
    os.makedirs(path.join(os.getcwd(), 'smac'))
except OSError:
    pass

CACHE_FILE = path.join(os.getcwd(), 'smac/cache.joblib.pkl')
PCS_FILE = path.join(os.getcwd(), 'smac/isolated.pcs')
SCENARIO_FILE = path.join(os.getcwd(), 'smac/isolated.scenario')
N_JOBS = 6
VERBOSE = True

def write_pcs(spec_cfg, clf_cfg, fname):
    s = ''
    for name, value in spec_cfg.iteritems():
        s += '{name:s} {{{value:s}}} [{default:s}]\n'.format(
            name=name,
            value=','.join(map(str, value)),
            default=str(value[-1])
        )
    for name, value in clf_cfg.iteritems():
        if isinstance(value[0], str):
            s += '{name:s} {{{value:s}}} [{default}]\n'.format(
                name=name,
                value=','.join(value),
                default=value[-1]
            )
        elif len(value) == 2:
            if name in ['C', 'gamma', 'alpha']:
                # log scale
                s += '{name:s} [{value}] [{default}]l\n'.format(
                    name=name,
                    value=','.join(map(str, value)),
                    default=str(value[-1])
                )
            elif name in ['n_components']:
                # integer values
                s += '{name:s} [{value}] [{default}]i\n'.format(
                    name=name,
                    value=','.join(map(str, value)),
                    default=str(value[-1])
                )
            else:
                s += '{name:s} [{value}] [{default}]\n'.format(
                    name=name,
                    value=','.join(map(str, value)),
                    default=str(value[-1])
                )
        else:
            s += '{name:s} {{{value}}} [{default}]\n'.format(
                name=name,
                value=','.join(map(str, value)),
                default=str(value[-1])
            )
    with open(fname, 'w') as fout:
        fout.write(s)

def write_scenario(script_fname, pcs_fname, scenario_fname):
    with open(scenario_fname, 'w') as fout:
        fout.write("""use-instances = false
numberOfRunsLimit = 100
runObj = QUALITY
pcs-file = {}
algo = python {}""".format(pcs_fname, script_fname))


if __name__ == '__main__':
    write_pcs(
        cfg_isolated.spec_cfg,
        cfg_isolated.clf_cfg,
        PCS_FILE
    )
    write_scenario(
        path.join(os.getcwd(), 'opt_isolated.py'),
        PCS_FILE,
        SCENARIO_FILE
    )
    stimuli = pd.read_csv('resources/annotation.csv')
    stimuli = stimuli[stimuli.call != 'SIL']
    X_text = stimuli[['filename', 'start', 'end']].values
    calls = stimuli.call.values
    spec_keys = sorted(cfg_isolated.spec_cfg.keys())
    features_params = dict(
        stacksize=50,
        normalize="mvn",
        n_noise_fr=50,
        remove_dc=True,
        medfilt_t=0,
        medfilt_s=0,
        fs=16000,
        pre_emph=0.97,
        window_length=0.025,
        window_shift=0.01,
        nfft=1024,
        scale="mel",
        nfilt=40,
        lowerf=133.3333,
        upperf=6855.4976,
        compression="log",
        deltas=False,
        dct=False,
        taper_filt=True
    )
    cache = {}
    n_iter = reduce(operator.mul,
                    imap(len, cfg_isolated.spec_cfg.itervalues()))
    for param_ix, params in enumerate(ParameterGrid(cfg_isolated.spec_cfg)):
        print 'iteration {}/{}'.format(param_ix+1, n_iter)
        features_params.update(params)
        features_params['medfilt_s'] = (features_params['medfilt_s'],
                                        features_params['medfilt_s'])
        current = dict(
            DESCR='transformed data',
            params=features_params.copy(),
            data={},
            calls={}
        )
        loader = FeatureLoader(
            n_jobs=N_JOBS, verbose=VERBOSE, **features_params
        )
        X_tf = loader.fit_transform(X_text)
        for ix, (filename, start, end) in enumerate(X_text):
            key = (filename, start, end)
            current['data'][key] = X_tf[ix, :]
            current['calls'][key]= calls[ix]

        cache_key = tuple([(spec_key, params[spec_key])
                           for spec_key in spec_keys])
        cache[cache_key] = current
    joblib.dump(cache, CACHE_FILE, compress=9)
