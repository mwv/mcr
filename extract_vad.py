#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: extract_vad.py
# date: Thu June 11 16:44 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""extract_vad: extract vads and save out to single file

"""

from __future__ import division

import numpy as np
import pandas as pd

from vad import VAD
from util import verb_print, load_config
from joblib import dump, Parallel, delayed
from scikits.audiolab import wavread

def extract_vad(fname, fs, window_length, window_shift, noise_fr):
    sig, fs_, _ = wavread(fname)
    if fs != fs_:
        raise ValueError('expected samplerate: {}, got {}'.format(fs, fs_))
    x = VAD(
        fs, win_size_sec=window_length, win_hop_sec=window_shift
    ).activations(sig, fs, noise_fr)
    return (fname, x)

if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(
            prog='extract_vad.py',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='extract voice activation features')
        parser.add_argument('datafile', metavar='DATAFILE',
                            nargs=1,
                            help='file with stimuli')
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
        config = load_config(config_file)['features']

        fs = config['spectral']['fs']
        window_length = config['vad']['window_length']
        window_shift = config['vad']['window_shift']
        noise_fr = config['vad']['noise_fr']

    with verb_print('reading stimuli from {}'.format(data_file), verbose):
        df = pd.read_csv(data_file)
        filenames = df.filename.unique()

    with verb_print('extracting activations from {} files'
                    .format(len(filenames)), verbose):
        r = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(extract_vad)(
                fname,
                fs,
                window_length,
                window_shift,
                noise_fr
            )
            for fname in filenames
        )
    r = {fname: x for fname, x in r}
    dump(r, output_file, compress=9)
