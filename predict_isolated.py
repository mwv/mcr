#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: test_isolated.py
# date: Tue June 09 19:39 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""predict_isolated:

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

from pprint import pformat

import numpy as np
np.seterr(all='raise')
import pandas as pd
import joblib

from util import verb_print


if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(
            prog='predict_isolated.py',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='use trained classifier for isolated calls to ' \
                        'predict call type')
        parser.add_argument('datafile', metavar='DATAFILE',
                            nargs=1,
                            help='file with test stimuli')
        parser.add_argument('clffile', metavar='CLFFILE',
                            nargs=1,
                            help='file with trained classifier')
        parser.add_argument('output', metavar='OUTPUT',
                            nargs=1,
                            help='output file')
        parser.add_argument('-v' '--verbose',
                            action='store_true',
                            dest='verbose',
                            default=False,
                            help='talk more')
        return vars(parser.parse_args())

    args = parse_args()

    data_file = args['datafile'][0]
    clf_file = args['clffile'][0]
    output_file = args['output'][0]

    verbose = args['verbose']

    with verb_print('reading stimuli from {}'.format(data_file), verbose):
        df = pd.read_csv(data_file)
        X = df[['filename', 'start', 'end']].values
    if verbose:
        print 'loaded {} stimuli'.format(X.shape[0])

    with verb_print('loading classifier from {}'.format(clf_file), verbose):
        clf, label2ix = joblib.load(clf_file)
        ix2label = {ix:label for label, ix in label2ix.iteritems()}
    if verbose:
        print pformat(clf.get_params(deep=False))

    with verb_print('predicting labels', verbose):
        y_pred = clf.predict(X)

    with verb_print('writing output to {}'.format(output_file), verbose):
        with open(output_file, 'w') as fout:
            fout.write('filename,start,end,call\n')
            for ix in xrange(X.shape[0]):
                fout.write(
                    '{fname:s},{start:.3f},{end:.3f},{call:s}\n'
                    .format(
                        fname=X[ix, 0],
                        start=X[ix, 1],
                        end=X[ix, 2],
                        call=ix2label[y_pred[ix]]))
