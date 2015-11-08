"""transcription_predict: use a trained classifier to transcribe audio.

"""

from __future__ import division

import numpy as np
np.seterr(all='raise')
import pandas as pd
import joblib
from itertools import izip

from mcr.load_transcription import extract_features, posteriors
from mcr.util import verb_print

def predictions_to_dataframe(y_pred, label2ix, frate):
    ix2label = {ix: label for label, ix in label2ix.iteritems()}
    window_shift = 1./frate
    data = []
    for fname in sorted(y_pred.keys()):
        pred = y_pred[fname]
        switch = [-1] + list(np.nonzero(np.diff(pred))[0]) + [len(pred)]
        for start, end in izip(switch, switch[1:]):
            start_fr = start + 1
            end_fr = end + 1
            label = ix2label[pred[start_fr]]
            start = start_fr * window_shift
            end = end_fr * window_shift
            data.append((fname, start, end, label))
    return pd.DataFrame(data, columns=['filename', 'start', 'end', 'label'])

if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(
            prog='transcription_predict.py',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='use trained classifier to transcribe audio')
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

    with verb_print('loading classifier from {}'.format(clf_file),
                    verbose=verbose):
        am, crf, label2ix, feat_params = joblib.load(clf_file)
        spec_config = feat_params['spec_config']
        vad_config = feat_params['vad_config']
        stacksize = feat_params['stacksize']
        frate = feat_params['frate']
        smoothing = feat_params['smoothing']

    with verb_print('extracting features', verbose=verbose):
        X = extract_features(
            df, label2ix, spec_config, vad_config, stacksize, frate,
            return_y=False
        )

    with verb_print('svm posteriors', verbose=verbose):
        X_post = posteriors(am, X)

    with verb_print('crf predictions', verbose=verbose):
        y_pred = {
            fname: crf.predict([post])[0]
            for fname, post in X_post.iteritems()
        }
        if smoothing > 0:
            for fname in y_pred.keys():
                pred = y_pred[fname]
                switch = \
                    [-1] + \
                    list(np.nonzero(np.diff(pred))[0]) + \
                    [len(pred)]
                for start, end in izip(switch, switch[1:]):
                    if end - start < smoothing:
                        pred[start+1: end+1] = pred[start]
                y_pred[fname] = pred

    with verb_print('formatting predictions', verbose=verbose):
        pred_df = predictions_to_dataframe(
            y_pred, label2ix, frate
        )

    with verb_print('writing predictions', verbose=verbose):
        pred_df.to_csv(output_file, index=False, float_format='%.3f')
