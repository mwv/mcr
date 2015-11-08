"""segmented_eval:

"""

from __future__ import division

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from mcr.util import verb_print, pretty_cm


if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(
            prog='segmented_eval.py',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='evaluate predictions for segmented calls')
        parser.add_argument('goldfile', metavar='GOLDFILE',
                            nargs=1,
                            help='file with test (gold) stimuli')
        parser.add_argument('predictfile', metavar='PREDICTFILE',
                            nargs=1,
                            help='file with predicted stimuli')
        parser.add_argument('-v', '--v',
                            action='store_true',
                            dest='verbose',
                            default=False,
                            help='talk more')
        return vars(parser.parse_args())

    args = parse_args()

    gold_file = args['goldfile'][0]
    predict_file = args['predictfile'][0]
    verbose = args['verbose']

    with verb_print('loading gold stimuli from {}'.format(gold_file), verbose):
        df_gold = pd.read_csv(gold_file)

    if verbose:
        print 'loaded {} gold stimuli'.format(len(df_gold))

    with verb_print('loading predicted stimuli from {}'.format(predict_file),
                    verbose):
        df_pred = pd.read_csv(predict_file)

    if verbose:
        print 'loaded {} predicted stimuli'.format(len(df_pred))

    if len(df_gold) != len(df_pred):
        print 'error: different number of stimuli in gold and predicted sets '\
            '({} vs {})'.format(len(df_gold), len(df_pred))
        exit()

    intervals_gold = sorted([(row.filename, row.start)
                             for _, row in df_gold.iterrows()])
    intervals_pred = sorted([(row.filename, row.start)
                             for _, row in df_pred.iterrows()])
    if intervals_gold != intervals_pred:
        print 'error: different intervals in gold and predicted sets'
        exit()

    callset_gold = set(df_gold.label.unique())
    callset_pred = set(df_pred.label.unique())
    if len(callset_pred - callset_gold) > 0:
        print 'error: call types in predicted set that are not in gold set'
        exit()

    labels = sorted(callset_gold)
    label2ix = {label: ix for ix, label in enumerate(labels)}
    y_gold = np.array([label2ix[call] for call in df_gold.label])
    y_pred = np.array([label2ix[call] for call in df_pred.label])

    print
    print classification_report(y_gold, y_pred, target_names=labels)
    print
    print pretty_cm(confusion_matrix(y_gold, y_pred), labels=labels,
                    offset=' '*8)
