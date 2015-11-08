from __future__ import division

import numpy as np
from spectral import Spectral
from vad import VAD

from util import wavread, roll_array

def extract_features(df, label2ix, spec_kwargs, vad_kwargs,
                     stacksize=1, frate=100, return_y=False):
    if return_y:
        return_y = 'label' in df
    X = {}
    if return_y:
        y = {}
    spectrum_encoder = Spectral(**spec_kwargs)
    vad_encoder = VAD(**vad_kwargs)
    for ix, fname in enumerate(df.filename.unique()):
        sig, fs = wavread(fname)
        if fs != spec_kwargs['fs']:
            raise ValueError('expected samplerate {}, got {}'.format(
                spec_kwargs['fs'], fs)
            )
        spec = spectrum_encoder.transform(sig)
        spec = (spec - spec.mean(0)) / spec.std(0)
        if stacksize > 1:
            spec = roll_array(spec, stacksize)
        vad = vad_encoder.activations(sig)
        vad = vad.reshape(vad.shape[0], -1)
        if stacksize > 1:
            vad = roll_array(vad, stacksize)

        X_curr = []
        if return_y:
            y_curr = []

        rows_iter = df[df.filename == fname].iterrows()
        for _, row in rows_iter:
            start = row.start
            end = row.end

            start_fr = int(start * frate)
            end_fr = int(end * frate)

            feat = np.hstack(
                (spec[start_fr: end_fr],
                 vad[start_fr: end_fr])
            )
            X_curr.append(
                feat.astype(np.float32)
            )
            if return_y:
                y_curr.append(
                    np.ones(feat.shape[0], dtype=np.uint8) * \
                    label2ix[row.label]
                )
        X[fname] = np.vstack(X_curr)
        if return_y:
            y[fname] = np.hstack(y_curr)
    if return_y:
        return X, y
    else:
        return X


def posteriors(clf, X):
    return {
        k: clf.decision_function(v)
        for k, v in X.iteritems()
    }
