#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: classifier.py
# date: Fri June 19 11:19 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""classifier:

"""

from __future__ import division

from sklearn.kernel_approximation import Nystroem
from sklearn.metrics.pairwise import pairwise_kernels
from lightning.classification import CDClassifier


class SparseApproximateKernelClassifier(CDClassifier):
    def __init__(self, kernel='rbf', gamma=1e-3, C=1, alpha=1,
                 n_components=500, n_jobs=1, verbose=False):
        self.kernel = kernel
        self.gamma = gamma
        self.n_components = n_components
        self.C = C
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.kernel_sampler = Nystroem(
            kernel=kernel,
            gamma=gamma,
            n_components=n_components
        )
        super(SparseApproximateKernelClassifier, self).__init__(
            C=C,
            alpha=alpha,
            loss='squared_hinge',
            penalty='l1',
            multiclass=False,
            debiasing=True,
            Cd=C,
            warm_debiasing=True,
            n_jobs=n_jobs,
            verbose=False,
        )

    def fit(self, X, y):
        X_ = self.kernel_sampler.fit_transform(X)
        super(SparseApproximateKernelClassifier, self).fit(
            X_, y
        )
        return self

    def decision_function(self, X):
        X_ = self.kernel_sampler.transform(X)
        return super(
            SparseApproximateKernelClassifier, self
        ).decision_function(X_)

class SparseExactKernelClassifier(CDClassifier):
    def __init__(self, kernel='rbf', gamma=1e-3, C=1, alpha=1,
                 n_jobs=1, verbose=False):
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.verbose = verbose
        super(SparseExactKernelClassifier, self).__init__(
            C=C,
            alpha=alpha,
            loss='squared_hinge',
            penalty='l1',
            multiclass=False,
            debiasing=True,
            Cd=C,
            warm_debiasing=True,
            n_jobs=n_jobs,
            verbose=False
        )

    def fit(self, X, y):
        K = pairwise_kernels(
            X,
            metric=self.kernel,
            filter_params=True,
            gamma=self.gamma
        )
        self.X_train_ = X
        super(SparseExactKernelClassifier, self).fit(K, y)
        return self

    def decision_function(self, X):
        K = pairwise_kernels(
            X, self.X_train_,
            metric=self.kernel,
            filter_params=True,
            gamma=self.gamma
        )
        return super(SparseExactKernelClassifier, self).decision_function(K)
