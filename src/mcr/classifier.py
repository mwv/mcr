"""classifier: exact and approximate sparse kernel classifiers

"""
from __future__ import division

from sklearn.kernel_approximation import Nystroem
from sklearn.metrics.pairwise import pairwise_kernels
from lightning.classification import CDClassifier
from sklearn.svm import LinearSVC


class SparseKernelClassifier(CDClassifier):
    def __init__(self, mode='exact', kernel='rbf', gamma=1e-3, C=1, alpha=1,
                 n_components=500, n_jobs=1, verbose=False):
        self.mode = mode
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.alpha = alpha
        self.n_components = n_components
        self.n_jobs = n_jobs
        self.verbose = verbose
        super(SparseKernelClassifier, self).__init__(
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
        if self.mode == 'exact':
            K = pairwise_kernels(
                X,
                metric=self.kernel,
                filter_params=True,
                gamma=self.gamma
            )
            self.X_train_ = X
        else:
            self.kernel_sampler_ = Nystroem(
                kernel=self.kernel,
                gamma=self.gamma,
                n_components=self.n_components
            )
            K = self.kernel_sampler_.fit_transform(X)
        super(SparseKernelClassifier, self).fit(K, y)
        return self

    def decision_function(self, X):
        if self.mode == 'exact':
            K = pairwise_kernels(
                X, self.X_train_,
                metric=self.kernel,
                filter_params=True,
                gamma=self.gamma
            )
        else:
            K = self.kernel_sampler_.transform(X)
        return super(SparseKernelClassifier, self).decision_function(K)


class WeightedSparseKernelClassifier(LinearSVC):
    def __init__(
            self, mode='exact', kernel='rbf', gamma=1e-3, C=1,
            multi_class='ovr', class_weight='auto', n_components=5000,
            verbose=False
    ):
        self.mode = mode
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.multi_class = multi_class
        self.class_weight = class_weight
        self.n_components = n_components
        self.verbose = verbose

        super(WeightedSparseKernelClassifier, self).__init__(
            C=C,
            loss='squared_hinge',
            penalty='l1',
            dual=False,
            verbose=verbose
        )

    def fit(self, X, y):
        if self.mode == 'exact':
            K = pairwise_kernels(
                X,
                metric=self.kernel,
                filter_params=True,
                gamma=self.gamma
            )
            self.X_train_ = X
        else:
            self.kernel_sampler_ = Nystroem(
                kernel=self.kernel,
                gamma=self.gamma,
                n_components=self.n_components
            )
            K = self.kernel_sampler_.fit_transform(X)
        return super(WeightedSparseKernelClassifier, self).fit(K, y)

    def decision_function(self, X):
        if self.mode == 'exact':
            K = pairwise_kernels(
                X, self.X_train_,
                metric=self.kernel,
                filter_params=True,
                gamma=self.gamma
            )
        else:
            K = self.kernel_sampler_.transform(X)
        return super(WeightedSparseKernelClassifier, self).decision_function(K)
