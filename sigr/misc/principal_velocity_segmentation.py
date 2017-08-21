'''
Implementation of principal-velocity segmentation in
Coordinative Structure of Manipulative Hand-Movements Facilitates Their Recognition
'''

from __future__ import division
import numpy as np


class PrincipalVelocitySegmentation(object):

    def __init__(self, threshold=0.02):
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=1)
        self.threshold = threshold

    def fit(self, X):
        self.pca.fit(X)
        return self

    def transform(self, X):
        '''Return [begin, end) of the longest static part'''
        p = self.pca.transform(X).ravel()
        v = np.abs(np.diff([p[0]] + list(p)))
        threshold = self.threshold * v.max()
        mask = v < threshold
        breaks = [0] + list(np.where(mask[:-1] != mask[1:])[0] + 1) + [len(mask)]
        begin, end = max([(begin, end) for begin, end
                          in zip(breaks[:-1], breaks[1:]) if mask[begin]],
                         key=lambda r: r[1] - r[0])
        return begin, end

    def fit_transform(self, X):
        return self.fit(X).transform(X)
