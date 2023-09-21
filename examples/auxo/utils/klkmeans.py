from nltk.cluster import KMeansClusterer, euclidean_distance
import numpy as np


class KLKmeans(object):
    def __init__(self,n_clusters, init_center =None ):
        self.labels_ = None

        def _processNegVals(x):
            x = np.array(x)
            minx = np.min(x)
            if minx < 0:
                x = x + abs(minx)
            """ 0.000001 is used here to avoid 0. """
            x = x + 0.000001
            # px = x / np.sum(x)
            return x

        def _KL(P, Q):
            epsilon = 0.00001
            P = _processNegVals(P)
            Q = _processNegVals(Q)
            # You may want to instead make copies to avoid changing the np arrays.
            divergence = np.sum(P * np.log(P / Q))
            return divergence

        self.klkmeans = KMeansClusterer(n_clusters, _KL, initial_means = init_center)

    def fit(self, x):
        print(x)
        self.klkmeans.cluster(x)
        self.cluster_centers_ = self.klkmeans.means()
        self.labels_ = self.predict(x)

    def predict(self, x):
        return [ self.klkmeans.classify(i) for i in x]

