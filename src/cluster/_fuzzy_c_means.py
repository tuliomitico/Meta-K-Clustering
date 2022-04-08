"""Fuzzy-c-means clustering"""

# Authors: Túlio de Freitas Castro <tcastro@dcomp.ufs.br>

import warnings

import numpy as np

from sklearn.base import BaseEstimator,ClusterMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import pairwise_distances, euclidean_distances

from ..utils import check_random_state, FLOAT_DTYPES

def _distance(X, centers, metric):

    return pairwise_distances(X, centers, metric = metric)

def _initialize_random(X, random_state, n_clusters, eps=1e-12):

    n_samples = X.shape[0]
    seed = check_random_state(random_state).permutation(n_samples)[:n_clusters]
    selection = X[seed] + eps
    distances = euclidean_distances(X, selection)
    normalized_distance = distances / np.sum(distances,axis =1)[:,np.newaxis]
    return  1 - normalized_distance, selection

def _initialize_alt_random(X, random_state, n_clusters, eps = 1e-12):

    n_samples = X.shape[0]
    seed = check_random_state(random_state).choice(n_samples,n_clusters,replace=False)
    selection = X[seed] + eps
    distances = euclidean_distances(X, selection)
    normalized_distance = distances / np.sum(distances,axis = 1)[:,np.newaxis]
    return 1 - normalized_distance, selection

def _initialize_heuristic(X, n_clusters, eps = 1e-12):

    n_samples = X.shape[0]
    seed = np.argpartition(np.sum(X,axis=1),n_clusters - 1)[:n_clusters]
    selection = X[seed] + eps
    distances = euclidean_distances(X, selection)
    normalized_distance = distances / np.sum(distances,axis = 1)[:,np.newaxis]
    return 1 - normalized_distance, selection

def _initialize(X, init, random_state, n_clusters,
                centers, memberships):

    if init == 'random':

        initialization = _initialize_random(X, random_state, n_clusters)

        if centers is None and memberships is None:

            memberships,centers = initialization

        elif memberships is None:

            memberships = initialization[0]

        elif centers is None:

            centers = initialization[1]

    if init == 'alt_random':

        initialization = _initialize_alt_random(X, random_state, n_clusters)

        if centers is None and memberships is None:

            memberships,centers = initialization

        elif memberships is None:

            memberships = initialization[0]

        elif centers is None:

            centers = initialization[1]

    if init == 'heuristic':

        initialization = _initialize_heuristic(X, n_clusters)

        if centers is None and memberships is None:

            memberships,centers = initialization

        elif memberships is None:

            memberships = initialization[0]

        elif centers is None:

            centers = initialization[1]
    return centers, memberships

def _fuzzifier(memberships,m):

    return np.power(memberships, m)

def _objective(X, centers, m, metric, memberships):

    if memberships is None or centers is None:
        return np.infty

    distances = _distance(X,centers,metric)
    return np.sum(_fuzzifier(memberships,m) * distances)

def _calculate_memberships(X, m, metric, centers):

    distances = _distance(X, centers, metric)
    distances[distances == .0] = 1e-18

    return np.sum(np.power(
        np.divide(distances[:,:,np.newaxis],distances[:,np.newaxis,:]),
        2 / (m-1)), axis = 2) ** -1


def _calculate_centers(X, memberships, m):

    return np.dot(_fuzzifier(memberships,m).T, X) / np.sum(_fuzzifier(memberships,m).T,axis=1)[:,np.newaxis]

def _update(X, init, m, metric,
            random_state, n_clusters, memberships,
            centers):

    if centers is None and memberships is None:
        centers, memberships = _initialize(X, init, random_state, n_clusters, centers, memberships)

    memberships = _calculate_memberships(X, m, metric, centers)
    centers = _calculate_centers(X, memberships, m)
    return centers, memberships

def _fcmeans(X, init, m, max_iter,
            metric, random_state, tol,
            n_clusters, verbose, centers, memberships):

    j_new = np.infty
    for i in range(max_iter):

        if verbose:
            print("Iteration {I} objective {J}".format(I=i,J=j_new))

        j_old = j_new
        centers, memberships = _update(X, init, m, metric, random_state, n_clusters, memberships, centers)
        labels = np.argmax(memberships,axis = 1)
        j_new = _objective(X, centers, m, metric, memberships)
        if np.abs(j_old - j_new) < tol:
            break
        elif i == max_iter - 1:
            warnings.warn(
                "Maximum number of iteration reached before"
                "convergence. Consider increase the number of iteration.",
            ConvergenceWarning)

    return memberships, centers, labels, j_new

class FCM(BaseEstimator, ClusterMixin):
    """Fuzzy-C-Means clustering.

    Read more in the :ref:`User Guide fuzzy c means`

    Parameters
    ----------

    n_clusters : int, default = 8
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, default = 300
        Maximun number of iterations of the fuzzy-c-means

    metric : str or callable, default = 'euclidean'
        Metric used to compute the inertia. Can be some metrics
        existing in pairwise_kernel function. However only "euclidean",
        "manhattan", "cosine", "correlation", "sqeuclidean","canberra", "haversine"
        were test. Be careful no exception was created to catch the possibles errors.

    m : float, default = 2
        Value used to determine when a points belong to a membership or not.

    n_init : int, default = 10
        Number of time the fuzzy-c-means will be run. The
        final results will be the best output of n_init consecutive
        runs in terms of objective.

    tol : float, default = 1e-4
        Relative tolerance with regards to objective to declare convergence

    eps : float, default = 1e-18
        Parameter used to replace any zeros present in the dataset to avoid
        the ZeroDivision exception

    random_state : int, RandomState instance, default = None
        Determines random number generation for centroid initialization. Use
        an int to makes randomness deterministic.
        See :term:`Glossary <random_state>`.

    verbose : int, default = 0
        Verbosity mode.

    init : str or callable, default = 'random'

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples)
        Labels of each point.

    objective_ :

    cluster_centers_ :
        Coordinates of cluster centers.

    memberships_ :

    Notes
    -----
    ..[1] J. C. Bezdek, J. Keller, R. Krisnapuram, N. R. Pal , 1999. "Fuzzy model and algorithms for
    pattern recognition and image processing". Springer Science.
    """

    def __init__(self, n_clusters = 8, max_iter = 300 , metric = 'euclidean', m = 2,
                 n_init = 10,tol = 1e-4, eps = 1e-18, random_state = None,
                 verbose = 0, init = 'random'):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.metric = metric
        self.m = m
        self.n_init = n_init
        self.init = init
        self.tol = tol
        self.eps = eps
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y = None):
        """Compute fuzzy-c-means clustering.

        Parameters
        ----------
        X : array or sparse-matrix,
            Training instances to cluster.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self
            Fitted estimator.
        """
        random_state = check_random_state(self.random_state)

        n_init = self.n_init
        if n_init <= 0:
            raise ValueError('Invalid number of initializations.'
                             ' n_init=%d must be bigger than zero.' %n_init)

        if self.max_iter <= 0:
            raise ValueError(
                "Number of iterations should be a positive number,"
                ' got %d instead' % self.max_iter
            )

        X = check_array(X,accept_sparse=['csr'],dtype=FLOAT_DTYPES)

        init = self.init
        if init == 'heuristic' or hasattr(init,'__array__'):
            if n_init != 1:
                warnings.warn(
                    'Heuristic mode is a deterministic initialization: '
                    'performing only one init in fuzzy-c-means instead of n_init=%d'
                    % n_init,RuntimeWarning, stacklevel=2)
                n_init = 1

        num_samples = X.shape[0]

        best_objective, best_centers, best_labels, best_memberships = np.infty, None, None, None

        iterator = range(n_init)
        for i in iterator:
            centers = None
            memberships = None
            memberships, centers, labels, objective = _fcmeans(X, self.init, self.m, self.max_iter,self.metric, random_state,
                                                               self.tol, self.n_clusters, self.verbose, centers, memberships
                                                               )
            if objective < best_objective:
                best_memberships = memberships.copy()
                best_centers = centers.copy()
                best_labels = labels
                best_objective = objective

        self.cluster_centers_ = best_centers
        self.memberships_ = best_memberships
        self.labels_ = best_labels
        self.objective_ = best_objective

        return self

    def predict(self, X):

        check_is_fitted(self)
        labels = _calculate_memberships(X, self.m, self.metric, self.cluster_centers_)
        return np.argmax(labels,axis=1)

    def fit_predict(self, X, y=None):

        return self.fit(X).labels_

def _calculate_covariance(X, centers, m, memberships):

    v = centers

    if v is None:
        return None
    d = X - v[:, np.newaxis]
    fuzzy_memberships = _fuzzifier(memberships, m)
    numerator = np.einsum('k...,...ki,...kj->...ij',fuzzy_memberships,d,d)
    denominator = np.sum(fuzzy_memberships, axis = 0) [..., np.newaxis, np.newaxis]
    return numerator / denominator

def _distance_GKFCM(X, centers, covariance,m,memberships):

    v = centers
    if v is None:
        return None
    q,p = v.shape
    covariance = covariance if covariance is not None else _calculate_covariance(X,centers,m,memberships)
    d = X - v[:, np.newaxis]
    A = (np.linalg.det(covariance) ** (1/p))[...,np.newaxis,np.newaxis] * np.linalg.inv(covariance)
    return np.einsum('...ki,...ij,...kj->...k',d,A,d).T ** 0.5

def _objective_GKFCM(X, centers, m, covariance, memberships):

    if memberships is None or centers is None:
        return np.infty

    distances = _distance_GKFCM(X,centers,covariance,m,memberships)
    return np.sum(_fuzzifier(memberships,m) * distances)

def _calculate_memberships_GKFCM(X, m, covariance, centers,memberships):

    distances = _distance_GKFCM(X, centers, covariance,m,memberships)
    distances[distances == .0] = 1e-18

    return np.sum(np.power(
        np.divide(distances[:,:,np.newaxis],distances[:,np.newaxis,:]),
        2 / (m-1)), axis = 2) ** -1



def _update_GKFCM(X, init, m, random_state,
                  n_clusters, memberships, centers, covariance):

    if centers is None and memberships is None:
        centers, memberships = _initialize(X, init, random_state,n_clusters,centers,memberships)

    memberships = _calculate_memberships_GKFCM(X, m, covariance, centers,memberships)
    covariance = _calculate_covariance(X, centers, m, memberships)
    centers = _calculate_centers(X, memberships, m)

    return centers, covariance, memberships

def _fcmeans_GKFCM(X, init, m, max_iter,
                    random_state, tol,
                   n_clusters, verbose, centers,
                   covariance, memberships):

    j_new = np.Infinity

    for i in range(max_iter):

        if verbose:
            print('Iteration {I} objective {J}'.format(I=i,J=j_new))

        j_old = j_new
        centers, covariance, memberships = _update_GKFCM(
            X,init,m,
            random_state,n_clusters,memberships,
            centers, covariance
            )
        labels = np.argmax(memberships,axis=1)
        j_new = _objective_GKFCM(X, centers, m, covariance, memberships)
        if np.abs(j_old - j_new) < tol:
            break

    return memberships, labels, centers, j_new ,covariance

class GustafsonKesselFCM(BaseEstimator, ClusterMixin):

    #covariance = None

    def __init__(self, n_clusters = 8, max_iter = 300, m = 2,
                 n_init = 10, tol = 1e-4, eps = 1e-18, random_state = None,
                verbose = 0, init = 'random'):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        #self.metric = metric
        self.m = m
        self.n_init = n_init
        self.tol = tol
        self.eps = eps
        self.random_state = random_state
        self.verbose = verbose
        self.init = init

    def fit(self, X, y = None):

        random_state = check_random_state(self.random_state)

        if self.max_iter <= 0:
            raise ValueError('Tudo errado')

        X = check_array(X,accept_sparse=['csr'],dtype=FLOAT_DTYPES)

        num_samples = X.shape[0]

        best_objective, best_centers, best_labels, best_memberships, best_covariance = np.infty, None, None, None, None

        iterator = range(self.n_init)
        for i in iterator:

            centers = None
            covariance = None
            memberships = None

            memberships, labels, centers, objective, covariance = _fcmeans_GKFCM(X, self.init, self.m, self.max_iter, random_state,
                                                                                self.tol, self.n_clusters, self.verbose, centers, covariance, memberships
                                                                                )

            if objective < best_objective:
                best_memberships = memberships.copy()
                best_centers = centers.copy()
                best_labels = labels
                best_objective = objective
                best_covariance = covariance

        self.cluster_centers_ = best_centers
        self.memberships_ = best_memberships
        self.labels_ = best_labels
        self.objective_ = best_objective
        self.covariance_ = best_covariance

        return self

    def predict(self, X):

        check_is_fitted(self)
        labels = _calculate_memberships_GKFCM(X, self.m, self.covariance_, self.cluster_centers_, self.memberships_)
        return np.argmax(labels, axis=1)

    def fit_predict(self, X, y=None):

        return self.fit(X).labels_
