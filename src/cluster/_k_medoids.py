"""K-medoids clustering"""

# Authors: Túlio de Freitas Castro <tfcastro@dcomp.ufs.br>
# License: BSD 3 Clause

import warnings

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import (
  pairwise_distances,
  pairwise_distances_argmin
)
from sklearn.utils import check_array
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted
from sklearn.exceptions import ConvergenceWarning

from ..utils import check_random_state

#################################################################
# Initialization heuristic

def check_is_in_X(D, medoids):
    indices = list()
    for i in medoids:
        k = np.flatnonzero((D == i).all(1))
        if k.size == 0:
            raise ValueError("Expected at one index, but got %s instead." %(k.size))
        indices.append(k[0])
    return indices

def _kpp_init(D,n_clusters,random_state,n_local_trials = None):
    """Init n_clusters seeds with a method similar to k-means++

    Paramenters
    -----------
    D : array or sparse matrix, shape (n_samples, n_features)
        The distance matrix we will use to select medoid indices

    n_clusters : int
        The number of seeds to choose.

    random_state : RandomState instance
        The generator used to initialize the centers.

    n_local_trials : int, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Notes
    -----
    Selects initial cluster centers for k-medoid clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007

    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.

    Returns
    -------
    centers : array of shape (n_features,n_cluster)
        The initial cluster centers.
    """
    n_samples,_ = D.shape
    centers = np.empty(n_clusters,int)

    if n_local_trials is None:

        n_local_trials = 2 + int(np.log(n_clusters))

    center_id = random_state.randint(n_samples)
    centers[0] = center_id

    closest_dist_sq = D[centers[0], :] ** 2
    current_pot = closest_dist_sq.sum()

    for cluster_index in range(1,n_clusters):
        rand_vals = (
            random_state.random_sample(n_local_trials) * current_pot
        )
        candidate_ids = np.searchsorted(
            stable_cumsum(closest_dist_sq), rand_vals
        )

        distance_to_candidates = D[candidate_ids,:] ** 2

        best_candidate,best_pot,best_dist_sq = None, None, None
        for trial in range(n_local_trials):
            new_dist_sq = np.minimum(
                closest_dist_sq,distance_to_candidates[trial]
            )
            new_pot = new_dist_sq.sum()

            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        centers[cluster_index] = best_candidate
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return centers

def _init_medoids(X,D,init,n_clusters : int, random_state):
    """Select initial medoids when begins the clustering"""

    n_samples = X.shape[0]
    random_state = check_random_state(random_state)
    if isinstance(init,str) and init == 'random':
        medoids = random_state.choice(n_samples,n_clusters,replace = False)
    elif isinstance(init,str) and init == 'heuristic':
        medoids = np.argpartition(np.sum(X,axis=1),n_clusters - 1)[:n_clusters]
    elif isinstance(init,str) and init == 'alt_random':
        medoids = random_state.permutation(n_samples)[:n_clusters]
    elif isinstance(init,str) and init == 'kmedoids++':
        medoids = _kpp_init(X,n_clusters,random_state)
    elif hasattr(init,"__array__"):
        medoids = np.array(init, dtype = X.dtype)
        medoids = check_is_in_X(D, medoids)
    elif callable(init):
        medoids = init(X,n_clusters, random_state = random_state)
        medoids = np.asarray(medoids,dtype = X.dtype)
    else:
        raise ValueError("the init parameter for the k-medoids should "
                         "be 'kmedoids++','random','alt_random','heuristic' or an ndarray, "
                         "'%s' (type '%s') was passed." % (init, type(init)))

    return medoids

def _compute_inertia(distances):
    """Compute inertia of new samples. Inertia is defined as the sum of the
        sample distances to closest cluster centers.

        Parameters
        ----------
        distances : {array-like, sparse matrix}, shape=(n_samples, n_clusters)
            Distances to cluster centers.

        Returns
        -------
        Sum of sample distances to closest cluster centers.
        """

    inertia = np.sum(np.min(distances,axis= 1))
    return inertia

def _update_medoids_idx_in_place(D,n_clusters,labels,medoids_index):
    """Update the medoid indices in place
    """
    for k in range(n_clusters):

        cluster_k_idx = np.where(labels == k)[0]

        if len(cluster_k_idx) == 0:
            warnings.warn("Cluster {k} is empty! "
                    "self.labels_[self.medoid_indices_[{k}]] "
                    "may not be labeled with "
                    "its corresponding cluster ({k}).".format(k=k)
                )
            continue

        in_cluster_distances = D[
            cluster_k_idx,cluster_k_idx[:,np.newaxis]
        ]

        in_cluster_all_costs = np.sum(in_cluster_distances,axis = 1)

        min_cost_idx = np.argmin(in_cluster_all_costs)
        min_cost = in_cluster_all_costs[min_cost_idx]
        curr_cost = in_cluster_all_costs[
            np.argmax(cluster_k_idx == medoids_index[k])
        ]

        if min_cost < curr_cost:
            medoids_index[k] = cluster_k_idx[min_cost_idx]

def _pam(X,n_clusters,init,metric,max_iter,random_state,verbose):
    """Partitioning Around Medoids technique.

    Parameters
    ----------
    X : array of shape (n_samples,n_features)
        The data to be clustered.

    n_clusters : int
        The number of cluster to form as well as the number
        of medoids to generate.

    init : {'random','heuristic','kmedoids++'}
        Method of initialization

    random_state : RandomState instance.
        The seed used to initialize the _init_medoids function.

    verbose : int
        Verbosity mode.

    Notes
    -----
    This algorithm still in development in the future will include
    another aproachs, for example CLARA for large datasets.
    For now only the PAM aproach is avaliable.

    Returns
    -------

    centers, labels, medoid_idx, inertia : tuple
    """
    random_state = check_random_state(random_state)
    D = pairwise_distances(X,metric = metric)
    medoid_idx = _init_medoids(D,X,init,n_clusters,random_state)

    labels = None
    for n_iter in range(max_iter):
        if verbose:
            print('Iteration {I}'.format(I=n_iter))
        old_medoids_idx = np.copy(medoid_idx)
        labels = np.argmin(D[medoid_idx,:], axis = 0)
        _update_medoids_idx_in_place(D,n_clusters,labels,medoid_idx)
        if np.all(old_medoids_idx == medoid_idx):
            break
        elif n_iter == max_iter - 1:
                warnings.warn(
                    "Maximum number of iteration reached before "
                    "convergence. Consider increasing max_iter to "
                    "improve the fit.",
                    ConvergenceWarning,
                )

    centers = X[medoid_idx]
    all_distance_metric = pairwise_distances(X,Y=centers)
    inertia = _compute_inertia(all_distance_metric)
    return centers, labels, medoid_idx, inertia

class KMedoids(BaseEstimator,ClusterMixin,TransformerMixin):
    '''K-Medoids clustering.

    Read more in the :ref: `User Guide k-medoids`.

    Parameters
    ----------
    n_clusters : int, default = 8
        The number of clusters to form as well as the number of
        medoids to generate.

    init : {'random','heuristic','kmedoids++','alt_random'}, default = 'random'
        Method for initialization

    metric : default = 'euclidean'

    n_init : int, default = 10

    max_iter : int, default = 300
       Maximum number of iterations of the k-medoids algorithm for
       a single run until converges.

    random_state : int, RandomState instance, default = None
        Determines random number generation for medoid initialization. Use
        an int to make randomness deterministic.
        See :term:`Glossary <random_state>`.

    verbose : int, default = 0
        Verbosity mode.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters,n_features)
        Coordinates of clusters centers. If the algorithm stops before fully
        converging (see ``max_iter``), these will not be consistent with
        ``labels_``.

    inertia_ : float
        Sum of squared distances of samples to their closest clusters centers.

    medoid_idxs_ : ndarray of shape (n_cluster,)
        The indices from the clusters centers.

    labels_: ndarray of shape (n_samples,)
        Labels of each point.

    Examples
    --------

    >>> from kmedoids_1 import KMedoids
    >>> import numpy as np
    >>> >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> kmedoids = KMedoids(n_clusters=2, random_state=0).fit(X)
    >>> kmedoids.labels_
    array([1, 1, 1, 0, 0, 0], dtype=int32)
    >>> kmedoids.predict([[0, 0], [12, 3]])
    array([1, 0], dtype=int32)
    >>> kmedoids.cluster_centers_
    array([[10.,  2.],
           [ 1.,  2.]])
    '''
    def __init__(self, n_clusters = 8,init = 'random',metric = 'euclidean' ,
                n_init = 10,max_iter = 300,random_state = None,verbose = 0 ):

        self.n_clusters = n_clusters
        self.init = init
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

    def fit(self,X,y=None):
        """Compute k-medoids clustering.

        Parameters
        ----------
        X : array or sparse matrix,
            Training instances to cluster.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self
            Fitted estimator
        """
        random_state_ = check_random_state(self.random_state)

        n_init = self.n_init
        if n_init <= 0:
            raise ValueError('Invalid number of initializations.'
                             ' n_init = %d must be bigger than zero.' %n_init)

        if self.max_iter <= 0:
            raise ValueError("Number of iterations should be a positive number,"
                             " got %d instead" %self.max_iter
            )
        X = check_array(X,accept_sparse=['csr'],dtype = FLOAT_DTYPES)

        init = self.init
        if init == 'heuristic' or hasattr(init,'__array__'):
            if n_init != 1:
                warnings.warn(
                    'Heuristic mode is a deterministic initialization: '
                    'performing only one init in k-medoids instead of n_init=%d'
                    %n_init, RuntimeWarning, stacklevel=2)
                n_init = 1

        num_samples = X.shape[0]

        if num_samples < self.n_clusters:
            raise ValueError("n_samples = %d should be >= n_clusters=%d" %(num_samples,self.n_clusters)
            )
        best_centers,best_medoid_idx,best_labels,best_inertia = None, None, None, None
        kmedoids_single = _pam

        seeds = random_state_.randint(np.iinfo(np.int32).max,size = n_init)

        for seed in seeds:
            centers, labels, medoid_idx,inertia = kmedoids_single(X,self.n_clusters,
                self.init,self.metric,self.max_iter,random_state = seed,verbose=self.verbose
            )
            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels = labels
                best_medoid_idx = medoid_idx

        self.inertia_ = best_inertia
        self.cluster_centers_ = best_centers
        self.medoid_idxs_ = best_medoid_idx
        self.labels_ = best_labels

        return self

    def predict(self, X):
        """Predict the closest cluster for each sample in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_query, n_features),
                or (n_query, n_indexed) if metric == 'precomputed'
            New data to predict.

        Returns
        -------
        labels : array, shape = (n_query,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)
        X = check_array(X,accept_sparse=['csr'],dtype=FLOAT_DTYPES)
        return pairwise_distances_argmin(X,Y=self.cluster_centers_)

    def fit_predict(self,X,y=None):
        return self.fit(X).labels_

    def fit_transform(self,X):
        return self.fit(X)._transform(X)

    def _transform(self,X):
        return pairwise_distances(X,self.cluster_centers_,metric=self.n_init)

    def transform(self,X):

        check_is_fitted(self)
        X = check_array(X,accept_sparse=['csr'], dtype = FLOAT_DTYPES)
        return self._transform(X)

def k_medoids(X, n_clusters, init = 'kmedoids++', metric = 'euclidean',
              n_init = 10, max_iter = 300, random_state = None,
              return_idx = False):

    est = KMedoids(
        n_clusters = n_clusters, init = init, metric = metric,
        n_init = n_init, max_iter = max_iter, random_state = random_state
        ).fit(X)
    if return_idx:
        return est.cluster_centers_, est.labels_, est.inertia_, est.medoid_idxs_
    else:
        return est.cluster_centers_, est.labels_, est.inertia_
