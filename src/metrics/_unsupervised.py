# -*- coding: utf-8 -*-
from typing import Union
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics import euclidean_distances

def wgss_score(
    X: Union[pd.DataFrame,np.ndarray],
    labels: Union[pd.Series,np.ndarray]
) -> float:
    """Compute the total within  group sum of square (WGSS) score.

    Also known as WSS

    Parameters
    ----------
    X : array-like, shape (``n_samples``,``n_features``)
        List of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.

    labels: array-like, shape (``n_samples``,)
        Predicted labels for each samples.

    Returns
    -------
    score: float
        The resulting WGSS score.

    References
    ----------
    """
    num_clusters: int = np.max(labels) + 1
    score = .0

    for i in range(num_clusters):
        cluster_member = X[labels == i]
        cluster_center = cluster_member.mean(axis=0)
        score += np.sum((cluster_member - cluster_center) ** 2)

    return score

def dunn_score(
    X: Union[pd.DataFrame,np.ndarray],
    labels: Union[pd.Series,np.ndarray]
) -> float:
    """Compute the Dunn Score

    Parameters
    ----------
    X : array-like, shape (``n_samples``,``n_features``)
        List of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.

    labels : array-like, shape (``n_samples``,)
        The predicted labels for each samples.

    Returns
    -------
    score : float
        The resulting Dunn index.

    References
    ----------
    ..[1] J. Dunn, 1974. "Well separated clusters and optimal fuzzy partitions".
          Journal of Cybernetics, 4:95–104.
    """

    num_cluster = np.max(labels) + 1
    diam = []
    l_min_dis = []

    for i in range(num_cluster-1):
        cluster_member = X[labels == i]
        intra_dist = distance.pdist(cluster_member)
        diam.append(np.max(intra_dist) if intra_dist.size else np.asarray([0],dtype='uint8'))
        for j in range(i+1,num_cluster):
            cluster_member2 = X[labels == j]
            diameter = distance.pdist(cluster_member2)
            if len(diameter) == 0:
                warnings.warn(
                    'Cannot calculate dunn_score, due to an undefined value',
                    UserWarning
                )
                score = 0
                return score
            diam.append(np.max(diameter))
            pair_dis = euclidean_distances(cluster_member,cluster_member2)
            min_dis = np.min(np.concatenate(pair_dis))
            l_min_dis.append(min_dis)


    score = min(l_min_dis)/ max(diam)
    return score

def sd_dis_score(
    X: Union[pd.DataFrame,np.ndarray],
    labels: Union[pd.Series,np.ndarray]
) -> float:
    """The SD Dis index, a measure of measure

    Parameters
    ----------
    X : array-like, shape (``n_samples``, ``n_features``)
        List of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.

    labels : array-like, shape (``n_samples``,)
        The predicted labels for each sampe.

    Returns
    -------
    score: float
        The resulting SD Dis score.

    References
    ----------
    .. [1] M. Halkidi, 2001. "On clustering validation techniques."
    """
    num_cluster = np.max(labels) + 1
    list_centers = list()
    nom_sum = 0

    for i in range(num_cluster):
        nsum = 0
        cluster_member = X[labels == i]
        cluster_center = cluster_member.mean(0)
        list_centers.append(cluster_center)

        for j in range(num_cluster):
            cluster_member2 = X[labels == j]
            cluster_center2 = cluster_member2.mean(0)
            nsum += np.linalg.norm(cluster_center - cluster_center2)
        nom_sum += 1 / nsum

    dmax = max(distance.pdist(list_centers))
    dmin = min(distance.pdist(list_centers))

    score = nom_sum * dmax / dmin

    return score
