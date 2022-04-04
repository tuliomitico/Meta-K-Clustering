import sys
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import ClusterMixin
from sklearn.metrics.cluster import silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_array

from ..metrics import dunn_score, sd_dis_score, wgss_score

try:
  from typing import Literal
except ImportError:
  from typing_extensions import Literal

def get_n_groups_information_criteria(
  dataset: Union[pd.DataFrame,np.ndarray],
  method: Literal['aic','bic'],
  min_nc: int,
  max_nc: int
):
  """Gets the number of groups through the Gaussian Mixture model.

  Parameters
  ----------
  dataset : array
      The data to be computed the n clusters within.

  method : {'aic','bic'}
      The kind of information criteria, bayesian or akaike.

  min_nc : int
      The minimum number of cluster to be calculated.

  max_nc : int
      The maximum number of cluster to be calculated in a given range.
  """
  X = check_array(dataset)
  values = []
  for i in range(min_nc,max_nc + 1):
    gm = GaussianMixture(
      n_components = i,
      covariance_type = 'full',
      tol = 0.001,
      reg_covar=0.000001,
      max_iter=100,
      n_init=1,
      random_state=42
    )
    gm.fit(X)
    value = 0
    if method == 'bic':
      value = gm.bic(X)
    elif method == 'aic':
      value = gm.aic(X)
    values.append(value)

  ng = int(values.index(min(values)) + min_nc)
  index = values
  return ng, index


def get_n_groups_elbow_technique(
  dataset: Union[pd.DataFrame,np.ndarray],
  estimator: ClusterMixin,
  min_nc:int,
  max_nc:int,
  **est_args: dict
) -> "tuple[int,list[float]]":
  X = check_array(dataset)
  wgss = []
  for i in range(min_nc, max_nc + 1):
    est = estimator(n_clusters = i, random_state = 42, **est_args)
    est.fit(X)
    y_pred = est.predict(X)
    elbow = wgss_score(X, y_pred)
    wgss.append(elbow)
  distances = []
  p1x = min_nc
  p1y = wgss[0]
  p2x = max_nc
  p2y = wgss[max_nc - min_nc]
  for i in range(max_nc - 1):
    x = min_nc + i
    y = wgss[i]
    x_diff = p2x - p1x
    y_diff = p2y - p1y
    num = abs(y_diff*x - x_diff*y + p2x*p1y - p2y*p1x)
    distances.append(num)
  ng = int(distances.index(max(distances)) + min_nc)
  index = wgss
  return ng, index

def get_n_groups_max_diff(
  dataset: Union[pd.DataFrame,np.ndarray],
  estimator: ClusterMixin,
  method: Literal['dunn','sil'],
  min_nc:int,
  max_nc:int,
  **est_args: dict
) -> "tuple[int,list[float]]":
  X = check_array(dataset)
  values = []
  for i in range(min_nc, max_nc + 1):
    est = estimator(n_clusters = i, random_state = 42, **est_args)
    est.fit(X)
    y_pred = est.predict(X)
    value = 0
    if method == 'dunn':
      value = dunn_score(X, y_pred)
    elif method == 'sil':
      value = silhouette_score(X, y_pred)
    values.append(value)
  ng = int(values.index(max(values)) + min_nc)
  index = values
  return ng, index

def get_n_groups_min_diff(
  dataset: Union[pd.DataFrame, np.ndarray],
  estimator,
  method: Literal['sddis','davies'],
  min_nc: int,
  max_nc: int,
  **est_args: dict
):
  X = check_array(dataset)
  values = []
  for i in range(min_nc, max_nc + 1):
    est = estimator(n_clusters = i, random_state = 42,**est_args)
    est.fit(X)
    y_pred = est.predict(X)
    value = 0
    if method == 'sddis':
      value = sd_dis_score(X,y_pred)
    elif method == 'davies':
      value = davies_bouldin_score(X,y_pred)
    values.append(value)
  ng = int(values.index(min(values)) + min_nc)
  index = values
  return ng, index
