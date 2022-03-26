from typing import Literal, Union
import numpy as np
import pandas as pd
from sklearn.base import ClusterMixin
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import silhouette_score, davies_bouldin_score

from ..metrics import dunn_score, sd_dis_score, wgss_score

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
    gm.fit(dataset)
    value = 0
    if method == 'bic':
      value = gm.bic(dataset)
    elif method == 'aic':
      value = gm.aic(dataset)
    values.append(value)

  ng = int(values.index(min(values)) + min_nc)
  index = values
  return ng, index

def get_n_groups_elbow_technique(
  dataset: Union[pd.DataFrame,np.ndarray],
  estimator: ClusterMixin,
  min_nc:int,
  max_nc:int
) -> "tuple[int,list[float]]":
  wgss = []
  for i in range(min_nc, max_nc + 1):
    est = estimator(n_clusters  = i, random_state = 42)
    est.fit(dataset)
    y_pred = est.predict(dataset)
    elbow = wgss_score(dataset, y_pred)
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

def get_n_groups(
  dataset: Union[pd.DataFrame, np.ndarray],
  estimator,
  method: Literal['sddis','dunn','davies','sil'],
  min_nc: int,
  max_nc: int,
  **est_args: dict
):
  values = []
  for i in range(min_nc, max_nc + 1):
    est = estimator(n_clusters = i, random_state = 42,**est_args)
    est.fit(dataset)
    y_pred = est.predict(dataset)
    value = 0
    if method == 'sddis':
      value = sd_dis_score(dataset,y_pred)
    elif method == 'dunn':
      value = dunn_score(dataset,y_pred)
    elif method == 'davies':
      value = davies_bouldin_score(dataset,y_pred)
    elif method == 'sil':
      value = silhouette_score(dataset,y_pred)
    values.append(value)
  ng = int(values.index(min(values)) + min_nc)
  index = values
  return ng, index
