import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils import check_array

def phi_fn(u: float) -> float:
  """The formula used to extract meta-features from datasets
  to achieve better prediction measures from a unsupervised
  algorithm.

  $$\phi(u) = \frac{1}{\sqrt{2\pi}}e^{\frac{-u^{2}}{2}}$$

  Parameters
  ----------
  u : float
      The value to be tranform.

  Returns
  -------
  float
      _description_

  References
  ----------
  .. [1] Pimentel B. 2020.
  """
  return np.exp(-u ** 2 / 2) / (np.sqrt(2 * np.pi))


def extract_meta_features(dataset: pd.DataFrame) -> np.ndarray:
  dataset = check_array(dataset)
  n, _ = dataset.shape
  P: np.ndarray = np.zeros(n)
  P_linha = np.zeros(n)
  Q = np.zeros((n,n))
  p = dataset.size
  for r in range(n):
    for s in range(r,n):
      u = np.linalg.norm(dataset[r] - dataset[s]) / 1
      Q[r,s] = phi_fn(u)

  for r in range(n):
    P[r] = (1/n) * np.sum(Q[r]/1)

  minimum, maximum = P.min(), P.max()

  P_linha = (P - minimum) / (maximum - minimum)
  # !70 is the value obtained by the function in make_dataset named number bins
  # !function
  dataset_hist, _ = np.histogram(P_linha,70)

  return dataset_hist

def generate_make_metadataset(path_in: str, path_out: str) -> None:
  filepath = Path(path_in).glob('*.csv')
  raw_dict = {}
  for file in filepath:
    # Unfortunally, my personal computer cannot make such a big array
    # in the order of 1e6x1e6
    if file.name.startswith('Airlines'):
      continue
    dataset = pd.read_csv(file)
    raw_dict[file.stem] = extract_meta_features(dataset)

  metadataset = pd.DataFrame().from_dict(data = raw_dict,orient='index')
  print(metadataset.head(20))
  if Path(path_out).exists():
    metadataset.to_csv(path_or_buf = path_out + "metadataset.csv", index = True)
  return metadataset
