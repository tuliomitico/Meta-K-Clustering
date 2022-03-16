import typing as t
import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

def preprocessing(dataset: pd.DataFrame) -> None:
  """A function to tranform categorical columns in to number columns.

  Parameters
  ----------
  dataset : pd.DataFrame
    The dataset to be transform.
  """
  le = LabelEncoder()
  ds_cols = dataset.columns
  if dataset.isna().any().any():
    imp_most = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
    dataset = imp_most.fit_transform(dataset)
    dataset = pd.DataFrame(dataset, columns = ds_cols)

  X = dataset.select_dtypes(exclude = 'number')
  for i in X.columns:
    dataset[i] = le.fit_transform(dataset[i])

def preprocessed_data_to_csv(path: t.Union[Path, str]) -> None:
  filepath = Path(path).glob('*.csv')
  path_out = Path('./data/interim')
  for file in filepath:
    dataset = pd.read_csv(file)
    preprocessing(dataset = dataset)
    if path_out.exists():
      dataset.to_csv(path_out / file.name, index = False)

def normalize(dataset: pd.DataFrame, scaler = MinMaxScaler) -> pd.DataFrame:
  sscaler = scaler()
  ds_cols = dataset.columns
  scaled_data = sscaler.fit_transform(dataset)
  scaled_dataset = pd.DataFrame(scaled_data, columns = ds_cols)
  return scaled_dataset

def normalize_data_to_csv(path_in: str, path_out: str):
  filepath = Path(path_in).glob('*.csv')
  for file in filepath:
    dataset = pd.read_csv(file)
    scaled_dataset = normalize(dataset)
    if Path(path_out).exists():
      scaled_dataset.to_csv(os.path.join(path_out,file.name), index = False)
