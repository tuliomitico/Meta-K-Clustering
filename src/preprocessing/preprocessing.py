import typing as t
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def preprocessing(dataset: pd.DataFrame) -> None:
  """A function to tranform categorical columns in to number columns.

  Parameters
  ----------
  dataset : pd.DataFrame
    The dataset to be transform.
  """
  le = LabelEncoder()
  if dataset.isna().any().any():
    imp_most = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
    dataset = imp_most.fit_transform(dataset)

  X = dataset.select_dtypes(exclude='number')
  for i in X.columns:
    dataset[i] = le.fit_transform(dataset[i])

def preprocessed_data_to_csv(path: t.Union[Path, str]) -> None:
  filepath = Path(path).glob('*.csv')
  path_out = Path('./data/interim')
  for file in filepath:
    dataset = pd.read_csv(file)
    preprocessing(dataset = dataset)
    if path_out.exists():
      dataset.to_csv(path_out / file.name)
