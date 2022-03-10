import typing as t
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocessing(dataset: pd.DataFrame) -> None:
  """A function to tranform categorical columns in to number columns.

  Parameters
  ----------
  dataset : pd.DataFrame
    The dataset to be transform.
  """
  le = LabelEncoder()
  X = dataset.select_dtypes(exclude='number')
  for i in X.columns:
    dataset[i] = le.fit_transform(dataset[i])

def preprocessed_data_to_csv(path: t.Union[Path, str]) -> None:
  return
