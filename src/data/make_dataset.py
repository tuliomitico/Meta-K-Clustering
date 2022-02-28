import os
import typing as t
from pathlib import Path

from sklearn.datasets import fetch_openml
from sklearn.utils import check_array
import numpy as np
import pandas as pd

from . import DATASETS

class Details(t.TypedDict):
  """A class to mapping the return of a fetch in OpenML site.
  """
  name: str
  version: str
class OpenMLData(t.NamedTuple):
  """A class to mapping a dataset from OpenMl site.
  """
  data: pd.DataFrame
  target: pd.Series
  details: Details

BASE_DIR = Path(Path.cwd()).resolve()

def make_dataset(id_list: 'list[int]' = DATASETS) -> None:
  template_name_file = 'data/raw/{name}_{version}.csv'
  for i in id_list:
    data: OpenMLData = fetch_openml(
      data_id=i,
      as_frame=True
    )
    if not Path(BASE_DIR / template_name_file.format(name=data.details['name'],version=data.details['version'])).is_file():
      data.data.to_csv(
        BASE_DIR / template_name_file.format(name=data.details['name'],version=data.details['version']),
        index=False,
        mode='w'
    )
  return

def number_bins_fn(list_files_length: 'list[int]') -> 'np.int64':
  list_files_length = check_array(list_files_length,ensure_2d = False)
  return np.ceil(np.sqrt(list_files_length.mean()))


def read_raw_datasets():
  file_list = []
  for dirname,_,filenames in os.walk(BASE_DIR / 'data/raw'):
    for filename in filenames:
      print(Path(os.path.join(dirname,filename)).stem)
      file_list.append(pd.read_csv(os.path.join(dirname,filename)).shape[0])
  print(number_bins_fn(file_list))

