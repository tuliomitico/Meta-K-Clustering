import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.utils import check_array

from ..utils.dataset_ids import DATASETS

try:
  from typing import TypedDict
except ImportError:
  from typing_extensions import TypedDict

class Details(TypedDict):
  """A class to mapping the return of a fetch in OpenML site.
  """
  name: str
  version: str

class OpenMLData(NamedTuple):
  """A class to mapping a dataset from OpenMl site.
  """
  data: pd.DataFrame
  target: pd.Series
  details: Details

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def make_dataset(id_list: 'list[int]' = DATASETS) -> None:
  filepath = Path(BASE_DIR / "data/raw/")
  filepath.parent.parent.mkdir(parents=True,exist_ok=True)
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
    break
  return

def number_bins_fn(list_files_length: 'list[int]') -> 'np.int64':
  list_files_length = check_array(list_files_length, ensure_2d = False)
  return np.ceil(np.sqrt(list_files_length.mean()))


def read_raw_datasets(verbose: bool = False) -> 'list[int]':
  """Read from the specified path all CSV datasets and append
  your row length to a list.

  Parameters
  ----------
  verbose : bool, optional
      To see the files being loaded set this to True, by default False

  Returns
  -------
  filelist : array-like
      A array containing the length of datasets.
  """
  file_list = []
  for dirname,_,filenames in os.walk(BASE_DIR / 'data/raw'):
    for filename in filenames:
      if verbose:
        print(Path(os.path.join(dirname,filename)).stem)
      dataset = pd.read_csv(os.path.join(dirname, filename))
      n, _ = dataset.shape
      file_list.append(n)
  return file_list

