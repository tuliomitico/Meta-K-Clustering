import pandas as pd
from sklearn.datasets import fetch_openml

from .make_dataset import OpenMLData
from ..utils.dataset_ids import DATASETS

def find_datasets_by_n_classes(start, end, id_list = DATASETS):
  """A function to select the datasets with a specified range of n classes,


  Parameters
  ----------
  start : int
      The initial number of classes you want to select.
  end : int
      The last number of classes you want to select.
  id_list : array-like , default = DATASETS
      A list of IDS to be passed to OpenML fetch from Scikit-Learn library

  Returns
  -------
  None
  """
  l = list()
  for i in id_list:
    data: OpenMLData = fetch_openml(data_id = i,as_frame=True)
    try:
      count: int = data.target.nunique()
    except AttributeError as attr_err:
      print(attr_err,i)
    if count in range(start,end):
      print("Name: {0} Version: {1}".format(data.details['name'],data.details['version']))
      l.append(f"{data.details['name']}_{data.details['version']}")
  series = pd.Series(l)
  print(series.shape[0])

