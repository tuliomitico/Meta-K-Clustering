from pathlib import WindowsPath
import numpy as np
from sklearn.datasets import fetch_openml

from ..make_dataset import BASE_DIR, make_dataset

def test_name_of_data_sets():
  iris = fetch_openml(data_id=61,as_frame=True)
  assert iris.details['name'] == 'iris'
  assert np.unique(iris.target).size == 3

def test_make_dataset():
  assert make_dataset() == None

def test_path():
  assert BASE_DIR / 'data/raw' == WindowsPath('D:/Biblioteca do Tulio/Documents/Faculdade Ufs Si Livros/12º período/TCC 2/Meta-K-Clustering/data/raw')
