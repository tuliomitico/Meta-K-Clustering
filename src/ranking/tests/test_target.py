import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler

from ..vector_measures import get_n_groups_max_diff
from ...cluster import FCM
from ...metrics import dunn_score

def test_get_n_groups_max_diff_dunn():
  "Should raise a Exception of type ValueError in wine_1 dataset with Fuzzy C-Means"
  wine_1 = load_wine().data
  mmc = MinMaxScaler()
  wine_scale = mmc.fit_transform(wine_1)
  random_state = 81
  ng, index = get_n_groups_max_diff(wine_scale,FCM,'dunn',min_nc=2, max_nc=16,step=2, random_state=random_state,)
  assert isinstance(wine_1,np.ndarray)
  assert ng == 4

def test_FCM_wine_1():
  wine_1 = load_wine().data
  random_state = 81
  mmc = MinMaxScaler()
  wine_scale = mmc.fit_transform(wine_1)
  fcm = FCM(n_clusters=14,random_state=random_state,verbose=1)
  y_pred = fcm.fit_predict(wine_scale)
  dunn_index = dunn_score(wine_scale,y_pred)
  assert dunn_index == 0
