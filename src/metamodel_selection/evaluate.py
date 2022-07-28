from collections import defaultdict
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


def evaluate_metamodel(X: pd.DataFrame, y: pd.DataFrame, repeats: Integral):
  random_state = 81
  cv = RepeatedKFold(n_splits=10,n_repeats=repeats,random_state=random_state)
  model = KNeighborsClassifier(n_neighbors=10,n_jobs=4)
  encoder = LabelEncoder()

  scores_acc = []
  scores_precision = []
  scores_recall = []
  scores_cm = []

  table_acc = defaultdict(list)
  table_precision = defaultdict(list)
  table_recall = defaultdict(list)
  table_cm = defaultdict(list)

  for i, (train, test) in enumerate(cv.split(X,y)):
    X_train, X_test = X.values[train], X.values[test]
    y_train, y_test = y.values[train], y.values[test]

    encoded_y_train = np.empty(y_train.shape,dtype='object')
    encoded_y_test = np.empty(y_test.shape,dtype='object')

    encoded_y_train[:,1] = y_train[:,1]
    encoded_y_test[:,1] = y_test[:,1]
    encoded_y_train[:,0] = encoder.fit_transform(y_train[:,0])
    encoded_y_test[:,0] = encoder.transform(y_test[:,0])

    model.fit(X_train, y_train)

  return table_acc, table_precision, table_recall, table_cm
