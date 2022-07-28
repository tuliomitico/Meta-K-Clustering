from collections import defaultdict
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.metrics import (
  accuracy_score, confusion_matrix, precision_score, recall_score
)
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


def evaluate_metamodel(
  X: pd.DataFrame, y: pd.DataFrame, repeats: Integral = 30
) -> """tuple[
  defaultdict[str,list],defaultdict[str,list],
  defaultdict[str,list],defaultdict[str,list]
]""":
  """Evaluate a meta classifier through a repeatead 10-fold

  Parameters
  ----------
  X: array-like, shape (``n_samples``,``n_features``)
    List of ``n_features``-dimensional data points. Each row corresponds
    to a single data point.

  y: array-like, shape ()
  repeats: int, optional (default=30)

  Returns
  -------
  scores: defaultdict[str,list]
    The mean and std from the cross validation.
  """
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

    model.fit(X_train, encoded_y_train.astype('double'))

    y_pred = model.predict(X_test)

    pred = y_pred.ravel()
    true = encoded_y_test.astype('double').ravel()

    score_acc = accuracy_score(true, pred)
    scores_acc.append(score_acc)

    score_precision = precision_score(
      true, pred, average='micro', zero_division='warn'
    )
    scores_precision.append(score_precision)

    score_recall = recall_score(
      true, pred, average='micro', zero_division='warn'
    )
    scores_recall.append(score_recall)

    score_cm = confusion_matrix(true, pred)
    scores_cm.append(score_cm)

  for j in range(0,cv.get_n_splits(),10):
    aux = j + 10
    if aux <= cv.get_n_splits():
      if aux % 10 == 0:
        table_acc["Accuracy_mean"]
        table_acc["Accuracy_std"]
        table_precision["Precision_mean"]
        table_precision["Precision_std"]
        table_recall["Recall_mean"]
        table_recall["Recall_std"]
        table_acc["_mean"]
  return table_acc, table_precision, table_recall, table_cm
