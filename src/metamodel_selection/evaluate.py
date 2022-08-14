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
) -> """tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]""":
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
  model = KNeighborsClassifier(n_neighbors=10,n_jobs=6)
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
        table_acc["Accuracy_mean"].append(np.mean(scores_acc[j:aux]))
        table_acc["Accuracy_std"].append(np.std(scores_acc[j:aux]))
        table_precision["Precision_mean"].append(np.mean(scores_precision[j:aux]))
        table_precision["Precision_std"].append(np.std(scores_precision[j:aux]))
        table_recall["Recall_mean"].append(np.mean(scores_recall[j:aux]))
        table_recall["Recall_std"].append(np.std(scores_recall[j:aux]))
        # table_acc["_mean"]

  rounds = pd.Index(["Rodada(s) {}".format(i) for i in range(1,31)])

  accuracy_df = pd.DataFrame(data=table_acc, index=rounds)
  accuracy_df.columns = ['Accuracy mean +/-','Accuracy std +/-']
  accuracy_df.loc['Total'] = [np.mean(scores_acc),np.std(scores_acc)]

  precision_df = pd.DataFrame(data=table_precision, index=rounds)
  precision_df.columns = ['Precision mean +/-','Precision std +/-']
  precision_df.loc['Total'] = [np.mean(scores_precision),np.std(scores_precision)]

  recall_df = pd.DataFrame(data=table_recall, index=rounds)
  recall_df.columns = ['Recall mean +/-','Recall std +/-']
  recall_df.loc['Total'] = [np.mean(scores_recall),np.std(scores_recall)]


  return accuracy_df, precision_df, recall_df
