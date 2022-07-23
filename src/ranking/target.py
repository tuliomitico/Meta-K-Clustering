from collections import defaultdict
from pathlib import Path
import logging

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import davies_bouldin_score, silhouette_score

from .vector_measures import get_n_groups_elbow_technique
from .vector_measures import get_n_groups_information_criteria
from .vector_measures import get_n_groups_max_diff
from .vector_measures import get_n_groups_min_diff
from ..cluster import FCM
from ..cluster import KMedoids
from ..metrics import dunn_score, sd_dis_score, wgss_score

def generate_target_columns(inp: str, verbose: bool = False) -> pd.DataFrame:
  """Function to generate the multi-target and multi-label columns.

  Parameters
  ----------
  imp : str
      The directory name where the datasets are located.
  verbose : bool, default: False
      Verbose mode.

  Returns
  -------
  target : Dataframe
      The resulting target to meta regression.
  """
  path_in = Path(inp).glob('*.csv')
  dict_metrics = defaultdict(list)

  min_nc = 2
  max_nc = 10

  for file in path_in:
    if verbose:
      print(file.name)
    dataset = pd.read_csv(file)

    dunn,_ = get_n_groups_max_diff(dataset,KMeans,'dunn',min_nc,max_nc)
    dict_metrics['Dunn'].append(dunn)

    sil,_ = get_n_groups_max_diff(dataset,KMeans,'sil',min_nc,max_nc)
    dict_metrics['Silhouette'].append(sil)

    elbow,_ = get_n_groups_elbow_technique(dataset,KMeans,min_nc,max_nc)
    dict_metrics['Elbow'].append(elbow)

    davies,_ = get_n_groups_min_diff(dataset,KMeans,'davies',min_nc,max_nc)
    dict_metrics['Davies'].append(davies)

    sddis,_ = get_n_groups_min_diff(dataset,KMeans,'sddis',min_nc,max_nc)
    dict_metrics['SD-Dis'].append(sddis)

    aic,_ = get_n_groups_information_criteria(dataset,'aic',min_nc,max_nc)
    dict_metrics['AIC'].append(aic)

    bic,_ = get_n_groups_information_criteria(dataset,'bic',min_nc,max_nc)
    dict_metrics['BIC'].append(bic)

  target = pd.DataFrame.from_dict(dict_metrics, orient = 'columns')

  return target

def generate_multi_target_multi_label_column(
  inp: str, verbose: bool = False
) -> pd.DataFrame:
  """Function to generate a multi-label and multi-target columns.

  Parameters
  ----------
  inp : str
      The path of the preprocessed datasets.

  verbose : bool, default = False
      Verbosity mode.

  Returns
  -------
  result : pd.DataFrame
      The result output of the function.
  """
  logging.getLogger(__name__)
  files = Path(inp).glob(r'*.csv')

  files_list = list(files)

  #* minimum n clusters and maximum n clusters
  min_nc, max_nc, step = 2, 16, 2

  #* Cluster algorithms
  fcm = FCM
  km = KMeans
  kmd = KMedoids

  #* Deterministic randomness
  random_state = 42
  fcm_randomness = 81

  df_index_metrics = [
    'AIC','BIC','Elbow','Dunn','Silhouette','Davies-Bouldin','SD-Dis'
  ]
  df_columns = ['K-Means','K-Medoids','Fuzzy C-Means']

  metrics_index = pd.Index(['Elbow','Dunn','Silhouette','Davies','SD-Dis'])

  result_dict = dict()
  for file in files_list:
    if verbose:
      logging.debug('File in process: {}'.format(file.stem))
      print(file.stem)
    try:
      if file.stem.startswith('Airlines'):
        continue
      if file.stem.endswith('neavote_2'):
        continue

      dataset = pd.read_csv(filepath_or_buffer=file)

      #* AIC and BIC metrics
      aic_ng, aic_index = get_n_groups_information_criteria(
        dataset=dataset,
        method='aic',
        min_nc=min_nc,
        max_nc=max_nc,
        step=step,
        random_state=random_state
      )
      bic_ng, bic_index = get_n_groups_information_criteria(
        dataset=dataset,
        method='bic',
        min_nc=min_nc,
        max_nc=max_nc,
        step=step,
        random_state=random_state
      )

      #* Elbow metric
      km_ng, km_index = get_n_groups_elbow_technique(
        dataset=dataset,
        estimator=km,
        min_nc=min_nc,
        max_nc=max_nc,
        step=step,
        random_state=random_state
      )
      kmd_ng, kmd_index = get_n_groups_elbow_technique(
        dataset=dataset,
        estimator=kmd,
        min_nc=min_nc,
        max_nc=max_nc,
        step=step,
        random_state=random_state,
        init='kmedoids++'
      )
      fcm_ng, fcm_index = get_n_groups_elbow_technique(
        dataset=dataset,
        estimator=fcm,
        min_nc=min_nc,
        max_nc=max_nc,
        step=step,
        random_state=fcm_randomness,
        max_iter=350
      )

      #* Dunn and Silhouette metrics
      # Kmeans
      km_dunn_ng, km_dunn_index = get_n_groups_max_diff(
        dataset=dataset,
        estimator=km,
        method='dunn',
        min_nc=min_nc,
        max_nc=max_nc,
        step=step,
        random_state=random_state,
      )
      km_sil_ng, km_sil_index = get_n_groups_max_diff(
        dataset=dataset,
        estimator=km,
        method='sil',
        min_nc=min_nc,
        max_nc=max_nc,
        step=step,
        random_state=random_state,
      )
      # Kmedoids
      kmd_dunn_ng, kmd_dunn_index = get_n_groups_max_diff(
        dataset=dataset,
        estimator=kmd,
        method='dunn',
        min_nc=min_nc,
        max_nc=max_nc,
        step=step,
        random_state=random_state,
        init='kmedoids++'
      )
      kmd_sil_ng, kmd_sil_index = get_n_groups_max_diff(
        dataset=dataset,
        estimator=kmd,
        method='sil',
        min_nc=min_nc,
        max_nc=max_nc,
        step=step,
        random_state=random_state,
        init='kmedoids++'
      )
      # FuzzyCmeans
      fcm_dunn_ng, fcm_dunn_index = get_n_groups_max_diff(
        dataset=dataset,
        estimator=fcm,
        method='dunn',
        min_nc=min_nc,
        max_nc=max_nc,
        step=step,
        random_state=fcm_randomness,
        max_iter=350
      )
      fcm_sil_ng, fcm_sil_index = get_n_groups_max_diff(
        dataset=dataset,
        estimator=fcm,
        method='sil',
        min_nc=min_nc,
        max_nc=max_nc,
        step=step,
        random_state=fcm_randomness,
        max_iter=350
      )

      #* SD-Dis and Davies metrics
      # Kmeans
      km_sddis_ng, km_sddis_index = get_n_groups_min_diff(
        dataset=dataset,
        estimator=km,
        method='sddis',
        min_nc=min_nc,
        max_nc=max_nc,
        step=step,
        random_state=random_state,
      )
      km_davies_ng, km_davies_index = get_n_groups_min_diff(
        dataset=dataset,
        estimator=km,
        method='davies',
        min_nc=min_nc,
        max_nc=max_nc,
        step=step,
        random_state=random_state,
      )
      # Kmedoids
      kmd_sddis_ng, kmd_sddis_index = get_n_groups_min_diff(
        dataset=dataset,
        estimator=kmd,
        method='sddis',
        min_nc=min_nc,
        max_nc=max_nc,
        step=step,
        random_state=random_state,
        init='kmedoids++'
      )
      kmd_davies_ng, kmd_davies_index = get_n_groups_min_diff(
        dataset=dataset,
        estimator=kmd,
        method='davies',
        min_nc=min_nc,
        max_nc=max_nc,
        step=step,
        random_state=random_state,
        init='kmedoids++'
      )
      # FuzzyCmeans
      fcm_sddis_ng, fcm_sddis_index = get_n_groups_min_diff(
        dataset=dataset,
        estimator=fcm,
        method='sddis',
        min_nc=min_nc,
        max_nc=max_nc,
        step=step,
        random_state=random_state,
        max_iter=350
      )
      fcm_davies_ng, fcm_davies_index = get_n_groups_min_diff(
        dataset=dataset,
        estimator=fcm,
        method='davies',
        min_nc=min_nc,
        max_nc=max_nc,
        step=step,
        random_state=random_state,
        max_iter=350
      )


      df = pd.DataFrame(data=list(
          zip([
            aic_ng, bic_ng, km_ng, km_dunn_ng, km_sil_ng, km_davies_ng,
            km_sddis_ng
          ],
          [
            aic_ng, bic_ng, kmd_ng, kmd_dunn_ng, kmd_sil_ng,
            kmd_davies_ng, kmd_sddis_ng
          ],
          [
            aic_ng, bic_ng, fcm_ng, fcm_dunn_ng, fcm_sil_ng,
            fcm_davies_ng, fcm_sddis_ng
          ])),
        index=df_index_metrics,
        columns=df_columns
      )

      kmeans_ser = df['K-Means'].value_counts().rank(
        ascending=True,method='min').astype('uint8')

      kmedoids_ser = df['K-Medoids'].value_counts().rank(
        ascending=True, method='min').astype('uint8')

      fcm_ser = df['Fuzzy C-Means'].value_counts().rank(
        ascending=True, method='min').astype('uint8')

      best_nc = [
        (
          'K-Means',
          {"n_clusters": kmeans_ser.idxmin(),'random_state': 42}
        ),(
          'K-Medoids',
          {"n_clusters": kmedoids_ser.idxmin(),'random_state': 42}
        ),(
          'Fuzzy C-Means',
          {"n_clusters": fcm_ser.idxmin(),'random_state': 81}
        )
      ]

      metrics_ddict = defaultdict(list)

      for i, (k, v) in enumerate(best_nc):
        estimator = None
        if k == 'K-Means':
          estimator = km(**v)
        elif k == 'K-Medoids':
          estimator = kmd(**v)
        else:
          estimator = fcm(**v)

        y_pred = estimator.fit_predict(dataset)

        values = [
          wgss_score(dataset.values,y_pred), dunn_score(dataset, y_pred),
          silhouette_score(dataset, y_pred),
          davies_bouldin_score(dataset, y_pred), sd_dis_score(dataset,y_pred)
        ]


        metrics_ddict[k].extend(values)

      df_metrics = pd.DataFrame.from_dict(metrics_ddict,orient='columns')
      df_metrics.index = metrics_index

      df_final_rank = pd.DataFrame(columns = df_columns, index = metrics_index)
      for i in df_metrics.index.values.tolist():
        if i=='Silhouette' or i == 'Dunn':
          df_final_rank.loc[[i]] = df_metrics.loc[[i]].rank(axis='columns',ascending=False,method='min')
        if i == 'Davies' or i == 'SD-Dis' or i == 'Elbow':
          df_final_rank.loc[[i]] = df_metrics.loc[[i]].rank(axis='columns',ascending=True,method='min')

      x = 0
      for k,v in best_nc:
        if df_final_rank.mean().rank(ascending=True,method='min').idxmin() == k:
          x = v['n_clusters']
      result_dict[file.stem] = [df_final_rank.mean().rank(ascending=True,method='min').idxmin(),x]

    except Exception as e:
      logging.error('Incompatible file: {}'.format(file.stem))
      print("{}: Incompatible file: {}".format(repr(e),file.stem))

  result = pd.DataFrame.from_dict(result_dict, orient='index')
  return result
