from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans

from .vector_measures import get_n_groups_elbow_technique
from .vector_measures import get_n_groups_information_criteria
from .vector_measures import get_n_groups_max_diff
from .vector_measures import get_n_groups_min_diff

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
