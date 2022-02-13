from sklearn.datasets import fetch_openml

iris_data, iris_target = fetch_openml(
  name='iris',
  version='1',
  return_X_y = True,
  as_frame=True
)

data = iris_data.values
