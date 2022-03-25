from src.data.make_dataset import read_raw_datasets

def test_read_raw_dataset():
  assert len(read_raw_datasets(False)) == 219
