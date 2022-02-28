import numpy as np
from src.data.make_dataset import data

n = len(data)


def phi_fn(u: float) -> float:
    x = 1 / np.sqrt(2 * np.pi)
    return x * np.e ** (-u ** 2 / 2)


def extract_meta_feature(dataset):
    pass


P = np.zeros(n)

P_linha = np.zeros(n)

Q = np.empty((n, n))

p = data.size

for r in range(n):
    for s in range(r, n):
        u = np.linalg.norm(data[r] - data[s]) / 1
        Q[r, s] = phi_fn(u)

for r in range(n):
    P[r] = (1/n) * np.sum((1/p) * Q[r])

minimum, maximum = np.min(P), np.max(P)
P_linha = (P - minimum) / (maximum - minimum)
