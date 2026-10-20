import numpy as np


def basis(n):
    # матрица B со слайда: строки — вершины правильного симплекса радиуса 1 с центром в нуле
    a = np.sqrt((n + 1) / (n * np.arange(1, n + 1) * np.arange(2, n + 2)))
    B = np.tile(a, (n + 1, 1))
    for r in range(1, n + 1):
        B[r, : r - 1] = 0
        B[r, r - 1] = -r * a[r - 1]
    return B


def regular_simplex(center, radius):
    center = np.asarray(center, dtype=float)
    return center + radius * basis(len(center))


def edges(V):
    i, j = np.triu_indices(len(V), 1)
    return np.linalg.norm(V[i] - V[j], axis=1)


def irregularity(V):
    # 0 для правильного симплекса: относительный разброс длин рёбер
    e = edges(V)
    return (e.max() - e.min()) / e.mean()


def radius(V):
    return np.linalg.norm(V[0] - V.mean(axis=0))
