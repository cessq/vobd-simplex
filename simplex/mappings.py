# Вершины V упорядочены по убыванию f: V[0] — наихудшая.
# Отображаются m наихудших вершин, следующие l сдвигаются вместе с центром,
# остальные n+1-m-l остаются на месте.


def delta(V, m, l):
    return V[m + l :].mean(axis=0) - V[:m].mean(axis=0)


def mapping1(V, m, l, alpha=2.0):
    d = delta(V, m, l)
    W = V.copy()
    W[:m] += alpha * d
    W[m : m + l] += alpha * m / (len(V) - l) * d
    return W


def mapping2(V, m, l, alpha=2.0):
    W = V.copy()
    W[:m] += alpha * (V[m + l :].mean(axis=0) - V[:m])
    W[m : m + l] += alpha * m / (len(V) - l) * delta(V, m, l)
    return W


MAPPINGS = {"1": mapping1, "2": mapping2}


def directions(n):
    # все пары (m, l): m = 1..n, l = 0..n-m, всего n(n+1)/2
    return [(m, l) for m in range(1, n + 1) for l in range(0, n - m + 1)]
