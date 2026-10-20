import numpy as np

from .mappings import delta, directions

# fv — значения f в вершинах по убыванию, fs — f*(x^N).
# Критерий возвращает кандидатов (оценка, m, l); выбирается наибольшая оценка.
# Все критерии, кроме I1 с градиентом, считаются по уже измеренным значениям.


def gap(fv, m, l):
    return fv[:m].mean() - fv[m + l :].mean()


def i1(V, fv, fs, alpha, grad):
    n = len(fv) - 1
    return [(gap(fv, m, l) / np.linalg.norm(delta(V, m, l)), m, l) for m, l in directions(n)]


def i1_grad(V, fv, fs, alpha, grad):
    n = len(fv) - 1
    out = []
    for m, l in directions(n):
        d = delta(V, m, l)
        out.append((-grad @ d / np.linalg.norm(d), m, l))
    return out


def i2(V, fv, fs, alpha, grad):
    # прогноз уменьшения f* по линейной модели; на слайде в знаменателе n+1-m,
    # но из I2 = -Δf и смещения центра alpha*m/(n+1-l)*Δ выходит n+1-l
    n1 = len(fv)
    return [(alpha * m / (n1 - l) * gap(fv, m, l), m, l) for m, l in directions(n1 - 1)]


def i2_slide(V, fv, fs, alpha, grad):
    n1 = len(fv)
    return [(alpha * m / (n1 - m) * gap(fv, m, l), m, l) for m, l in directions(n1 - 1)]


def i3(V, fv, fs, alpha, grad):
    return [(s / (m + l), m, l) for s, m, l in i2(V, fv, fs, alpha, grad)]


def i4(V, fv, fs, alpha, grad):
    return [(np.sum(fv[:m] - fs), m, 0) for m in range(1, len(fv))]


def i5(V, fv, fs, alpha, grad):
    return [(fv[:m].mean(), m, 0) for m in range(1, len(fv))]


def i6(V, fv, fs, alpha, grad):
    phi = (fv[0] + fv[-1]) / 2
    return [(np.sum(fv[:m] - phi), m, 0) for m in range(1, len(fv))]


def i7(V, fv, fs, alpha, grad):
    # на слайде условие для l записано как Ф' <= f <= Ф'', но Ф' > Ф'';
    # читаем как «средняя треть»: Ф'' <= f <= Ф'
    hi = (2 * fv[0] + fv[-1]) / 3
    lo = (fv[0] + 2 * fv[-1]) / 3
    n = len(fv) - 1
    out = []
    for m in range(1, n + 1):
        l = 0
        while m + l < n and lo <= fv[m + l] <= hi:
            l += 1
        out.append((np.sum(fv[:m] - hi), m, l))
    return out


CRITERIA = {
    "I1": i1,
    "I1grad": i1_grad,
    "I2": i2,
    "I2slide": i2_slide,
    "I3": i3,
    "I4": i4,
    "I5": i5,
    "I6": i6,
    "I7": i7,
}


def ranked(name, V, fv, fs, alpha=2.0, grad=None):
    cands = CRITERIA[name](V, fv, fs, alpha, grad)
    # при равных оценках дешевле то направление, где меньше новых вершин
    return sorted(cands, key=lambda c: (-c[0], c[1] + c[2]))
