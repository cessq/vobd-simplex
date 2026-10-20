from dataclasses import asdict, dataclass, field

import numpy as np

from .criteria import ranked
from .geometry import irregularity, regular_simplex
from .mappings import MAPPINGS

ALPHA = 2.0  # только при alpha = 2 симплекс после отображения остаётся правильным

STOP_TEXT = {
    "radius": "радиус симплекса меньше r_min",
    "spread": "разброс f по вершинам меньше f_spread",
    "evals": "исчерпан бюджет вычислений f",
    "iters": "достигнут лимит итераций",
}


@dataclass
class Settings:
    criterion: str = "I2"  # "single" — отображение одной наихудшей вершины
    mapping: str = "1"
    fstar: str = "mean"  # "center" — f в центре (+1 вычисление), "mean" — среднее по вершинам
    eps: float = 0.01
    gamma: float = 0.5
    tries: int = 1  # сколько направлений пробовать, прежде чем сжимать
    r_min: float = 1e-6
    f_spread: float = 0.0  # 0 — правило выключено
    max_evals: int = 5000
    max_iters: int = 20000


@dataclass
class Result:
    x: np.ndarray
    f: float
    evals: int
    iters: int
    stop: str
    settings: Settings
    steps: list = field(repr=False)


def fstar(f, V, fv, s):
    return f(V.mean(axis=0)) if s.fstar == "center" else fv.mean()


def num_grad(f, x, h):
    g = np.empty(len(x))
    for i in range(len(x)):
        e = np.zeros(len(x))
        e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def candidates(s, V, fv, fs, grad):
    if s.criterion == "single":
        # сначала наихудшая вершина, при неудаче — следующая по порядку
        for k in range(len(V) - 1):
            order = [k] + [i for i in range(len(V)) if i != k]
            yield 1, 0, V[order], fv[order]
    else:
        for _, m, l in ranked(s.criterion, V, fv, fs, ALPHA, grad):
            yield m, l, V, fv


def stop_reason(s, N, calls, R, fv):
    if R < s.r_min:
        return "radius"
    if s.f_spread and fv[0] - fv[-1] < s.f_spread:
        return "spread"
    if calls >= s.max_evals:
        return "evals"
    if N >= s.max_iters:
        return "iters"
    return None


def minimize(f, x0, R0, s=None, callback=None):
    s = s or Settings()
    V = regular_simplex(x0, R0)
    fv = np.array([f(v) for v in V])
    fs = fstar(f, V, fv, s)
    if not (np.all(np.isfinite(fv)) and np.isfinite(fs)):
        raise ValueError("функция не определена в вершинах начального симплекса — смените x0 или R1")
    R = float(R0)
    steps = []

    def log(N, m, l, ok, tries, cost):
        rec = dict(
            N=N,
            evals=f.calls,
            cost=cost,
            f_star=fs,
            f_best=fv.min(),
            R=R,
            m=m,
            l=l,
            ok=ok,
            tries=tries,
            irregularity=irregularity(V),
            center=V.mean(axis=0),
            V=V.copy(),
        )
        steps.append(rec)
        if callback:
            callback(rec)

    log(0, 0, 0, True, 0, f.calls)
    N = 0
    while True:
        order = np.argsort(-fv, kind="stable")
        V, fv = V[order], fv[order]
        stop = stop_reason(s, N, f.calls, R, fv)
        if stop:
            break
        N += 1
        before = f.calls
        grad = num_grad(f, V.mean(axis=0), R * 1e-2) if s.criterion == "I1grad" else None

        ok = False
        tries = 0
        for m, l, Vp, fvp in candidates(s, V, fv, fs, grad):
            if tries == s.tries:
                break
            tries += 1
            W = MAPPINGS[s.mapping](Vp, m, l, ALPHA)
            fw = fvp.copy()
            fw[: m + l] = [f(w) for w in W[: m + l]]
            fsw = fstar(f, W, fw, s)
            shift = W.mean(axis=0) - V.mean(axis=0)
            if fs - fsw >= s.eps * (shift @ shift):
                ok = True
                break

        if ok:
            V, fv, fs = W, fw, fsw
        else:
            # шаг неуспешный: сжимаем старый симплекс к наилучшей вершине
            best = V[-1]
            V = best + s.gamma * (V - best)
            fv = np.append([f(v) for v in V[:-1]], fv[-1])
            fs = fstar(f, V, fv, s)
            R *= s.gamma
        log(N, m, l, ok, tries, f.calls - before)

    i = int(np.argmin(fv))
    return Result(x=V[i], f=float(fv[i]), evals=f.calls, iters=N, stop=stop, settings=s, steps=steps)


def settings_dict(s):
    return asdict(s)
