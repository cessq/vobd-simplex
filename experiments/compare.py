"""Эксперименты для отчёта. Запуск из корня репозитория:

    python experiments/compare.py

Таблицы пишутся в experiments/results/, рисунки — в experiments/figures/.
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from simplex import PRESETS, Objective, Settings, minimize  # noqa: E402
from simplex.criteria import gap  # noqa: E402
from simplex.functions import sphere  # noqa: E402
from simplex.geometry import irregularity, regular_simplex  # noqa: E402
from simplex.mappings import MAPPINGS, directions  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"
TOL = 1e-6
ALL = ["single", "I1", "I1grad", "I2", "I2slide", "I3", "I4", "I5", "I6", "I7"]

FUNCS = {
    "овраг": PRESETS["Овраг x1² + 10·x2² (слайды, рис. 7–8)"],
    "Химмельблау": PRESETS["Химмельблау (4 минимума)"],
    "сфера n=5": PRESETS["Сфера, n = 5"],
    "сфера n=10": PRESETS["Сфера, n = 10"],
    "Розенброк": PRESETS["Розенброк"],
    "негладкий овраг": PRESETS["Негладкий овраг (слайды, рис. 15–16)"],
}


def run(p, noise=0.0, seed=0, **kw):
    f = Objective(p["expr"], noise, seed)
    res = minimize(f, p["x0"], p["R0"], Settings(**kw))
    return f, res


def reach(res, fmin, key):
    # сколько итераций / вычислений понадобилось, чтобы f лучшей вершины ушла ниже fmin + TOL
    for st in res.steps:
        if st["f_best"] - fmin < TOL:
            return st[key]
    return ""


def write(name, rows):
    RES.mkdir(exist_ok=True)
    with open(RES / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: {len(rows)} строк")


def criteria():
    rows = []
    for fname, p in FUNCS.items():
        for crit in ALL:
            for mapping in ("1", "2") if crit != "single" else ("1",):
                f, res = run(p, criterion=crit, mapping=mapping)
                fail = sum(st["cost"] for st in res.steps[1:] if not st["ok"])
                # та же доля, но только до достижения точности (без хвоста сжатий у минимума)
                n_tol = reach(res, p["fmin"], "N")
                head = [st for st in res.steps[1:] if n_tol != "" and st["N"] <= n_tol]
                fail_tol = sum(st["cost"] for st in head if not st["ok"])
                total_tol = head[-1]["evals"] if head else 0
                rows.append(
                    dict(
                        function=fname,
                        criterion=crit,
                        mapping=mapping,
                        iters_to_tol=reach(res, p["fmin"], "N"),
                        evals_to_tol=reach(res, p["fmin"], "evals"),
                        iters=res.iters,
                        evals=res.evals,
                        f_final=f"{f.exact(res.x) - p['fmin']:.3g}",
                        fail_share=f"{fail / res.evals:.2f}",
                        fail_share_to_tol=f"{fail_tol / total_tol:.2f}" if total_tol else "",
                        stop=res.stop,
                    )
                )
    write("criteria.csv", rows)


def fstar_noise():
    rows = []
    p = FUNCS["овраг"]
    for sigma in (0.0, 0.01, 0.1, 1.0):
        for fs in ("mean", "center"):
            errs, evals = [], []
            for seed in range(20):
                f, res = run(p, sigma, seed, criterion="I1", fstar=fs, max_evals=3000)
                errs.append(float(f.exact(res.x)))
                evals.append(res.evals)
            rows.append(
                dict(
                    sigma=sigma,
                    fstar=fs,
                    err_median=f"{np.median(errs):.3g}",
                    err_max=f"{np.max(errs):.3g}",
                    evals_median=int(np.median(evals)),
                )
            )
    write("noise.csv", rows)


def gamma_tries():
    rows = []
    p = FUNCS["сфера n=5"]
    for crit in ("single", "I1", "I3"):
        for gamma in (0.3, 0.5, 0.7, 0.9):
            for tries in (1, 2, 5):
                f, res = run(p, criterion=crit, gamma=gamma, tries=tries)
                rows.append(
                    dict(
                        criterion=crit,
                        gamma=gamma,
                        tries=tries,
                        evals_to_tol=reach(res, p["fmin"], "evals"),
                        iters_to_tol=reach(res, p["fmin"], "N"),
                    )
                )
    write("gamma_tries.csv", rows)


def dimension():
    rows = []
    for n in (2, 5, 10, 20):
        p = dict(expr=sphere(n), x0=[-2] * n, R0=1.0, fmin=0.0)
        for crit in ("single", "I1", "I3", "I4"):
            f, res = run(p, criterion=crit, max_evals=50000)
            rows.append(
                dict(
                    n=n,
                    criterion=crit,
                    evals_to_tol=reach(res, 0.0, "evals"),
                    iters_to_tol=reach(res, 0.0, "N"),
                )
            )
    write("dimension.csv", rows)


def gamma_eps():
    # γ и ε на разных функциях: универсально лучшего γ нет, ε почти не влияет
    rows = []
    for fname in ("овраг", "Химмельблау", "сфера n=5", "Розенброк", "негладкий овраг"):
        p = FUNCS[fname]
        for gamma in (0.3, 0.5, 0.7, 0.9):
            f, res = run(p, criterion="I1", gamma=gamma)
            rows.append(
                dict(
                    function=fname,
                    param="gamma",
                    value=gamma,
                    evals=res.evals,
                    f_final=f"{f.exact(res.x) - p['fmin']:.3g}",
                    stop=res.stop,
                )
            )
        for eps in (1e-4, 1e-2, 1.0, 10.0):
            f, res = run(p, criterion="I1", eps=eps)
            rows.append(
                dict(
                    function=fname,
                    param="eps",
                    value=eps,
                    evals=res.evals,
                    f_final=f"{f.exact(res.x) - p['fmin']:.3g}",
                    stop=res.stop,
                )
            )
    write("gamma_eps.csv", rows)


def cycling():
    # отображение одной вершины: сколько раз симплекс возвращался к положению двумя шагами раньше
    def key(V):
        return tuple(sorted(map(tuple, np.round(V, 9))))

    rows = []
    p = FUNCS["овраг"]
    for label, eps in (("с условием успешности", 0.01), ("условие выключено", -1e9)):
        f, res = run(p, criterion="single", eps=eps, max_iters=300)
        seen = [key(st["V"]) for st in res.steps]
        back = sum(1 for i in range(2, len(seen)) if seen[i] == seen[i - 2])
        rows.append(dict(variant=label, iters=res.iters, returns=back, f_final=f"{f.exact(res.x):.3g}"))
    write("cycling.csv", rows)


def nelder_mead():
    # не входит в задание (деформируемый симплекс) — только внешняя точка сравнения
    try:
        from scipy.optimize import minimize as sp_min
    except ImportError:
        print("scipy не установлен — сравнение с Нелдером–Мидом пропущено")
        return
    rows = []
    for fname, p in FUNCS.items():
        f = Objective(p["expr"])
        hist = []

        def g(x):
            y = f(x)
            hist.append((f.calls, y))
            return y

        sp_min(g, p["x0"], method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-12, maxfev=5000))
        to_tol = next((c for c, y in hist if y - p["fmin"] < TOL), "")
        rows.append(dict(function=fname, evals_to_tol=to_tol, evals=f.calls, f_final=f"{min(y for _, y in hist):.3g}"))
    write("nelder_mead.csv", rows)


def alpha_check():
    # разброс длин рёбер после отображения при разных alpha: правильным симплекс остаётся только при alpha = 2
    rows = []
    for alpha in (1.5, 2.0, 3.0):
        for kind in ("1", "2"):
            worst = 0.0
            for n in range(2, 8):
                V = regular_simplex(np.zeros(n), 1.0)
                for m, l in directions(n):
                    worst = max(worst, irregularity(MAPPINGS[kind](V, m, l, alpha)))
            rows.append(dict(alpha=alpha, mapping=kind, n="2..7", max_irregularity=f"{worst:.2g}"))
    write("alpha_check.csv", rows)


def i2_check():
    # прогноз I2 против фактического уменьшения f* (среднего по вершинам) на линейной функции
    rng = np.random.default_rng(3)
    n = 5
    c = rng.normal(size=n)
    V = regular_simplex(rng.normal(size=n), 1.0)
    fv = np.array([c @ v for v in V])
    o = np.argsort(-fv)
    V, fv = V[o], fv[o]
    rows = []
    for m, l in directions(n):
        W = MAPPINGS["1"](V, m, l)
        real = fv.mean() - np.mean([c @ w for w in W])
        fixed = 2 * m / (n + 1 - l) * gap(fv, m, l)
        slide = 2 * m / (n + 1 - m) * gap(fv, m, l)
        rows.append(
            dict(
                m=m,
                l=l,
                real=f"{real:.4f}",
                formula_n1_minus_l=f"{fixed:.4f}",
                formula_slide=f"{slide:.4f}",
                slide_error=f"{slide / real - 1:+.0%}",
            )
        )
    write("i2_check.csv", rows)


def figure_cost():
    # одна и та же гонка в двух осях: по итерациям и по вычислениям f
    p = FUNCS["сфера n=10"]
    runs = [
        ("single", "одна вершина", "#2a78d6", "-"),
        ("I1", "I1 по вершинам", "#eb6834", "--"),
        ("I1grad", "I1 по градиенту", "#1baf7a", ":"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for crit, label, color, ls in runs:
        f, res = run(p, criterion=crit)
        it = [st["N"] for st in res.steps]
        ev = [st["evals"] for st in res.steps]
        fb = [st["f_best"] for st in res.steps]
        axes[0].semilogy(it, fb, color=color, ls=ls, lw=2, label=label)
        axes[1].semilogy(ev, fb, color=color, ls=ls, lw=2, label=label)
    axes[0].set_title("по итерациям", fontsize=11)
    axes[1].set_title("по вычислениям f", fontsize=11)
    axes[0].set_xlabel("итерации")
    axes[1].set_xlabel("вычисления f")
    axes[0].set_ylabel("f лучшей вершины")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "cost_sphere10.png", dpi=150)
    print("cost_sphere10.png")


if __name__ == "__main__":
    criteria()
    fstar_noise()
    gamma_tries()
    dimension()
    gamma_eps()
    cycling()
    nelder_mead()
    alpha_check()
    i2_check()
    figure_cost()
