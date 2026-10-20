import numpy as np
import pytest

from simplex import Objective, Settings, minimize
from simplex.criteria import CRITERIA, i2
from simplex.functions import parse
from simplex.geometry import basis, irregularity, regular_simplex
from simplex.mappings import MAPPINGS, delta, directions
from simplex.runlog import load, save


def sorted_simplex(f, n, seed=0):
    rng = np.random.default_rng(seed)
    V = regular_simplex(rng.normal(size=n), 1.3)
    fv = np.array([f(v) for v in V])
    o = np.argsort(-fv)
    return V[o], fv[o]


@pytest.mark.parametrize("n", range(2, 9))
def test_basis_is_unit_regular_simplex(n):
    B = basis(n)
    assert np.allclose(np.linalg.norm(B, axis=1), 1)
    assert np.allclose(B.mean(axis=0), 0)
    assert irregularity(B) < 1e-12


@pytest.mark.parametrize("n", range(2, 8))
@pytest.mark.parametrize("kind", ["1", "2"])
def test_alpha_2_keeps_simplex_regular(n, kind):
    V = regular_simplex(np.zeros(n), 1.0)
    for m, l in directions(n):
        W = MAPPINGS[kind](V, m, l, 2.0)
        assert irregularity(W) < 1e-12
        assert np.allclose(W.mean(axis=0) - V.mean(axis=0), 2.0 * m / (n + 1 - l) * delta(V, m, l))


def test_other_alpha_deforms_simplex():
    V = regular_simplex(np.zeros(3), 1.0)
    assert irregularity(MAPPINGS["1"](V, 1, 0, 1.5)) > 0.1


def test_i2_matches_real_decrease_on_linear_function():
    c = np.array([0.3, -1.2, 0.7, 2.0])
    V, fv = sorted_simplex(lambda x: c @ x, 4)
    for score, m, l in i2(V, fv, None, 2.0, None):
        W = MAPPINGS["1"](V, m, l)
        real = fv.mean() - np.mean([c @ w for w in W])
        assert score == pytest.approx(real)


def test_number_of_directions():
    for n in range(1, 10):
        assert len(directions(n)) == n * (n + 1) // 2


@pytest.mark.parametrize("crit", ["single"] + list(CRITERIA))
@pytest.mark.parametrize("fstar", ["mean", "center"])
def test_converges_on_quadratic(crit, fstar):
    f = Objective("x1**2 + 10*x2**2 + (x3 - 1)**2")
    res = minimize(f, [-3, 2, 4], 1.0, Settings(criterion=crit, fstar=fstar))
    assert res.stop == "radius"
    assert f.exact(res.x) < 1e-8
    assert max(st["irregularity"] for st in res.steps) < 1e-8
    assert res.evals == f.calls == sum(st["cost"] for st in res.steps)


def test_budget_stop():
    f = Objective("100*(x2 - x1**2)**2 + (1 - x1)**2")
    res = minimize(f, [-1.2, 1], 0.5, Settings(max_evals=200))
    assert res.stop == "evals"
    assert 200 <= res.evals < 200 + 10


def test_parse():
    f, n = parse("x1^2 + sin(x3)")
    assert n == 3
    assert f([2, 0, 0]) == 4
    with pytest.raises(ValueError):
        parse("x1 + y")
    with pytest.raises(ValueError):
        parse("x1.__class__")


def test_log_roundtrip(tmp_path):
    f = Objective("x1**2 + 10*x2**2")
    res = minimize(f, [-5, -4], 1.0)
    save(tmp_path / "run.csv", res, dict(expr=f.expr, x0=[-5, -4], R0=1.0, noise=0.0, seed=0))
    meta, steps = load(tmp_path / "run.csv")
    assert meta["result"]["evals"] == res.evals
    assert len(steps) == len(res.steps)
    assert np.allclose(steps[-1]["V"], res.steps[-1]["V"])


def test_bad_formula_is_value_error():
    with pytest.raises(ValueError):
        parse("x1 ** ")


def test_undefined_start_is_reported():
    with pytest.raises(ValueError):
        minimize(Objective("log(x1) + x2**2"), [-3, 1], 1.0)


def test_undefined_region_counts_as_worst():
    # log определён только при x1 > 0: шаги в запрещённую область должны отбрасываться
    f = Objective("log(x1)**2 + x2**2")
    res = minimize(f, [0.5, 1], 0.4)
    assert res.x[0] > 0
    assert f.exact(res.x) < 1e-8
