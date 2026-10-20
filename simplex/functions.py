import ast
import re

import numpy as np

NAMES = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "arctan": np.arctan,
    "tanh": np.tanh,
    "pi": np.pi,
    "e": np.e,
}


def parse(expr):
    """Строка вида 'x1**2 + 10*x2**2' -> (функция от вектора, размерность)."""
    try:
        tree = ast.parse(expr.replace("^", "**"), mode="eval")
    except SyntaxError:
        raise ValueError(f"формула «{expr}» не разобрана — проверьте скобки и знаки") from None
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            raise ValueError("точка в формуле не поддерживается")
        if isinstance(node, ast.Name):
            used.add(node.id)
    xs = {u for u in used if re.fullmatch(r"x[1-9]\d*", u)}
    unknown = used - xs - NAMES.keys()
    if unknown:
        raise ValueError("неизвестные имена: " + ", ".join(sorted(unknown)))
    if not xs:
        raise ValueError("в формуле нет переменных x1, x2, ...")
    n = max(int(u[1:]) for u in xs)
    code = compile(tree, "<f>", "eval")

    def f(x):
        # x может быть набором сеток — тогда функция считается сразу на всей сетке
        env = dict(NAMES)
        env.update({f"x{i + 1}": x[i] for i in range(n)})
        return eval(code, {"__builtins__": {}}, env)

    return f, n


class Objective:
    """Функция с подсчётом вычислений и, при желании, шумом «измерения»."""

    def __init__(self, expr, noise=0.0, seed=0):
        self.expr = expr
        self.exact, self.n = parse(expr)
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        with np.errstate(all="ignore"):
            y = float(self.exact(x))
        if not np.isfinite(y):
            # точка вне области определения (log отрицательного и т.п.) — считаем её наихудшей
            return np.inf
        if self.noise:
            y += self.rng.normal(0, self.noise)
        return y


def sphere(n):
    return " + ".join(f"(x{i} - 1)**2" for i in range(1, n + 1))


# x0, R0 — старт по умолчанию; fmin — известный минимум (для оценки точности)
PRESETS = {
    "Овраг x1² + 10·x2² (слайды, рис. 7–8)": dict(expr="x1**2 + 10*x2**2", x0=[-5, -4], R0=1.0, fmin=0.0),
    "Негладкий овраг (слайды, рис. 15–16)": dict(
        expr="abs(x2 - abs(x1)) + 2.92*abs(1 - x1)", x0=[-15, 25], R0=3.0, fmin=0.0
    ),
    "Розенброк": dict(expr="100*(x2 - x1**2)**2 + (1 - x1)**2", x0=[-1.2, 1], R0=0.5, fmin=0.0),
    "Химмельблау (4 минимума)": dict(expr="(x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2", x0=[0, 0], R0=1.0, fmin=0.0),
    "Сфера, n = 5": dict(expr=sphere(5), x0=[-2] * 5, R0=1.0, fmin=0.0),
    "Сфера, n = 10": dict(expr=sphere(10), x0=[-2] * 10, R0=1.0, fmin=0.0),
}
