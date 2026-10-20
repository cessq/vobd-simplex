"""Запуск метода из консоли по JSON-конфигу.

python run.py experiments/configs/ravine.json
python run.py experiments/configs/ravine.json -o out/ravine.csv
"""

import argparse
import json
from pathlib import Path

from simplex import PRESETS, Objective, Settings, minimize
from simplex.method import STOP_TEXT
from simplex.runlog import save


def load_task(path):
    task = json.loads(Path(path).read_text())
    if "preset" in task:
        base = dict(PRESETS[task["preset"]])
        base.update({k: v for k, v in task.items() if k != "preset"})
        task = base
    task.setdefault("noise", 0.0)
    task.setdefault("seed", 0)
    task.setdefault("settings", {})
    return task


def run(task):
    f = Objective(task["expr"], task["noise"], task["seed"])
    if len(task["x0"]) != f.n:
        raise SystemExit(f"в x0 {len(task['x0'])} координат, а в функции {f.n} переменных")
    res = minimize(f, task["x0"], task["R0"], Settings(**task["settings"]))
    return f, res


def main():
    ap = argparse.ArgumentParser(description="Метод деформируемых конфигураций с правильным симплексом")
    ap.add_argument("config")
    ap.add_argument("-o", "--out", help="куда писать лог (по умолчанию experiments/results/<имя конфига>.csv)")
    args = ap.parse_args()

    task = load_task(args.config)
    try:
        f, res = run(task)
    except ValueError as e:
        raise SystemExit(f"ошибка: {e}") from None
    out = args.out or Path("experiments/results") / (Path(args.config).stem + ".csv")
    save(out, res, {k: task[k] for k in ("expr", "x0", "R0", "noise", "seed")})

    print(f"f(x) = {task['expr']}")
    print(f"x*   = [{', '.join(f'{v:.6g}' for v in res.x)}]")
    print(f"f(x*) = {f.exact(res.x):.6g}")
    print(f"итераций: {res.iters}, вычислений f: {res.evals}")
    print(f"останов: {STOP_TEXT[res.stop]}")
    print(f"лог: {out}")


if __name__ == "__main__":
    main()
