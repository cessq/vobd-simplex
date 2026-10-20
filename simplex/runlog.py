import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

FIELDS = ["N", "evals", "cost", "f_star", "f_best", "R", "m", "l", "ok", "tries", "irregularity"]


def save(path, result, task):
    """Лог шагов в CSV и рядом JSON с постановкой и итогом — по ним прогон можно повторить."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(result.x)
    head = FIELDS + [f"c{k}" for k in range(1, n + 1)]
    head += [f"v{j}_{k}" for j in range(1, n + 2) for k in range(1, n + 1)]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        for st in result.steps:
            row = [st[k] for k in FIELDS[:8]] + [int(st["ok"]), st["tries"], st["irregularity"]]
            w.writerow(row + list(st["center"]) + list(st["V"].ravel()))
    meta = dict(task)
    meta["settings"] = asdict(result.settings)
    meta["result"] = dict(x=list(result.x), f=result.f, evals=result.evals, iters=result.iters, stop=result.stop)
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def load(path):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    n = len(meta["x0"])
    steps = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            st = {k: float(row[k]) for k in FIELDS}
            st["ok"] = bool(st["ok"])
            st["center"] = np.array([float(row[f"c{k}"]) for k in range(1, n + 1)])
            st["V"] = np.array([[float(row[f"v{j}_{k}"]) for k in range(1, n + 1)] for j in range(1, n + 2)])
            steps.append(st)
    return meta, steps
