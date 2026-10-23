"""Окно для запуска метода и просмотра траектории. Запуск: python app.py"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from simplex import PRESETS, Objective, Settings, minimize  # noqa: E402
from simplex.functions import parse  # noqa: E402
from simplex.method import STOP_TEXT  # noqa: E402
from simplex.runlog import load, save  # noqa: E402

OWN = "своя формула"

MAPPINGS = {
    "одной вершины": ("single", "1"),
    "отображение 1 (m+l вершин)": (None, "1"),
    "отображение 2 (m+l вершин)": (None, "2"),
}

CRITERIA = {
    "I1 — близость к антиградиенту": "I1",
    "I1 по численному градиенту (+2n вычислений)": "I1grad",
    "I2 — наибольшее уменьшение f*": "I2",
    "I2 в записи слайда (n+1−m)": "I2slide",
    "I3 — уменьшение f* на одну вершину": "I3",
    "I4 — вершины хуже f*": "I4",
    "I5 — среднее по отображаемым": "I5",
    "I6 — вершины хуже (fmax+fmin)/2": "I6",
    "I7 — верхняя треть, l — средняя": "I7",
}

FSTAR = {"среднее по вершинам": "mean", "значение в центре (+1 вычисление)": "center"}


def key_of(d, value):
    return next(k for k, v in d.items() if v == value)


def numbers(text):
    try:
        return [float(t) for t in text.replace(";", ",").split(",") if t.strip()]
    except ValueError:
        raise ValueError(f"начальная точка «{text}»: нужны числа через запятую, например -5, -4") from None


def num(text, what, cast=float):
    try:
        return cast(text.replace(",", "."))
    except ValueError:
        raise ValueError(f"{what}: «{text}» — не число") from None


class App:
    def __init__(self, root):
        self.root = root
        root.title("Правильный симплекс — поисковая оптимизация")
        self.steps = []
        self.expr = None
        self.task = None
        self.result = None
        self.view = None
        self.drag = None

        left = ttk.Frame(root, padding=8)
        left.pack(side="left", fill="y")
        right = ttk.Frame(root, padding=(0, 8, 8, 8))
        right.pack(side="left", fill="both", expand=True)

        self.vars = {}
        row = 0

        def section(title):
            nonlocal row
            ttk.Label(left, text=title, font=("TkDefaultFont", 10, "bold")).grid(
                row=row, column=0, columnspan=2, sticky="w", pady=(8, 2)
            )
            row += 1

        def field(name, label, value, values=None, width=28):
            nonlocal row
            ttk.Label(left, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6))
            var = tk.StringVar(value=value)
            if values:
                w = ttk.Combobox(left, textvariable=var, values=values, state="readonly", width=width)
            else:
                w = ttk.Entry(left, textvariable=var, width=width + 2)
            w.grid(row=row, column=1, sticky="we", pady=1)
            self.vars[name] = var
            row += 1
            return w

        section("Функция")
        first = next(iter(PRESETS))
        field("preset", "пример", first, list(PRESETS) + [OWN], width=34)
        field("expr", "f(x) =", PRESETS[first]["expr"], width=34)
        field("x0", "начальная точка", "-5, -4")
        field("R0", "радиус R1", "1")
        field("noise", "шум измерений σ", "0")

        section("Метод")
        field("mapping", "отображение", "отображение 1 (m+l вершин)", list(MAPPINGS), width=34)
        self.crit_box = field("criterion", "критерий", key_of(CRITERIA, "I1"), list(CRITERIA), width=34)
        field("fstar", "f*", key_of(FSTAR, "mean"), list(FSTAR), width=34)
        field("eps", "ε в условии успешности", "0.01")
        field("gamma", "γ — коэффициент сжатия", "0.5")
        field("tries", "направлений до сжатия", "1")

        section("Останов (срабатывает первое)")
        field("r_min", "радиус меньше", "1e-6")
        field("f_spread", "разброс f меньше (0 — выкл.)", "0")
        field("max_evals", "бюджет вычислений f", "5000")
        field("max_iters", "лимит итераций", "20000")

        buttons = ttk.Frame(left)
        buttons.grid(row=row, column=0, columnspan=2, pady=10, sticky="we")
        ttk.Button(buttons, text="Запустить", command=self.run).pack(side="left")
        ttk.Button(buttons, text="Сохранить лог…", command=self.save_log).pack(side="left", padx=4)
        ttk.Button(buttons, text="Открыть лог…", command=self.open_log).pack(side="left")
        row += 1

        self.info = tk.StringVar()
        ttk.Label(left, textvariable=self.info, justify="left", wraplength=380).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )

        self.fig = Figure(figsize=(10, 5.2))
        self.ax_path = self.fig.add_subplot(1, 2, 1)
        self.ax_conv = self.fig.add_subplot(1, 2, 2)
        self.ax_path.set_axis_off()
        self.ax_conv.set_axis_off()
        self.ax_path.text(0.5, 0.5, "задайте параметры и нажмите «Запустить»", ha="center", color="gray")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        bottom = ttk.Frame(right)
        bottom.pack(fill="x")
        self.step_text = tk.StringVar()
        ttk.Label(bottom, text="шаг").pack(side="left")
        self.slider = ttk.Scale(bottom, from_=0, to=0, command=lambda _: self.draw_path())
        self.slider.pack(side="left", fill="x", expand=True, padx=6)
        ttk.Label(bottom, textvariable=self.step_text, width=46).pack(side="left")

        zoom = ttk.Frame(right)
        zoom.pack(fill="x", pady=(4, 0))
        ttk.Button(zoom, text="Весь путь", command=self.zoom_all).pack(side="left")
        ttk.Button(zoom, text="К текущему симплексу", command=self.zoom_simplex).pack(side="left", padx=4)
        ttk.Label(zoom, text="колесо мыши — зум, перетаскивание — сдвиг", foreground="gray").pack(side="left", padx=6)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)

        self.vars["preset"].trace_add("write", lambda *_: self.on_preset())
        self.vars["mapping"].trace_add("write", lambda *_: self.on_mapping())

    def on_preset(self):
        name = self.vars["preset"].get()
        if name == OWN:
            return
        p = PRESETS[name]
        self.vars["expr"].set(p["expr"])
        self.vars["x0"].set(", ".join(f"{v:g}" for v in p["x0"]))
        self.vars["R0"].set(f"{p['R0']:g}")

    def on_mapping(self):
        single = MAPPINGS[self.vars["mapping"].get()][0] == "single"
        self.crit_box.configure(state="disabled" if single else "readonly")

    def read_task(self):
        v = {k: var.get().strip() for k, var in self.vars.items()}
        _, n = parse(v["expr"])
        x0 = numbers(v["x0"])
        if len(x0) != n:
            raise ValueError(f"в начальной точке {len(x0)} координат, а в функции {n} переменных")
        single, mapping = MAPPINGS[v["mapping"]]
        s = Settings(
            criterion=single or CRITERIA[v["criterion"]],
            mapping=mapping,
            fstar=FSTAR[v["fstar"]],
            eps=num(v["eps"], "ε"),
            gamma=num(v["gamma"], "γ"),
            tries=num(v["tries"], "число направлений", int),
            r_min=num(v["r_min"], "порог радиуса"),
            f_spread=num(v["f_spread"], "порог разброса f"),
            max_evals=num(v["max_evals"], "бюджет вычислений", int),
            max_iters=num(v["max_iters"], "лимит итераций", int),
        )
        if not 0 < s.gamma < 1:
            raise ValueError("γ должно быть в интервале (0, 1)")
        if s.eps <= 0:
            raise ValueError("ε должно быть больше нуля")
        if s.tries < 1:
            raise ValueError("направлений до сжатия — хотя бы одно")
        if s.r_min <= 0:
            raise ValueError("порог радиуса должен быть больше нуля")
        if s.f_spread < 0:
            raise ValueError("порог разброса f не может быть отрицательным")
        if s.max_evals < n + 2:
            raise ValueError(f"бюджет меньше {n + 2} вычислений: не хватит даже на начальный симплекс и один шаг")
        if s.max_iters < 1:
            raise ValueError("лимит итераций — хотя бы одна")
        task = dict(expr=v["expr"], x0=x0, R0=num(v["R0"], "радиус R1"), noise=num(v["noise"], "σ шума"), seed=0)
        if task["R0"] <= 0:
            raise ValueError("радиус R1 должен быть больше нуля")
        if task["noise"] < 0:
            raise ValueError("σ шума не может быть отрицательной")
        return task, s

    def run(self):
        try:
            task, s = self.read_task()
        except ValueError as e:
            messagebox.showerror("Проверьте параметры", str(e))
            return
        self.root.config(cursor="watch")
        self.root.update()
        try:
            f = Objective(task["expr"], task["noise"], task["seed"])
            res = minimize(f, task["x0"], task["R0"], s)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
            messagebox.showerror("Ошибка при вычислении f", str(e))
            return
        finally:
            self.root.config(cursor="")
        self.task, self.result = task, res
        self.show(task["expr"], res.steps, float(f.exact(res.x)), res.x, res.stop)

    def show(self, expr, steps, fx, x, stop):
        self.expr, self.steps = expr, steps
        self.view = None
        ok = sum(1 for st in steps[1:] if st["ok"])
        self.info.set(
            f"x* = ({', '.join(f'{v:.6g}' for v in x)})\n"
            f"f(x*) = {fx:.6g}\n"
            f"итераций: {len(steps) - 1}, успешных шагов: {ok}, сжатий: {len(steps) - 1 - ok}\n"
            f"вычислений f: {int(steps[-1]['evals'])}\n"
            f"останов: {STOP_TEXT[stop]}"
        )
        self.slider.configure(to=len(steps) - 1)
        self.slider.set(len(steps) - 1)
        self.draw_conv()
        self.draw_path()

    def draw_conv(self):
        ax = self.ax_conv
        ax.clear()
        ax.set_axis_on()
        ev = [st["evals"] for st in self.steps]
        fb = np.array([st["f_best"] for st in self.steps])
        shift = 0 if fb.min() > 0 else -fb.min() + 1e-12
        ax.semilogy(ev, fb + shift, lw=1.2, label="лучшая вершина")
        ax.semilogy(ev, [st["R"] for st in self.steps], lw=1, ls="--", label="радиус R")
        ax.set_xlabel("вычислений f")
        ax.set_title("сходимость")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    def draw_path(self):
        if not self.steps:
            return
        N = int(round(float(self.slider.get())))
        st = self.steps[N]
        mv = f"m={int(st['m'])}, l={int(st['l'])}"
        what = "старт" if N == 0 else (f"шаг {mv}" if st["ok"] else f"{mv} не прошло, сжатие")
        self.step_text.set(f"{N} из {len(self.steps) - 1}: {what}, R={st['R']:.3g}")
        ax = self.ax_path
        ax.clear()
        n = len(self.steps[0]["center"])
        if n != 2:
            ax.text(0.5, 0.5, f"n = {n}: траектория рисуется только для двух переменных", ha="center", wrap=True)
            ax.set_axis_off()
            self.canvas.draw_idle()
            return
        ax.set_axis_on()
        xl, yl = self.view or self.bounds(np.vstack([s["V"] for s in self.steps[: N + 1]]), 0.15)
        ax.set_xlim(xl)
        ax.set_ylim(yl)
        ax.set_aspect("equal", adjustable="datalim")
        ax.apply_aspect()
        g1 = np.linspace(*ax.get_xlim(), 160)
        g2 = np.linspace(*ax.get_ylim(), 160)
        f, _ = parse(self.expr)
        A, B = np.meshgrid(g1, g2)
        Z = f((A, B)) + np.zeros_like(A)
        levels = np.unique(np.quantile(Z[np.isfinite(Z)], np.linspace(0.02, 0.98, 14)))
        if len(levels) > 1:
            ax.contour(g1, g2, Z, levels=levels, colors="#4060c0", linewidths=0.6)
        tris = [np.vstack([s["V"], s["V"][:1]]) for s in self.steps[: N + 1]]
        ax.add_collection(LineCollection(tris, colors="#d03030", linewidths=0.5, alpha=0.6))
        cur = np.vstack([st["V"], st["V"][:1]])
        ax.plot(cur[:, 0], cur[:, 1], color="#a00000", lw=1.8)
        c = np.array([s["center"] for s in self.steps[: N + 1]])
        ax.plot(c[:, 0], c[:, 1], ":", color="black", lw=0.8)
        ax.set_xlim(g1[0], g1[-1])
        ax.set_ylim(g2[0], g2[-1])
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title("траектория симплексов")
        self.canvas.draw_idle()

    @staticmethod
    def bounds(pts, pad):
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        d = (hi - lo).max() * pad + 1e-12 * (1 + np.abs(lo).max())
        return (lo[0] - d, hi[0] + d), (lo[1] - d, hi[1] + d)

    def zoomable(self):
        return len(self.steps) > 0 and len(self.steps[0]["center"]) == 2

    def zoom_all(self):
        self.view = None
        self.draw_path()

    def zoom_simplex(self):
        if not self.zoomable():
            return
        st = self.steps[int(round(float(self.slider.get())))]
        self.view = self.bounds(np.asarray(st["V"]), 1.5)
        self.draw_path()

    def on_scroll(self, e):
        if e.inaxes is not self.ax_path or not self.zoomable():
            return
        k = 0.8 if e.button == "up" else 1.25
        (x0, x1), (y0, y1) = self.ax_path.get_xlim(), self.ax_path.get_ylim()
        if k < 1 and (x1 - x0) < 1e-10 * (1 + abs(e.xdata)):
            return
        self.view = (
            (e.xdata + (x0 - e.xdata) * k, e.xdata + (x1 - e.xdata) * k),
            (e.ydata + (y0 - e.ydata) * k, e.ydata + (y1 - e.ydata) * k),
        )
        self.draw_path()

    def on_press(self, e):
        if e.inaxes is not self.ax_path or not self.zoomable() or e.button != 1:
            return
        if e.dblclick:
            self.zoom_all()
            return
        self.drag = (e.x, e.y, self.ax_path.get_xlim(), self.ax_path.get_ylim())

    def on_motion(self, e):
        if not self.drag or e.x is None:
            return
        px, py, xl, yl = self.drag
        inv = self.ax_path.transData.inverted()
        (ax0, ay0), (ax1, ay1) = inv.transform([(px, py), (e.x, e.y)])
        dx, dy = ax1 - ax0, ay1 - ay0
        self.ax_path.set_xlim(xl[0] - dx, xl[1] - dx)
        self.ax_path.set_ylim(yl[0] - dy, yl[1] - dy)
        self.canvas.draw_idle()

    def on_release(self, e):
        if not self.drag:
            return
        start = self.drag[2:]
        self.drag = None
        if (self.ax_path.get_xlim(), self.ax_path.get_ylim()) == start:
            return
        self.view = self.ax_path.get_xlim(), self.ax_path.get_ylim()
        self.draw_path()

    def save_log(self):
        if not self.result:
            messagebox.showinfo("Нечего сохранять", "Сначала запустите метод.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            save(path, self.result, self.task)

    def open_log(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            meta, steps = load(path)
        except (OSError, KeyError, ValueError) as e:
            messagebox.showerror("Не удалось открыть лог", f"{e}\nРядом с CSV должен лежать JSON с тем же именем.")
            return
        s = meta["settings"]
        self.vars["preset"].set(OWN)
        self.vars["expr"].set(meta["expr"])
        self.vars["x0"].set(", ".join(f"{v:g}" for v in meta["x0"]))
        self.vars["R0"].set(f"{meta['R0']:g}")
        self.vars["noise"].set(f"{meta['noise']:g}")
        if s["criterion"] == "single":
            self.vars["mapping"].set("одной вершины")
        else:
            self.vars["mapping"].set(key_of(MAPPINGS, (None, s["mapping"])))
            self.vars["criterion"].set(key_of(CRITERIA, s["criterion"]))
        self.vars["fstar"].set(key_of(FSTAR, s["fstar"]))
        for k in ("eps", "gamma", "tries", "r_min", "f_spread", "max_evals", "max_iters"):
            self.vars[k].set(f"{s[k]:g}")
        self.result = None
        r = meta["result"]
        f, _ = parse(meta["expr"])
        self.show(meta["expr"], steps, float(f(r["x"])), r["x"], r["stop"])


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
