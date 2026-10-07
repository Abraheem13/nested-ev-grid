"""Static figures generated from code: feeder topology, framework architecture and
the Level-3 correction loop. Everything drawn here is derived from the network
data and the configuration, so the figures cannot drift from the implementation."""
from __future__ import annotations

import pathlib

import numpy as np

from ..grid.network import load_network


def _tree_layout(net):
    """x = depth along the main path, y separates laterals (simple tidy tree)."""
    children = {i: [] for i in range(net.n_bus)}
    for i in range(1, net.n_bus):
        children[net.parent[i]].append(i)
    size = {}

    def subtree(u):
        size[u] = 1 + sum(subtree(c) for c in children[u])
        return size[u]

    subtree(0)
    pos = {}

    def place(u, x, y):
        pos[u] = (x, y)
        kids = sorted(children[u], key=lambda c: -size[c])
        if not kids:
            return
        place(kids[0], x + 1, y)                       # longest branch continues straight
        off = y
        for c in kids[1:]:
            off -= 1.0 + 0.0 * size[c]
            place(c, x + 1, off)

    place(0, 0, 0)
    # resolve overlaps of laterals by pushing later laterals further down
    ys = {}
    for u, (x, y) in sorted(pos.items(), key=lambda kv: kv[1][0]):
        ys.setdefault(round(y, 3), []).append(u)
    return pos, children


FONT = {"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix", "pdf.fonttype": 42}


def fig_feeder(cfg: dict, network: str, out: pathlib.Path) -> None:
    """Drawn at its printed size (one column), so 7 pt here is 7 pt on paper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(FONT)
    net = load_network(network)
    children = {i: [] for i in range(net.n_bus)}
    for i in range(1, net.n_bus):
        children[net.parent[i]].append(i)
    depth_size = {}

    def subtree(u):
        depth_size[u] = 1 + sum(subtree(c) for c in children[u])
        return depth_size[u]

    subtree(0)
    pos, row = {}, [0]

    def place(u, x, y):
        pos[u] = (x, y)
        kids = sorted(children[u], key=lambda c: -depth_size[c])
        if kids:
            place(kids[0], x + 1, y)
            for c in kids[1:]:
                row[0] -= 1
                place(c, x + 1, row[0])

    place(0, 0, 0)
    agg = np.asarray(cfg["aggregators"]["buses"][network]) - 1
    fig, ax = plt.subplots(figsize=(3.45, 1.3))
    for i in range(1, net.n_bus):
        (x0, y0), (x1, y1) = pos[net.parent[i]], pos[i]
        ax.plot([x0, x0, x1], [y0, y1, y1], color="0.35", lw=0.7, zorder=1)
    xs = np.array([pos[i][0] for i in range(net.n_bus)])
    ys = np.array([pos[i][1] for i in range(net.n_bus)])
    ax.scatter(xs, ys, s=6, color="k", zorder=2)
    ax.scatter(xs[agg], ys[agg], s=45, facecolor="none", edgecolor="#c00000", lw=1.0, zorder=3)
    for k, b in enumerate(agg):
        ax.annotate(f"A{k + 1} (bus {b + 1})", (xs[b], ys[b]), xytext=(0, 5), textcoords="offset points",
                    fontsize=7, ha="center", color="#c00000")
    ax.scatter([xs[0]], [ys[0]], marker="s", s=30, color="#0072b2", zorder=3)
    ax.annotate("substation", (xs[0], ys[0]), xytext=(0, 5), textcoords="offset points", fontsize=7,
                ha="center", color="#0072b2")
    ax.set_xlim(xs.min() - 1.9, xs.max() + 2.0)
    ax.set_ylim(ys.min() - 0.6, ys.max() + 1.0)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(out / f"fig_feeder_{network}.pdf")
    plt.close(fig)


# Diagrams in a plain technical style: black rectangles on white, thin black
# arrows, labels on the arrows, dashed grey boundaries. Axis units are inches,
# and every figure is drawn at its printed size.
LW, FS, GREY = 0.7, 7.0, "#6e6e6e"


def _canvas(w, h):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(FONT)
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_axis_off()
    return fig, ax


def _rect(ax, x, y, w, h, text, fs=FS):
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec="black", lw=LW, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.15, zorder=3)
    return {"l": x, "r": x + w, "b": y, "t": y + h, "cx": x + w / 2, "cy": y + h / 2}


def _path(ax, pts, dashed=False, head=True):
    """Polyline through pts with an arrowhead at the last point."""
    ls = (0, (3, 2)) if dashed else "-"
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color="black", lw=LW, ls=ls, zorder=1,
            solid_capstyle="butt")
    if head:
        ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=4,
                    arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.18", color="black", lw=LW,
                                    shrinkA=0, shrinkB=0))


def _label(ax, x, y, text, ha="left", va="center", fs=FS - 0.5, color="black"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, zorder=5)


def _boundary(ax, x, y, w, h, label, where="below"):
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=GREY, lw=0.6, ls=(0, (4, 2.5)), zorder=0))
    if where == "below":
        _label(ax, x, y - 0.07, label, va="top", color=GREY, fs=FS)
    else:
        _label(ax, x, y + h + 0.05, label, va="bottom", color=GREY, fs=FS)


def fig_architecture(out: pathlib.Path) -> None:
    """Nested controller and the simulated environment (text width)."""
    fig, ax = _canvas(7.16, 1.98)
    pr = _rect(ax, 0.02, 1.24, 0.82, 0.36, "day-ahead\nprices")
    pl = _rect(ax, 0.02, 0.67, 0.82, 0.36, "planning\nprior $u^0_k$")
    l1 = _rect(ax, 1.22, 1.24, 1.86, 0.36, "Level 1: price corridor\n(PPO, hourly)")
    l2 = _rect(ax, 1.22, 0.67, 1.86, 0.36, "Level 2: residual dispatch\n(shared DDPG, 15 min)")
    l3 = _rect(ax, 1.22, 0.10, 1.86, 0.36, "Level 3: reactive correction\n(non-learned, 60 s)")
    dr = _rect(ax, 4.42, 1.24, 1.72, 0.36, "drivers accept or\ndecline $p_k$ (2)")
    al = _rect(ax, 4.42, 0.67, 1.72, 0.36, "least-laxity-first allocation\nwith deadline guard")
    fd = _rect(ax, 4.42, 0.10, 1.72, 0.36, "radial feeder,\nAC power flow (1)")
    _boundary(ax, 1.10, 0.02, 2.10, 1.66, "nested controller", where="above")
    _boundary(ax, 4.30, 0.02, 1.96, 1.66, "simulated environment", where="above")
    _path(ax, [(pr["r"], pr["cy"]), (l1["l"], l1["cy"])])
    _path(ax, [(pr["cx"], pr["b"]), (pl["cx"], pl["t"])])
    _path(ax, [(pl["r"], pl["cy"]), (l2["l"], l2["cy"])])
    _path(ax, [(l1["cx"], l1["b"]), (l2["cx"], l2["t"])])
    _label(ax, l1["cx"] + 0.05, (l1["b"] + l2["t"]) / 2, "corridor $[p^{\\min}_t, p^{\\max}_t]$")
    yp = l2["cy"] + 0.1
    _path(ax, [(l2["r"], yp), (3.62, yp), (3.62, dr["cy"]), (dr["l"], dr["cy"])])
    _label(ax, 3.67, dr["cy"] + 0.08, "price $p_k$")
    ys = l2["cy"] - 0.08
    _path(ax, [(l2["r"], ys), (al["l"], ys)])
    _label(ax, 3.67, ys - 0.09, "set point $u_k$")
    _path(ax, [(dr["cx"], dr["b"]), (al["cx"], al["t"])])
    _label(ax, dr["cx"] + 0.05, (dr["b"] + al["t"]) / 2, "accepting vehicles")
    _path(ax, [(al["cx"], al["b"]), (fd["cx"], fd["t"])])
    _label(ax, al["cx"] + 0.05, (al["b"] + fd["t"]) / 2, "charging power $c_i$")
    _path(ax, [(l3["r"], fd["cy"] + 0.08), (fd["l"], fd["cy"] + 0.08)])
    _label(ax, 3.2, fd["cy"] + 0.16, "reactive power $Q_i$; curtailment", va="bottom")
    _path(ax, [(fd["l"], fd["cy"] - 0.08), (l3["r"], fd["cy"] - 0.08)], dashed=True)
    _label(ax, 3.2, fd["cy"] - 0.15, "measured bus voltages", va="top")
    yt = 1.92
    _path(ax, [(fd["r"], fd["cy"]), (6.62, fd["cy"]), (6.62, yt), (l1["cx"], yt), (l1["cx"], l1["t"])], dashed=True)
    _label(ax, 6.67, 0.95, "measured\nstates\n$s^{(1)}_t$, $s^{(2)}_{\\tau,k}$", va="center")
    fig.savefig(out / "fig_architecture.pdf")
    import matplotlib.pyplot as plt
    plt.close(fig)


def fig_l3_loop(out: pathlib.Path) -> None:
    """Control flow of nflev.env.qcontrol.ReactiveController.correct (one column)."""
    fig, ax = _canvas(3.45, 1.6)
    x0, w, h = 0.70, 1.55, 0.2
    pf = _rect(ax, x0, 1.37, w, h, "solve AC power flow")
    ck = _rect(ax, x0, 1.06, w, h, "$\\min_n V_n < \\underline{V} + \\delta$ ?")
    hr = _rect(ax, x0, 0.75, w, h, "reactive headroom left?")
    inj = _rect(ax, x0, 0.39, w, 0.26, "inject $Q$: proportional step,\nthen secant step")
    rs = _rect(ax, x0, 0.06, w, h, "re-solve power flow")
    cu = _rect(ax, 2.55, 0.29, 0.86, 0.48, "curtail EV power,\nmost at the most\ndepressed bus")
    for a, b in ((pf, ck), (ck, hr), (hr, inj), (inj, rs)):
        _path(ax, [(a["cx"], a["b"]), (b["cx"], b["t"])])
    _label(ax, ck["cx"] + 0.05, (ck["b"] + hr["t"]) / 2, "YES", fs=5.8)
    _label(ax, hr["cx"] + 0.05, (hr["b"] + inj["t"]) / 2, "YES", fs=5.8)
    _path(ax, [(ck["r"], ck["cy"]), (ck["r"] + 0.32, ck["cy"])])
    _label(ax, ck["r"] + 0.36, ck["cy"], "NO: done", fs=5.8)
    _path(ax, [(hr["r"], hr["cy"]), (cu["cx"], hr["cy"]), (cu["cx"], cu["t"])])
    _label(ax, hr["r"] + 0.05, hr["cy"] + 0.06, "NO", va="bottom", fs=5.8)
    _path(ax, [(cu["cx"], cu["b"]), (cu["cx"], rs["cy"]), (rs["r"], rs["cy"])])
    _path(ax, [(rs["l"], rs["cy"]), (x0 - 0.12, rs["cy"]), (x0 - 0.12, ck["cy"]), (ck["l"], ck["cy"])])
    _label(ax, 0.02, (rs["cy"] + ck["cy"]) / 2, "repeat:\n$\\leq$8 power\nflows\n(+12 when\ncurtailing)",
           fs=FS - 0.5)
    fig.savefig(out / "fig_l3_loop.pdf")
    import matplotlib.pyplot as plt
    plt.close(fig)


def fig_pipeline(cfg: dict, out: pathlib.Path, n_seeds: int, n_aux: int) -> None:
    """Study pipeline from public data to every reported number (one column);
    counts are read from the configuration and the stored runs."""
    d, ev, tr = cfg["data"], cfg["evaluation"], cfg["training"]
    yrs = lambda k: ", ".join(str(y) for y in d[k])  # noqa: E731
    fig, ax = _canvas(3.45, 2.06)
    x, w, h, g = 0.62, 2.45, 0.27, 0.13
    ys = [1.72, 1.32, 0.92, 0.52, 0.12]
    texts = ["public data: ENTSO-E prices, Pecan Street loads,\nACN-Data sessions, MATPOWER feeders",
             f"quasi-static simulation every {cfg['simulation']['resolution_s']} s: fleets,\ndriver acceptance, AC power flow",
             f"training on {yrs('train_years')} days: {tr['episodes']} episodes, {n_seeds} seeds\n"
             f"({n_aux} for ablations); tariff calibration",
             f"paired evaluation: {ev['episodes']} held-out {yrs('test_years')} days,\n"
             f"S1–S{len(ev['scenarios'])}, 8 baselines with and without Level 3",
             "bootstrap intervals, Holm–Wilcoxon tests;\ntables, figures, number macros, checked claims"]
    boxes = [_rect(ax, x, y, w, h, t, fs=FS - 0.3) for y, t in zip(ys, texts)]
    edge = ["", "episodes", "frozen policies", "daily results"]
    for (a, b), lab in zip(zip(boxes[:-1], boxes[1:]), edge):
        _path(ax, [(a["cx"], a["b"]), (b["cx"], b["t"])])
        if lab:
            _label(ax, a["cx"] + 0.05, (a["b"] + b["t"]) / 2, lab, fs=FS - 0.8)
    _boundary(ax, x - 0.1, 0.05, w + 0.2, ys[0] + h - 0.05 + 0.05, "", where="below")
    ax.text(x - 0.16, (0.05 + ys[0] + h) / 2, "one command: reproduce.py", rotation=90, ha="right", va="center",
            fontsize=FS, color=GREY)
    fig.savefig(out / "fig_pipeline.pdf")
    import matplotlib.pyplot as plt
    plt.close(fig)


def build_diagrams(cfg: dict, out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig_feeder(cfg, "ieee33", out)
    fig_feeder(cfg, "ieee69", out)
    fig_architecture(out)
    fig_l3_loop(out)
