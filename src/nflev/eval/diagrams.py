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


def _box(ax, xy, w, h, text, fc, fs=7):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec="k", lw=0.6))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.15)


def _arrow(ax, a, b, color="k", style="-|>", ls="-"):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle=style, color=color, lw=0.8, linestyle=ls))


def fig_architecture(out: pathlib.Path) -> None:
    """Drawn at its printed size (text width); axis units are inches."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(FONT)
    W, H = 7.16, 1.96
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    fs = 7
    rows = {"l1": 1.42, "l2": 0.66, "l3": 0.04}
    h = {"l1": 0.38, "l2": 0.62, "l3": 0.44}
    x_obs, w_obs = 0.02, 1.62
    x_lev, w_lev = 1.86, 1.92
    x_mid, w_mid = 4.00, 1.62
    x_fdr, w_fdr = 5.84, 1.28
    _box(ax, (x_obs, rows["l1"]), w_obs, h["l1"], "System state (11-d): voltages,\nsubstation power, prices, fleet", "#eef3fb", fs)
    _box(ax, (x_obs, rows["l2"]), w_obs, h["l2"], "Aggregator state (28-d):\nbus voltage, 12 h of prices,\ncorridor, need by laxity,\nplan set point $u^0_k$", "#eef3fb", fs)
    _box(ax, (x_obs, rows["l3"]), w_obs, h["l3"], "Measured bus voltages,\ncharger powers", "#eef3fb", fs)
    _box(ax, (x_lev, rows["l1"]), w_lev, h["l1"], "Level 1: pricing (PPO, 1 h)\ncorridor $[p^{\\min}_t, p^{\\max}_t]$", "#d6e6f8", fs)
    _box(ax, (x_lev, rows["l2"]), w_lev, h["l2"], "Level 2: dispatch (DDPG, 15 min)\nshared actor-critic; residual\non plan $u^0_k$: set point $u_k$,\nexecution price $p_k$", "#dceedd", fs)
    _box(ax, (x_lev, rows["l3"]), w_lev, h["l3"], "Level 3: voltage correction\n(non-learned, 60 s)", "#fde6cf", fs)
    _box(ax, (x_mid, rows["l2"]), w_mid, h["l2"], "Drivers accept or decline $p_k$;\nleast-laxity-first allocation\nwith deadline guard, within\ncharger and transformer limits", "#f3f3f3", fs)
    _box(ax, (x_mid, rows["l3"]), w_mid, h["l3"], "$Q_i^{\\max}=\\sqrt{S_i^2-P_i^2}$\nsecant step,\ncurtailment fallback", "#fbe0e0", fs)
    _box(ax, (x_fdr, rows["l3"]), w_fdr, rows["l2"] + h["l2"] - rows["l3"], "Radial feeder\nIEEE 33/69-bus\nAC power flow\nevery 60 s", "#ececec", fs)
    for key in rows:
        y = rows[key] + h[key] / 2
        _arrow(ax, (x_obs + w_obs, y), (x_lev, y))
    xc = x_lev + w_lev / 2
    _arrow(ax, (xc, rows["l1"]), (xc, rows["l2"] + h["l2"]), "#0072b2")
    _arrow(ax, (xc, rows["l2"]), (xc, rows["l3"] + h["l3"]), "#0072b2")
    y2 = rows["l2"] + h["l2"] / 2
    _arrow(ax, (x_lev + w_lev, y2), (x_mid, y2), "#0072b2")
    _arrow(ax, (x_mid + w_mid, y2), (x_fdr, y2), "#0072b2")
    y3 = rows["l3"] + h["l3"] / 2
    _arrow(ax, (x_lev + w_lev, y3), (x_mid, y3), "#e69f00")
    _arrow(ax, (x_mid + w_mid, y3), (x_fdr, y3), "#e69f00")
    yt = rows["l1"] + h["l1"] / 2
    xf = x_fdr + w_fdr / 2
    _arrow(ax, (xf, rows["l2"] + h["l2"]), (xf, yt), "#c00000", style="-", ls="--")
    _arrow(ax, (xf, yt), (x_lev + w_lev, yt), "#c00000", ls="--")
    ax.text((x_lev + w_lev + xf) / 2, yt + 0.05, "voltages, cost, curtailment, acceptance", fontsize=fs,
            color="#c00000", ha="center", va="bottom")
    ax.text(x_obs, H - 0.02, "black: observations    blue: set points    orange: reactive set points    "
            "red dashed: measurements fed back", fontsize=fs, va="top")
    fig.savefig(out / "fig_architecture.pdf")
    plt.close(fig)


def fig_l3_loop(out: pathlib.Path) -> None:
    """Control flow of nflev.env.qcontrol.ReactiveController.correct, drawn at its
    printed size (one column); axis units are inches."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(FONT)
    W, H = 3.45, 1.72
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    fs = 7
    x0, w, hh = 0.62, 1.62, 0.22
    ys = {"pf": 1.58, "chk": 1.24, "hr": 0.90, "inj": 0.54, "re": 0.15}
    _box(ax, (x0, ys["pf"] - hh / 2), w, hh, "Solve AC power flow", "#ececec", fs)
    _box(ax, (x0, ys["chk"] - hh / 2), w, hh, "$\\min_n V_n < \\underline{V}+\\delta$ ?", "#fff4cc", fs)
    _box(ax, (x0, ys["hr"] - hh / 2), w, hh, "Reactive headroom left?", "#fff4cc", fs)
    _box(ax, (x0, ys["inj"] - 0.15), w, 0.30, "Inject $Q$: proportional step,\nthen secant step", "#fde6cf", fs)
    _box(ax, (x0, ys["re"] - hh / 2), w, hh, "Re-solve power flow", "#ececec", fs)
    xs, ws = 2.52, 0.91
    _box(ax, (xs, ys["inj"] - 0.22), ws, 0.44, "Curtail EV power,\nmost at the most\ndepressed bus", "#fbe0e0", fs)
    cx = x0 + w / 2
    _arrow(ax, (cx, ys["pf"] - hh / 2), (cx, ys["chk"] + hh / 2))
    _arrow(ax, (cx, ys["chk"] - hh / 2), (cx, ys["hr"] + hh / 2))
    _arrow(ax, (cx, ys["hr"] - hh / 2), (cx, ys["inj"] + 0.15))
    _arrow(ax, (cx, ys["inj"] - 0.15), (cx, ys["re"] + hh / 2))
    ax.text(cx + 0.04, (ys["chk"] + ys["hr"]) / 2, "yes", fontsize=fs, va="center")
    ax.text(cx + 0.04, (ys["hr"] + ys["inj"] + 0.04) / 2, "yes", fontsize=fs, va="center")
    _arrow(ax, (x0 + w, ys["chk"]), (x0 + w + 0.3, ys["chk"]))
    ax.text(x0 + w + 0.33, ys["chk"], "no: done", fontsize=fs, va="center")
    ax.annotate("", xy=(xs + ws / 2, ys["inj"] + 0.22), xytext=(x0 + w, ys["hr"]),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, connectionstyle="angle,angleA=0,angleB=90,rad=0"))
    ax.text(x0 + w + 0.08, ys["hr"] + 0.03, "no", fontsize=fs, va="bottom")
    ax.annotate("", xy=(x0 + w, ys["re"]), xytext=(xs + ws / 2, ys["inj"] - 0.22),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, connectionstyle="angle,angleA=90,angleB=0,rad=0"))
    ax.annotate("", xy=(x0, ys["chk"]), xytext=(x0, ys["re"]),
                arrowprops=dict(arrowstyle="-|>", lw=0.8,
                                connectionstyle="arc,angleA=180,angleB=180,armA=8,armB=8,rad=0"))
    ax.text(0.01, (ys["chk"] + ys["re"]) / 2, "repeat:\n$\\leq$8 power\nflows (+12\nwhen\ncurtailing)",
            fontsize=fs, va="center", linespacing=1.1)
    fig.savefig(out / "fig_l3_loop.pdf")
    plt.close(fig)


def fig_pipeline(cfg: dict, out: pathlib.Path, n_seeds: int, n_aux: int) -> None:
    """Study pipeline from public data to every reported number (reproduce.py), drawn
    at its printed size (one column); counts are read from the configuration."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update(FONT)
    W, H = 3.45, 2.42
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    fs = 6.6
    d, ev, tr = cfg["data"], cfg["evaluation"], cfg["training"]
    yrs = lambda k: ", ".join(str(y) for y in d[k])  # noqa: E731
    bands = [
        ("Data", "#dfe9f7", "#2a5a9a",
         [f"ENTSO-E prices\n{yrs('alt_regime_years')}, {yrs('train_years')}, {yrs('test_years')}",
          "Pecan Street\nhousehold load", "ACN-Data\nsessions", "MATPOWER\n33-/69-bus"]),
        ("Simulation", "#ece5f5", "#5b3f8c",
         [f"Fleets\n({cfg['aggregators']['households']} households)", "Driver acceptance\n(Fishbein model)",
          f"AC power flow\nevery {cfg['simulation']['resolution_s']} s"]),
        ("Training", "#fbe8d6", "#a2561b",
         [f"Curriculum,\n{tr['episodes']} episodes", f"{n_seeds} seeds\n({n_aux} for ablations)",
          "Tariff calibration\n(training days)"]),
        ("Evaluation", "#dcefe2", "#2c7a4b",
         [f"{ev['episodes']} held-out\ntest days, paired", f"S1–S{len(ev['scenarios'])},\n8 baselines (+L3)",
          "Bootstrap CI,\nHolm–Wilcoxon"]),
        ("Outputs", "#f8dede", "#a12a2a",
         ["Tables, figures and number macros; every claim checked on the data\n(one command: reproduce.py)"]),
    ]
    x0, lab_w, gap, top = 0.03, 0.5, 0.06, H - 0.02
    bh = [0.43, 0.42, 0.42, 0.42, 0.32]
    y = top
    centers = []
    for (name, tint, edge, boxes), h in zip(bands, bh):
        y -= h
        ax.add_patch(FancyBboxPatch((x0, y), W - 2 * x0, h, boxstyle="round,pad=0,rounding_size=0.04",
                                    fc=tint, ec="none"))
        ax.text(x0 + 0.04, y + h / 2, name, fontsize=7, fontweight="bold", color=edge, va="center", ha="left")
        n = len(boxes)
        bx0 = x0 + lab_w + 0.08
        bw = (W - x0 - 0.05 - bx0 - (n - 1) * 0.05) / n
        for i, t in enumerate(boxes):
            bx = bx0 + i * (bw + 0.05)
            ax.add_patch(FancyBboxPatch((bx, y + 0.05), bw, h - 0.1, boxstyle="round,pad=0,rounding_size=0.03",
                                        fc="white", ec=edge, lw=0.7))
            ax.text(bx + bw / 2, y + h / 2, t, fontsize=fs, ha="center", va="center", linespacing=1.1)
        centers.append((y, h))
        y -= gap
    for (y1, h1), (y2, h2) in zip(centers[:-1], centers[1:]):
        ax.annotate("", xy=(W / 2 + 0.25, y2 + h2), xytext=(W / 2 + 0.25, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#52514e", lw=0.9))
    fig.savefig(out / "fig_pipeline.pdf")
    plt.close(fig)


def build_diagrams(cfg: dict, out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig_feeder(cfg, "ieee33", out)
    fig_feeder(cfg, "ieee69", out)
    fig_architecture(out)
    fig_l3_loop(out)
