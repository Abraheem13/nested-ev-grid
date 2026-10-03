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


def fig_feeder(cfg: dict, network: str, out: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    for i in range(1, net.n_bus):
        (x0, y0), (x1, y1) = pos[net.parent[i]], pos[i]
        ax.plot([x0, x0, x1], [y0, y1, y1], color="0.35", lw=0.7, zorder=1)
    xs = np.array([pos[i][0] for i in range(net.n_bus)])
    ys = np.array([pos[i][1] for i in range(net.n_bus)])
    ax.scatter(xs, ys, s=7, color="k", zorder=2)
    ax.scatter(xs[agg], ys[agg], s=60, facecolor="none", edgecolor="#c00000", lw=1.0, zorder=3)
    for k, b in enumerate(agg):
        ax.annotate(f"A{k + 1} (bus {b + 1})", (xs[b], ys[b]), xytext=(0, 6), textcoords="offset points",
                    fontsize=5.5, ha="center", color="#c00000")
    ax.scatter([xs[0]], [ys[0]], marker="s", s=40, color="#0072b2", zorder=3)
    ax.annotate("substation", (xs[0], ys[0]), xytext=(0, -9), textcoords="offset points", fontsize=5.5,
                ha="center", color="#0072b2")
    ax.set_axis_off()
    fig.tight_layout(pad=0.1)
    fig.savefig(out / f"fig_feeder_{network}.pdf")
    plt.close(fig)


def _box(ax, xy, w, h, text, fc, fs=6.5):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec="k", lw=0.6))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, a, b, color="k", style="-|>", ls="-"):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle=style, color=color, lw=0.8, linestyle=ls))


def fig_architecture(out: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.16, 2.75))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 4.45)
    ax.set_axis_off()
    fs = 5.5
    fl = 5.9
    rows = {"l1": 3.25, "l2": 1.85, "l3": 0.45}
    h = {"l1": 0.8, "l2": 1.0, "l3": 0.8}
    _box(ax, (0.05, rows["l1"]), 2.4, h["l1"], "System state (11-d)\nmean voltage, substation $P$, $Q$,\nprice now and 12 h ahead, fleet,\nline and substation loading", "#eef3fb", fs)
    _box(ax, (0.05, rows["l2"]), 2.4, h["l2"], "Aggregator state (28-d)\nbus voltage, time, 12 h of prices,\ncorridor, need by laxity,\nacceptance, plan set point $u^0_k$", "#eef3fb", fs)
    _box(ax, (0.05, rows["l3"]), 2.4, h["l3"], "Local measurements\nbus voltages, charger power,\nvehicle needs", "#eef3fb", fs)
    _box(ax, (2.8, rows["l1"]), 3.0, h["l1"], "Level 1: DSO pricing (PPO, 1 h)\nretail corridor $[p^{\\min}_t, p^{\\max}_t]$", "#d6e6f8", fl)
    _box(ax, (2.8, rows["l2"]), 3.0, h["l2"], "Level 2: aggregators (DDPG, 15 min)\nshared actor-critic,\nresidual on the plan $u^0_k$:\nset point $u_k$, execution price $p_k$", "#dceedd", fl)
    _box(ax, (2.8, rows["l3"]), 3.0, h["l3"], "Level 3 (non-parametric)\n3a: price acceptance (15 min)\n3b: reactive correction (60 s)", "#fde6cf", fl)
    _box(ax, (6.2, rows["l2"]), 2.05, h["l2"], "Feasibility layer\nleast-laxity-first allocation\n+ deadline guard\n$\\sum_i c_i \\leq P^{\\mathrm{cap}}_k$, $c_i \\leq \\bar c_i$", "#f3f3f3", fs)
    _box(ax, (6.2, rows["l3"]), 2.05, h["l3"], "$Q_i^{\\max}=\\sqrt{S_i^2-P_i^2}$\nmeasured-sensitivity step\ncurtailment fallback", "#fbe0e0", fs)
    _box(ax, (8.65, 1.05), 1.7, 2.3, "Radial feeder\nIEEE 33/69-bus\nAC power flow\nevery 60 s", "#ececec", fl)
    for key in rows:
        y = rows[key] + h[key] / 2
        _arrow(ax, (2.45, y), (2.8, y))
    _arrow(ax, (4.3, rows["l1"]), (4.3, rows["l2"] + h["l2"]), "#0072b2")
    _arrow(ax, (4.3, rows["l2"]), (4.3, rows["l3"] + h["l3"]), "#0072b2")
    _arrow(ax, (5.8, rows["l2"] + 0.5), (6.2, rows["l2"] + 0.5), "#0072b2")
    _arrow(ax, (8.25, rows["l2"] + 0.5), (8.65, rows["l2"] + 0.5), "#0072b2")
    _arrow(ax, (5.8, rows["l3"] + 0.4), (6.2, rows["l3"] + 0.4), "#e69f00")
    _arrow(ax, (8.25, rows["l3"] + 0.4), (8.65, 1.3), "#e69f00")
    _arrow(ax, (9.5, 3.35), (9.5, 3.85), "#c00000", style="-")
    _arrow(ax, (9.5, 3.85), (5.8, 3.85), "#c00000", ls="--")
    ax.text(7.4, 3.95, "measured voltages, cost, curtailment, acceptance", fontsize=fs, color="#c00000", ha="center")
    ax.text(0.05, 0.08, "blue: set points (top-down)     orange: reactive set points     red dashed: measurements (bottom-up)",
            fontsize=fs)
    fig.savefig(out / "fig_architecture.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_l3_loop(out: pathlib.Path) -> None:
    """Control flow of nflev.env.qcontrol.ReactiveController.correct."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.set_xlim(0, 6.2)
    ax.set_ylim(0, 5.6)
    ax.set_axis_off()
    fs = 6
    x0, w, hh = 1.15, 2.9, 0.62
    ys = {"pf": 5.0, "chk": 3.95, "hr": 2.9, "inj": 1.85, "re": 0.75}
    _box(ax, (x0, ys["pf"] - hh / 2), w, hh, "Solve AC power flow", "#ececec", fs)
    _box(ax, (x0, ys["chk"] - hh / 2), w, hh, "$\\min_n V_n < V_{\\min}+\\delta$ ?", "#fff4cc", fs)
    _box(ax, (x0, ys["hr"] - hh / 2), w, hh, "Reactive headroom left?", "#fff4cc", fs)
    _box(ax, (x0, ys["inj"] - hh / 2), w, hh + 0.1, "Inject $Q$: proportional first,\nthen measured sensitivity", "#fde6cf", fs)
    _box(ax, (x0, ys["re"] - hh / 2), w, hh, "Re-solve power flow", "#ececec", fs)
    _box(ax, (4.4, ys["inj"] - 0.45), 1.75, 0.9, "Curtail EV power,\nmost at the most\ndepressed bus\n(frees headroom)", "#fbe0e0", 5.5)
    cx = x0 + w / 2
    _arrow(ax, (cx, ys["pf"] - hh / 2), (cx, ys["chk"] + hh / 2))
    _arrow(ax, (cx, ys["chk"] - hh / 2), (cx, ys["hr"] + hh / 2))
    _arrow(ax, (cx, ys["hr"] - hh / 2), (cx, ys["inj"] + hh / 2 + 0.1))
    _arrow(ax, (cx, ys["inj"] - hh / 2), (cx, ys["re"] + hh / 2))
    ax.text(cx + 0.08, (ys["chk"] + ys["hr"]) / 2, "yes", fontsize=fs, va="center")
    ax.text(cx + 0.08, (ys["hr"] + ys["inj"]) / 2 + 0.05, "yes", fontsize=fs, va="center")
    _arrow(ax, (x0 + w, ys["chk"]), (x0 + w + 0.7, ys["chk"]))
    ax.text(x0 + w + 0.75, ys["chk"], "no: done", fontsize=fs, va="center")
    ax.annotate("", xy=(5.27, ys["inj"] + 0.45), xytext=(x0 + w, ys["hr"]),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, connectionstyle="angle,angleA=0,angleB=90,rad=0"))
    ax.text(x0 + w + 0.12, ys["hr"] + 0.1, "no", fontsize=fs)
    ax.annotate("", xy=(x0 + w, ys["re"]), xytext=(5.27, ys["inj"] - 0.45),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, connectionstyle="angle,angleA=90,angleB=0,rad=0"))
    ax.annotate("", xy=(x0, ys["chk"]), xytext=(x0, ys["re"]),
                arrowprops=dict(arrowstyle="-|>", lw=0.8,
                                connectionstyle="arc,angleA=180,angleB=180,armA=10,armB=10,rad=0"))
    ax.text(0.0, (ys["chk"] + ys["re"]) / 2, "repeat\n(at most\n8 power\nflows)", fontsize=5.5, va="center")
    fig.savefig(out / "fig_l3_loop.pdf", bbox_inches="tight")
    plt.close(fig)


def build_diagrams(cfg: dict, out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig_feeder(cfg, "ieee33", out)
    fig_feeder(cfg, "ieee69", out)
    fig_architecture(out)
    fig_l3_loop(out)
