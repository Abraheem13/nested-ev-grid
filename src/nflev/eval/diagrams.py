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
    fig, ax = plt.subplots(figsize=(7.16, 2.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.set_axis_off()
    _box(ax, (0.1, 3.15), 2.0, 0.8, "System state (11-d)\n$\\bar V$, $P$, $Q$, price now/ahead,\nfleet, line & substation loading", "#eef3fb", 5.6)
    _box(ax, (0.1, 1.75), 2.0, 0.95, "Aggregator state (27-d)\nbus voltage, time, 12 h day-ahead\nprices, corridor, energy need\nby laxity, acceptance", "#eef3fb", 5.6)
    _box(ax, (0.1, 0.35), 2.0, 0.8, "Local measurements\nbus voltages, charger\npower, vehicle needs", "#eef3fb", 5.6)
    _box(ax, (2.6, 3.15), 2.6, 0.8, "Level 1: DSO (PPO, hourly)\nretail price corridor $[p^{\\min}_t, p^{\\max}_t]$", "#d6e6f8")
    _box(ax, (2.6, 1.75), 2.6, 0.95, "Level 2: aggregators (DDPG, 15 min,\nshared actor-critic)\npower set point $u_k$ + execution price $\\phi_k$", "#dceedd")
    _box(ax, (2.6, 0.35), 2.6, 0.8, "Level 3 (non-parametric)\n3a: price acceptance (15 min)\n3b: reactive correction (60 s)", "#fde6cf")
    _box(ax, (5.7, 1.75), 1.9, 0.95, "Feasibility layer\nleast-laxity-first allocation\n+ deadline guard\n$\\sum c_i \\leq P^{\\mathrm{cap}}_k$, $c_i \\leq \\bar c_i$", "#f3f3f3", 5.6)
    _box(ax, (8.0, 1.2), 1.9, 2.1, "Radial feeder\nIEEE 33 / 69-bus\nfull AC power flow\n(backward/forward\nsweep) every 60 s", "#ececec", 6)
    _box(ax, (5.7, 0.35), 1.9, 0.8, "$Q_i^{\\max}=\\sqrt{S_i^2-P_i^2}$\nNewton closure\ncurtailment fallback", "#fbe0e0", 5.6)
    for y in (3.55, 2.22, 0.75):
        _arrow(ax, (2.1, y), (2.6, y))
    _arrow(ax, (3.9, 3.15), (3.9, 2.7), "#0072b2")
    _arrow(ax, (5.2, 2.22), (5.7, 2.22), "#0072b2")
    _arrow(ax, (7.6, 2.22), (8.0, 2.22), "#0072b2")
    _arrow(ax, (5.2, 0.75), (5.7, 0.75), "#e69f00")
    _arrow(ax, (7.6, 0.75), (8.0, 1.4), "#e69f00")
    _arrow(ax, (8.95, 3.3), (5.2, 3.55), "#c00000", ls="--")
    _arrow(ax, (8.0, 1.75), (5.2, 1.95), "#c00000", ls="--")
    ax.text(6.6, 3.62, "measured voltages, cost, curtailment", fontsize=5.5, color="#c00000", ha="center")
    ax.text(0.1, 0.05, "blue: set points (top-down)   red dashed: measurement feedback (bottom-up)   orange: reactive set points",
            fontsize=5.5)
    fig.savefig(out / "fig_architecture.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_l3_loop(out: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.4, 3.3))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 6.3)
    ax.set_axis_off()
    steps = [(5.6, "Solve AC power flow", "#ececec"),
             (4.6, "$\\min_n V_n < V_{\\min}+\\delta$ ?", "#fff4cc"),
             (3.6, "Inject $Q$ from inverter headroom\n(proportional, then measured-\nsensitivity Newton step)", "#fde6cf"),
             (2.5, "Re-solve power flow", "#ececec"),
             (1.5, "$Q$ exhausted ?", "#fff4cc"),
             (0.5, "Curtail EV power at the most\ndepressed buses (frees $Q$)", "#fbe0e0")]
    for y, t, c in steps:
        _box(ax, (0.7, y - 0.35), 2.6, 0.7, t, c, 6)
    for (y1, _, _), (y2, _, _) in zip(steps[:-1], steps[1:]):
        _arrow(ax, (2.0, y1 - 0.35), (2.0, y2 + 0.35))
    ax.text(3.35, 4.6, "no: done", fontsize=6, va="center")
    ax.annotate("", xy=(0.7, 4.6), xytext=(0.7, 2.5), arrowprops=dict(arrowstyle="-|>", lw=0.8,
                connectionstyle="arc,angleA=180,angleB=180,armA=12,armB=12,rad=0"))
    ax.text(0.05, 3.5, "repeat\n(at most\n8 solves)", fontsize=5.5, va="center")
    fig.savefig(out / "fig_l3_loop.pdf", bbox_inches="tight")
    plt.close(fig)


def build_diagrams(cfg: dict, out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig_feeder(cfg, "ieee33", out)
    fig_feeder(cfg, "ieee69", out)
    fig_architecture(out)
    fig_l3_loop(out)
