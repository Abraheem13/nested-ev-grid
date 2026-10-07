"""Diagrams generated from code: the controller architecture, the Level-3
correction loop and the reproduction pipeline, drawn in plain black and white at
their printed size. layout_problems() checks every figure of the paper for text
that overlaps another text, a line, an arrow or a box edge."""
from __future__ import annotations

import pathlib


FONT = {"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix", "pdf.fonttype": 42}


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


def _rect(ax, x, y, w, h, text, fs=FS, spacing=1.15):
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec="black", lw=LW, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=spacing, zorder=3)
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


def _label(ax, x, y, text, ha="left", va="center", fs=FS - 0.5, color="black", spacing=1.2):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, linespacing=spacing, zorder=5)


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
    bw, bh = 1.68, 0.36                                   # level and environment boxes
    xc, xe = 1.16, 4.76                                   # their left edges
    gap = (xc + bw + 0.10 + xe - 0.10) / 2                # middle of the gap between the two boundaries
    pr = _rect(ax, 0.02, 1.24, 0.84, bh, "day-ahead\nprices")
    pl = _rect(ax, 0.02, 0.67, 0.84, bh, "planning\nprior $u^{0}_{k}$", spacing=1.45)
    l1 = _rect(ax, xc, 1.24, bw, bh, "Level 1: price corridor\n(PPO, hourly)")
    l2 = _rect(ax, xc, 0.67, bw, bh, "Level 2: residual dispatch\n(shared DDPG, 15 min)")
    l3 = _rect(ax, xc, 0.10, bw, bh, "Level 3: reactive correction\n(non-learned, 60 s)")
    dr = _rect(ax, xe, 1.24, bw - 0.12, bh, "drivers accept or\ndecline $p_k$ (2)")
    al = _rect(ax, xe, 0.67, bw - 0.12, bh, "least-laxity-first allocation\nwith deadline guard")
    fd = _rect(ax, xe, 0.10, bw - 0.12, bh, "radial feeder,\nAC power flow (1)")
    _boundary(ax, xc - 0.10, 0.02, bw + 0.20, 1.66, "nested controller", where="above")
    _boundary(ax, xe - 0.10, 0.02, bw + 0.08, 1.66, "simulated environment", where="above")
    _path(ax, [(pr["r"], pr["cy"]), (l1["l"], l1["cy"])])
    _path(ax, [(pr["cx"], pr["b"]), (pl["cx"], pl["t"])])
    _path(ax, [(pl["r"], pl["cy"]), (l2["l"], l2["cy"])])
    _path(ax, [(l1["cx"], l1["b"]), (l2["cx"], l2["t"])])
    _label(ax, l1["cx"] + 0.05, (l1["b"] + l2["t"]) / 2, "corridor $[p^{\\min}_t, p^{\\max}_t]$")
    yp, xv = l2["cy"] + 0.1, gap - 0.42                   # price: right, up, right into the drivers box
    _path(ax, [(l2["r"], yp), (xv, yp), (xv, dr["cy"]), (dr["l"], dr["cy"])])
    _label(ax, (xv + xe - 0.10) / 2, dr["cy"] + 0.04, "price $p_k$", ha="center", va="bottom")
    ys = l2["cy"] - 0.08
    _path(ax, [(l2["r"], ys), (al["l"], ys)])
    _label(ax, gap, ys - 0.04, "set point $u_k$", ha="center", va="top")
    _path(ax, [(dr["cx"], dr["b"]), (al["cx"], al["t"])])
    _label(ax, dr["cx"] + 0.05, (dr["b"] + al["t"]) / 2, "accepting vehicles")
    _path(ax, [(al["cx"], al["b"]), (fd["cx"], fd["t"])])
    _label(ax, al["cx"] + 0.05, (al["b"] + fd["t"]) / 2, "charging power $c_i$")
    _path(ax, [(l3["r"], fd["cy"] + 0.08), (fd["l"], fd["cy"] + 0.08)])
    _label(ax, gap, fd["cy"] + 0.12, "reactive power $Q_i$, curtailment", ha="center", va="bottom")
    _path(ax, [(fd["l"], fd["cy"] - 0.08), (l3["r"], fd["cy"] - 0.08)], dashed=True)
    _label(ax, gap, fd["cy"] - 0.12, "measured bus voltages", ha="center", va="top")
    yt, xf = 1.92, fd["r"] + 0.30
    _path(ax, [(fd["r"], fd["cy"]), (xf, fd["cy"]), (xf, yt), (l1["cx"], yt), (l1["cx"], l1["t"])], dashed=True)
    _label(ax, xf + 0.05, 0.95, "measured\nstates\n$s^{(1)}_{t}$, $s^{(2)}_{\\tau,k}$", va="center", spacing=1.45)
    save_checked(fig, out / "fig_architecture.pdf")
    import matplotlib.pyplot as plt
    plt.close(fig)


def fig_l3_loop(out: pathlib.Path) -> None:
    """Control flow of nflev.env.qcontrol.ReactiveController.correct (one column)."""
    fig, ax = _canvas(3.45, 1.6)
    x0, w, h = 0.70, 1.55, 0.18
    pf = _rect(ax, x0, 1.38, w, h, "solve AC power flow")
    ck = _rect(ax, x0, 1.065, w, h, "$\\min_n V_n < \\underline{V} + \\delta$ ?")
    hr = _rect(ax, x0, 0.75, w, h, "reactive headroom left?")
    inj = _rect(ax, x0, 0.365, w, 0.25, "inject $Q$: proportional step,\nthen secant step")
    rs = _rect(ax, x0, 0.05, w, h, "re-solve power flow")
    cu = _rect(ax, 2.55, 0.28, 0.86, 0.46, "curtail EV power,\nmost at the most\ndepressed bus")
    for a, b in ((pf, ck), (ck, hr), (hr, inj), (inj, rs)):
        _path(ax, [(a["cx"], a["b"]), (b["cx"], b["t"])])
    _label(ax, ck["cx"] + 0.05, (ck["b"] + hr["t"]) / 2, "YES", fs=5.8)
    _label(ax, hr["cx"] + 0.05, (hr["b"] + inj["t"]) / 2, "YES", fs=5.8)
    _path(ax, [(ck["r"], ck["cy"]), (ck["r"] + 0.32, ck["cy"])])
    _label(ax, ck["r"] + 0.36, ck["cy"], "NO: done", fs=5.8)
    _path(ax, [(hr["r"], hr["cy"]), (cu["cx"], hr["cy"]), (cu["cx"], cu["t"])])
    _label(ax, hr["r"] + 0.05, hr["cy"] + 0.03, "NO", va="bottom", fs=5.8)
    _path(ax, [(cu["cx"], cu["b"]), (cu["cx"], rs["cy"]), (rs["r"], rs["cy"])])
    _path(ax, [(rs["l"], rs["cy"]), (x0 - 0.12, rs["cy"]), (x0 - 0.12, ck["cy"]), (ck["l"], ck["cy"])])
    _label(ax, 0.02, (rs["cy"] + ck["cy"]) / 2, "repeat:\nat most 8\npower flows\n(+12 when\ncurtailing)",
           fs=FS - 0.5)
    save_checked(fig, out / "fig_l3_loop.pdf")
    import matplotlib.pyplot as plt
    plt.close(fig)


def fig_pipeline(cfg: dict, out: pathlib.Path, n_seeds: int, n_aux: int) -> None:
    """Study pipeline from public data to every reported number (one column);
    counts are read from the configuration and the stored runs."""
    d, ev, tr = cfg["data"], cfg["evaluation"], cfg["training"]
    yrs = lambda k: ", ".join(str(y) for y in d[k])  # noqa: E731
    fig, ax = _canvas(3.45, 2.06)
    x, w, h = 0.62, 2.45, 0.27
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
    save_checked(fig, out / "fig_pipeline.pdf")
    import matplotlib.pyplot as plt
    plt.close(fig)


def build_diagrams(cfg: dict, out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig_architecture(out)
    fig_l3_loop(out)


def layout_problems(fig, pad_px: float = 0.6, gap_pt: float = 1.5) -> list[str]:
    """Overlaps in a rendered figure: a text touching a drawn line, an arrow or a
    box edge, or closer than gap_pt to another text. Returns one message per
    problem (empty = clean). Grid lines and the legend frame are ignored; a text
    may sit inside its box."""
    import numpy as np
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, Rectangle
    from matplotlib.text import Text
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    grid, hidden = set(), set()
    for ax in fig.axes:
        grid |= set(ax.get_xgridlines()) | set(ax.get_ygridlines())
        for axis in (ax.xaxis, ax.yaxis):                 # tick labels outside the view are not drawn
            lo, hi = sorted(axis.get_view_interval())
            tol = 1e-9 * max(1.0, abs(hi - lo))
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not (lo - tol <= tick.get_loc() <= hi + tol):
                    hidden |= {tick.label1, tick.label2}
    texts = []
    for t in fig.findobj(Text):
        s = t.get_text().strip()
        if t.get_visible() and s and t.get_alpha() != 0 and t not in hidden:
            bb = t.get_window_extent(r)
            texts.append((s.replace("\n", " "), bb.x0 + pad_px, bb.y0 + pad_px, bb.x1 - pad_px, bb.y1 - pad_px))
    segs = []                               # (name, x0, y0, x1, y1, clip) in display units
    for ln in fig.findobj(Line2D):
        if ln in grid or not ln.get_visible() or ln.get_linestyle() in ("None", "", " "):
            continue
        cb = ln.get_clip_box() if ln.get_clip_on() else None      # data outside the axes is not drawn
        clip = None if cb is None else (cb.x0, cb.y0, cb.x1, cb.y1)
        v = ln.get_transform().transform(ln.get_xydata())
        segs += [("line", *v[i], *v[i + 1], clip) for i in range(len(v) - 1)]
    for p in fig.findobj(FancyArrowPatch):
        for path in p._get_path_in_displaycoord()[0]:
            v = path.vertices
            segs += [("arrow", *v[i], *v[i + 1], None) for i in range(len(v) - 1)]
    for p in fig.findobj(Rectangle):
        if not p.get_visible() or p.get_edgecolor()[3] == 0 or p.axes is None and p.figure is p:
            continue
        if p.get_linewidth() == 0:
            continue
        v = p.get_transform().transform(p.get_path().vertices)
        segs += [("box edge", *v[i], *v[i + 1], None) for i in range(len(v) - 1)]
    for c in fig.findobj(LineCollection):
        tr = c.get_transform()
        cb = c.get_clip_box() if c.get_clip_on() else None
        clip = None if cb is None else (cb.x0, cb.y0, cb.x1, cb.y1)
        for sg in c.get_segments():
            v = tr.transform(np.asarray(sg))
            segs += [("data line", *v[i], *v[i + 1], clip) for i in range(len(v) - 1)]

    def hits(seg, b):
        _, x0, y0, x1, y1, clip = seg
        bx0, by0, bx1, by1 = b
        if clip is not None:                            # only the visible part of a clipped line
            bx0, by0, bx1, by1 = max(bx0, clip[0]), max(by0, clip[1]), min(bx1, clip[2]), min(by1, clip[3])
            if bx0 >= bx1 or by0 >= by1:
                return False
        t0, t1, dx, dy = 0.0, 1.0, x1 - x0, y1 - y0     # Liang-Barsky clipping
        for p_, q_ in ((-dx, x0 - bx0), (dx, bx1 - x0), (-dy, y0 - by0), (dy, by1 - y0)):
            if abs(p_) < 1e-12:
                if q_ < 0:
                    return False
            else:
                t = q_ / p_
                if p_ < 0:
                    t0 = max(t0, t)
                else:
                    t1 = min(t1, t)
                if t0 > t1:
                    return False
        return True

    probs = []
    g = gap_pt * fig.dpi / 72 + 2 * pad_px             # required clearance between two texts
    for i, (s, *b) in enumerate(texts):
        for s2, *b2 in texts[i + 1:]:
            if b[0] < b2[2] + g and b2[0] < b[2] + g and b[1] < b2[3] + g and b2[1] < b[3] + g:
                probs.append(f"text '{s}' overlaps text '{s2}'")
        for seg in segs:
            if hits(seg, b):
                probs.append(f"text '{s}' touches a {seg[0]}")
                break
    return probs


def save_checked(fig, path: pathlib.Path, **kw) -> None:
    """Save a figure of the paper; refuses to write one in which a text overlaps."""
    probs = layout_problems(fig)
    if probs:
        raise RuntimeError(f"{path.name}: overlapping text: " + "; ".join(probs))
    fig.savefig(path, **kw)
