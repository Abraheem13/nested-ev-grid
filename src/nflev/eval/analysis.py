"""Turn artifacts/eval/*.csv into every table, figure and number in the paper.

Statistics
  * Learned methods are trained with several seeds. For each evaluation day the
    metric is first averaged over seeds, so every method contributes one value
    per day (50 held-out days) and all comparisons are paired by day.
  * Means are reported with 95 % bootstrap confidence intervals over days
    (10 000 resamples, fixed RNG seed).
  * Pairwise tests: Wilcoxon signed-rank on the per-day differences, Holm
    correction across the comparison family.
Outputs (paper/generated/): tab_*.tex, numbers.tex (\\newcommand macros used in
the text) and fig_*.pdf.
"""
from __future__ import annotations

import glob
import json
import math
import pathlib

import numpy as np
import pandas as pd
from scipy import stats

from .methods import label, short_label

ROOT = pathlib.Path(__file__).resolve().parents[3]
BOOT = 10_000
RULE_ORDER = ["uncoordinated", "uncoordinated+L3", "tou", "tou+L3", "price_aware", "price_aware+L3",
              "lp_opf", "lp_opf+L3"]
LEARN_ORDER = ["flat_ddpg", "flat_ddpg+L3", "ppo_lag", "ppo_lag+L3", "cpo", "cpo+L3", "hrl", "hrl+L3"]
ORDER = RULE_ORDER + LEARN_ORDER + ["nested"]
METRICS = ["cost_eur", "cost_per_kwh", "service_quality", "violation_rate_pct", "min_voltage_pu",
           "curtailed_kwh", "q_activation_pct", "peak_ev_kw", "retail_price_paid", "energy_delivered_kwh",
           "unmet_kwh"]


# ------------------------------------------------------------------ loading
def load(art: pathlib.Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(art / "eval" / "*.csv")))
    if not files:
        raise SystemExit(f"no evaluation files in {art / 'eval'}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def per_day(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (variant, method, scenario, fleet, network, split, episode):
    metrics averaged over training seeds."""
    keys = ["variant", "method", "scenario", "fleet", "network", "split", "episode"]
    return df.groupby(keys, as_index=False)[METRICS].mean()


def boot_ci(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    m = x.mean()
    if len(x) < 2 or np.allclose(x, x[0]):
        return m, m, m
    idx = rng.integers(0, len(x), (BOOT, len(x)))
    bm = x[idx].mean(axis=1)
    return m, float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


def summarise(d: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for k, g in d.groupby(["variant", "method", "scenario", "fleet", "network", "split"]):
        r = dict(zip(["variant", "method", "scenario", "fleet", "network", "split"], k), n=len(g))
        for m in METRICS:
            r[m], r[m + "_lo"], r[m + "_hi"] = boot_ci(g[m].values, rng)
        rows.append(r)
    return pd.DataFrame(rows)


def seed_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Std over training seeds of the per-seed mean (learned methods only)."""
    g = df[df.train_seed >= 0].groupby(["variant", "method", "scenario", "fleet", "network", "split",
                                        "train_seed"])[METRICS].mean()
    return g.groupby(level=[0, 1, 2, 3, 4, 5]).std(ddof=1).reset_index()


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return list(adj)


def paired_tests(d: pd.DataFrame, ref: str, others: list[str], metric: str, scenario: str,
                 variant: str = "main") -> pd.DataFrame:
    sel = d[(d.variant == variant) & (d.scenario == scenario) & (d.fleet == "residential")
            & (d.network == "ieee33") & (d.split == "test")]
    a = sel[sel.method == ref].set_index("episode")[metric]
    rows, ps = [], []
    rng = np.random.default_rng(1)
    for o in others:
        b = sel[sel.method == o].set_index("episode")[metric]
        common = a.index.intersection(b.index)
        diff = (a.loc[common] - b.loc[common]).values
        if len(diff) == 0:
            continue
        p = 1.0 if np.allclose(diff, 0) else float(stats.wilcoxon(diff, zero_method="zsplit").pvalue)
        m, lo, hi = boot_ci(diff, rng)
        rows.append({"method": o, "mean_diff": m, "lo": lo, "hi": hi, "p_raw": p,
                     "wins": int((diff < 0).sum()), "n": len(diff)})
        ps.append(p)
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(ps)
    return out


# ----------------------------------------------------------------- formatting
def fmt(x, nd):
    return "--" if pd.isna(x) else f"{x:.{nd}f}"


def fmt_ci(r, m, nd):
    return f"{fmt(r[m], nd)} {{\\scriptsize[{fmt(r[m + '_lo'], nd)}, {fmt(r[m + '_hi'], nd)}]}}"


def fmt_p(p):
    return "$<\\!0.001$" if p < 0.001 else f"{p:.3f}"


def macro_name(*parts) -> str:
    """'uncoordinated+L3 S1 cost' -> 'GUncoordinatedLThreeSOneCost' (LaTeX macro names
    may contain letters only; the G prefix marks generated values)."""
    import re
    words = re.split(r"[^A-Za-z0-9]+", " ".join(str(p) for p in parts))
    s = "G" + "".join(w[:1].upper() + w[1:] for w in words if w)
    for d, w in zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven",
                                   "Eight", "Nine"]):
        s = s.replace(d, w)
    return s


class Numbers:
    def __init__(self):
        self.lines = ["% Auto-generated by nflev.eval.analysis -- do not edit by hand."]
        self.names = set()

    def add(self, name, value, nd=2, rnd="nearest"):
        """rnd="up"/"down" rounds outward for bounds stated in the text as
        "at most"/"more than" (up) or "at least" (down)."""
        name = macro_name(name)
        if name in self.names:
            raise ValueError(f"duplicate macro {name}")
        self.names.add(name)
        if rnd != "nearest" and not isinstance(value, str):
            f = 10 ** nd
            value = (math.ceil(value * f - 1e-9) if rnd == "up" else math.floor(value * f + 1e-9)) / f
        v = value if isinstance(value, str) else (f"{int(value)}" if nd == 0 else f"{value:.{nd}f}")
        self.lines.append(f"\\newcommand{{\\{name}}}{{{v}}}")

    def write(self, path):
        path.write_text("\n".join(self.lines) + "\n")


# --------------------------------------------------------------------- tables
def row(s, variant, method, scenario, fleet="residential", network="ieee33", split="test"):
    r = s[(s.variant == variant) & (s.method == method) & (s.scenario == scenario) & (s.fleet == fleet)
          & (s.network == network) & (s.split == split)]
    return None if r.empty else r.iloc[0]


def table_main(s, scenario, path):
    lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r"Method & Cost (\euro) & \euro/kWh & Retail & SQ & Viol.\ (\%) & $V_{\min}$ & Curt.\ (kWh) \\",
             r"\midrule"]
    for i, m in enumerate(ORDER):
        r = row(s, "main", m, scenario)
        if r is None:
            continue
        if m in ("flat_ddpg", "nested"):
            lines.append(r"\midrule")
        name = label(m)
        cells = [fmt_ci(r, "cost_eur", 1), fmt(r["cost_per_kwh"], 4), fmt(r["retail_price_paid"], 3),
                 fmt(r["service_quality"], 3),
                 fmt(r["violation_rate_pct"], 2), fmt(r["min_voltage_pu"], 4), fmt(r["curtailed_kwh"], 1)]
        if m == "nested":
            name = r"\textbf{" + name + "}"
            cells = [r"\textbf{" + c + "}" for c in cells]
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def table_scenarios(s, methods, scenarios, path):
    cols = "".join("rr" for _ in scenarios)
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{sc}}}" for sc in scenarios)
    sub = " & ".join(r"Cost & Viol." for _ in scenarios)
    lines = [rf"\begin{{tabular}}{{@{{}}l{cols}@{{}}}}", r"\toprule", f"Method & {head} \\\\",
             f" & {sub} \\\\", r"\midrule"]
    for m in methods:
        cells = []
        for sc in scenarios:
            r = row(s, "main", m, sc)
            cells += ["--", "--"] if r is None else [fmt(r["cost_eur"], 0), fmt(r["violation_rate_pct"], 2)]
        lines.append(label(m) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def table_stats(t_cost, t_sq, path):
    lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Nested vs. & $\Delta$Cost (\euro) & $p_{\mathrm{Holm}}$ & Days & $\Delta$SQ & $p_{\mathrm{Holm}}$ \\",
             r"\midrule"]
    sq = t_sq.set_index("method")
    for _, r in t_cost.iterrows():
        q = sq.loc[r["method"]]
        lines.append(f"{short_label(r['method'])} & {r['mean_diff']:+.1f} {{\\scriptsize[{r['lo']:+.1f}, {r['hi']:+.1f}]}}"
                     f" & {fmt_p(r['p_holm'])} & {int(r['wins'])}/{int(r['n'])} & {q['mean_diff']:+.4f}"
                     f" & {fmt_p(q['p_holm'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


ABL_LABELS = {"main": "Complete framework", "abl_no_prior": "No planning prior",
              "abl_no_l1": "No Level 1", "abl_flat_timescale": "Single timescale",
              "abl_no_behavior": "No behaviour model", "abl_no_l3": "No Level 3 (trained)",
              "ablation": "L3 removed at test", "abl_no_curriculum": "No curriculum",
              "abl_no_guard": "No deadline guard", "abl_proportional": "Proportional allocation",
              "abl_llf": "Least-laxity-first allocation"}


def table_ablation(s, scenarios, path):
    lines = [r"\begin{tabular}{@{}l" + "rrr" * len(scenarios) + "@{}}", r"\toprule",
             "Configuration & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{sc}}}" for sc in scenarios) + r" \\",
             " & " + " & ".join(r"Cost & SQ & Viol." for _ in scenarios) + r" \\", r"\midrule"]
    for v, lab in ABL_LABELS.items():
        cells, any_row = [], False
        for sc in scenarios:
            m = "nested-noL3" if v in ("abl_no_l3", "ablation") else "nested"
            r = row(s, v, m, sc)
            if r is None:
                cells += ["--"] * 3
            else:
                any_row = True
                cells += [fmt(r["cost_eur"], 0), fmt(r["service_quality"], 3), fmt(r["violation_rate_pct"], 2)]
        if any_row:
            lines.append(lab + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


GENERAL = [("main", "residential", "ieee33", "test", "33-bus, residential"),
           ("general", "residential", "ieee69", "test", "69-bus, residential"),
           ("general", "acn_caltech", "ieee33", "test", "33-bus, ACN Caltech"),
           ("general", "acn_jpl", "ieee33", "test", "33-bus, ACN JPL"),
           ("regime", "residential", "ieee33", "alt", "33-bus, 2019 prices")]


def table_general(s, path):
    """One row per setting at S3: daily cost of the reference methods, and
    service quality / violations of uncoordinated charging and the nested controller."""
    lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{4}{c}{Daily cost (\euro)} & \multicolumn{2}{c}{Viol.\ (\%)} & SQ \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(lr){8-8}",
             r"Setting & Unc. & PA+L3 & LP$^\ast$+L3 & Nested & Unc. & Nested & Nested \\", r"\midrule"]
    for variant, fleet, net, split, name in GENERAL:
        g = {m: row(s, variant, m, "S3", fleet, net, split)
             for m in ("uncoordinated", "price_aware+L3", "lp_opf+L3", "nested")}
        if g["nested"] is None:
            continue
        c = lambda m, k, nd: "--" if g[m] is None else fmt(g[m][k], nd)  # noqa: E731
        lines.append(f"{name} & {c('uncoordinated', 'cost_eur', 1)} & {c('price_aware+L3', 'cost_eur', 1)} & "
                     f"{c('lp_opf+L3', 'cost_eur', 1)} & {c('nested', 'cost_eur', 1)} & "
                     f"{c('uncoordinated', 'violation_rate_pct', 2)} & {c('nested', 'violation_rate_pct', 2)} & "
                     f"{c('nested', 'service_quality', 3)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def table_stress(s, path):
    methods = ["uncoordinated", "tou", "uncoordinated+L3", "tou+L3", "price_aware+L3", "lp_opf+L3", "nested"]
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"Method & \multicolumn{2}{c}{S6} & \multicolumn{4}{c}{S7} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-7}",
             r" & Viol. & $V_{\min}$ & Viol. & Curt. & Unmet & SQ \\", r"\midrule"]
    for m in methods:
        a, b = row(s, "main", m, "S6"), row(s, "main", m, "S7")
        cells = (["--"] * 2 if a is None else [fmt(a["violation_rate_pct"], 2), fmt(a["min_voltage_pu"], 3)]) + \
                (["--"] * 4 if b is None else [fmt(b["violation_rate_pct"], 2), fmt(b["curtailed_kwh"], 0),
                                              fmt(b["unmet_kwh"], 0), fmt(b["service_quality"], 3)])
        lines.append(short_label(m) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def table_sensitivity(s, path):
    rows = [("sens_delta0.0005", r"$\delta = 0.0005$"), ("main", r"Default"),
            ("sens_delta0.002", r"$\delta = 0.002$"), ("sens_S10", r"$S_i = 10$ kVA"), ("sens_S14", r"$S_i = 14$ kVA")]
    lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Setting & Cost (\euro) & SQ & Viol.\ (\%) & $Q$ act.\ (\%) & Curt.\ (kWh) \\", r"\midrule"]
    for v, lab in rows:
        r = row(s, v, "nested", "S3")
        if r is None:
            continue
        lines.append(f"{lab} & {fmt(r['cost_eur'], 1)} & {fmt(r['service_quality'], 3)} & "
                     f"{fmt(r['violation_rate_pct'], 3)} & {fmt(r['q_activation_pct'], 2)} & {fmt(r['curtailed_kwh'], 1)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def table_compute(art, path):
    rows = []
    for d in sorted((art / "runs").glob("*")):
        log = d / "train_log.csv"
        if not log.exists():
            continue
        method, fleet, net, abl, seed = d.name.rsplit("_", 4)[0], *d.name.rsplit("_", 4)[1:]
        lg = pd.read_csv(log)
        rows.append({"run": d.name, "method": d.name.split("_residential")[0].split("_acn")[0],
                     "fleet": fleet, "network": net, "ablation": abl,
                     "episodes": len(lg), "wall_s_ep": lg.wall_s.mean(), "total_h": lg.wall_s.sum() / 3600})
    df = pd.DataFrame(rows)
    main = df[(df.fleet == "residential") & (df.network == "ieee33") & (df.ablation == "none")]
    g = main.groupby("method").agg(runs=("run", "count"), episodes=("episodes", "mean"),
                                   wall=("wall_s_ep", "mean"), total=("total_h", "mean"))
    names = {"nested": "Nested (proposed)", "flat": "Flat DDPG", "ppo": "PPO-Lagrangian", "cpo": "CPO", "hrl": "Hierarchical RL"}
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Method & Seeds & Episodes & s / episode & h / run \\", r"\midrule"]
    for m, r in g.iterrows():
        lines.append(f"{names.get(m.split('_')[0], m)} & {int(r['runs'])} & {int(r['episodes'])} & "
                     f"{r['wall']:.2f} & {r['total']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    return df


DECOMP_ROWS = [("main", "price_aware+L3"), ("decomp", "plan+L3"), ("decomp", "nested-planprice"),
               ("decomp", "nested-flatprice"), ("main", "nested")]


def table_decomp(s, path):
    """What the learned levels add: the plan executed without learning, with the
    learned prices only, with the learned dispatch only, and the full controller."""
    lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{S3} & \multicolumn{4}{c}{S7} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-8}",
             r"Configuration & Cost & Retail & SQ & Cost & Viol. & Curt. & Unmet \\", r"\midrule"]
    for v, m in DECOMP_ROWS:
        a, b = row(s, v, m, "S3"), row(s, v, m, "S7")
        if a is None and b is None:
            continue
        cells = (["--"] * 3 if a is None else [fmt(a["cost_eur"], 1), fmt(a["retail_price_paid"], 3),
                                              fmt(a["service_quality"], 3)]) + \
                (["--"] * 4 if b is None else [fmt(b["cost_eur"], 1), fmt(b["violation_rate_pct"], 2),
                                              fmt(b["curtailed_kwh"], 0), fmt(b["unmet_kwh"], 0)])
        lines.append(short_label(m) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def table_hyper(cfg: dict, path: pathlib.Path) -> None:
    """Hyperparameters, read from configs/base.yaml (cannot drift from the code)."""
    t, d, pp, l2 = cfg["training"], cfg["training"]["ddpg"], cfg["training"]["ppo"], cfg["level2"]

    def thin(x):
        return f"{int(x):,}".replace(",", "\\,")
    rows = [
        ("Level 1 (PPO)", "actor/critic 256--128--64, Beta policy"),
        ("", f"lr {pp['lr']:g}, $\\gamma$ {pp['gamma']}, GAE $\\lambda$ {pp['gae_lambda']}, clip {pp['clip']}"),
        ("", f"{pp['epochs']} epochs per update, entropy {pp['entropy']:g}"),
        ("Level 2 (DDPG)", "actor/critic 256--128--64, shared, one-hot id"),
        ("", f"lr actor/critic {d['lr_actor']:g}/{d['lr_critic']:g}, $\\gamma$ {d['gamma']}, $\\tau$ {d['tau']}"),
        ("", f"batch {d['batch']}, buffer {thin(d['buffer'])}, warm-up {thin(d['warmup'])} transitions"),
        ("", f"noise {d['noise_start']}$\\to${d['noise_end']} (decay {d['noise_decay']}/episode)"),
        ("", f"residual scale $\\rho$ {l2.get('residual_scale', 1.0)}"),
        ("", f"PI multiplier: $K_p$ {l2['revenue_neutral']['kp']}, $K_i$ {l2['revenue_neutral']['ki']}, "
             f"EMA {l2['revenue_neutral']['ema']}, $\\omega\\in[{l2['revenue_neutral']['w_min']}, "
             f"{l2['revenue_neutral']['w_max']}]$"),
        ("Training", f"{t['episodes']} episodes; curriculum window {cfg['curriculum']['window']}"),
        ("Level 3", f"$\\delta$ {cfg['voltage']['correction_margin']} p.u., $S_i$ {cfg['reactive_power']['s_rated_kva']} kVA, "
                    f"$\\le${cfg['voltage']['max_correction_iters']} (+{cfg['voltage']['max_fallback_iters']} when curtailing) power flows"),
    ]
    lines = [r"\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{0.68\columnwidth}@{}}", r"\toprule"]
    lines += [f"{a} & {b} \\\\" for a, b in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


class Claims:
    """Qualitative statements made in the paper, evaluated on the data.
    scripts/check_paper.py fails if the text tags a claim (% claim: name) that
    is false or missing."""

    def __init__(self):
        self.c = {}

    def add(self, name: str, value: bool, detail: str = "") -> None:
        self.c[name] = {"holds": bool(value), "detail": detail}

    def write(self, path: pathlib.Path) -> None:
        path.write_text(json.dumps(self.c, indent=2))


# -------------------------------------------------------------------- figures
# Categorical palette (validated: lightness band, chroma, adjacent CVD and normal-vision
# separation); identity is also carried by marker and line style, and every value is in a table.
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
NEUTRAL = "#7f7f7f"
STYLE = {  # method -> (colour, line style, marker); colour follows the entity in every figure
    "nested": (PAL[0], "-", "o"), "price_aware+L3": (PAL[1], "--", "s"), "lp_opf+L3": (PAL[2], "--", "D"),
    "uncoordinated": (PAL[3], "--", "v"), "flat_ddpg": (PAL[4], ":", "^"), "ppo_lag": (PAL[5], ":", "<"),
    "cpo": (PAL[6], ":", ">"), "hrl": (PAL[7], ":", "P"), "tou": (NEUTRAL, "--", "x"),
}
PEN = [("S1", 40), ("S2", 67), ("S3", 80), ("S5", 100)]       # S4 is S3 with forecast error


def figures(s, d, art, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "pdf.fonttype": 42, "axes.linewidth": 0.6,
                         "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.0), gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.array([p for _, p in PEN])

    def line(ax, m, metric, lw=1.1):
        col, ls, mk = STYLE[m]
        rows = [row(s, "main", m, sc) for sc, _ in PEN]
        if any(r is None for r in rows):
            return None
        y = np.array([r[metric] for r in rows])
        lo = np.array([r[metric + "_lo"] for r in rows])
        hi = np.array([r[metric + "_hi"] for r in rows])
        ax.errorbar(x, y, yerr=[y - lo, hi - y], color=col, ls=ls, marker=mk, ms=4.2, lw=lw, elinewidth=0.6,
                    capsize=1.5, label=label(m), zorder=3 if m == "nested" else 2,
                    mec="white", mew=0.4)
        return y
    # (a) paired: daily cost relative to the perfect-foresight LP-OPF on the same day
    rng = np.random.default_rng(2)
    sel = d[(d.variant == "main") & (d.fleet == "residential") & (d.network == "ieee33") & (d.split == "test")]
    ref = sel[sel.method == "lp_opf+L3"].set_index(["scenario", "episode"]).cost_eur
    for m in ["uncoordinated", "flat_ddpg", "ppo_lag", "cpo", "hrl", "price_aware+L3", "nested"]:
        col, ls, mk = STYLE[m]
        mm = sel[sel.method == m].set_index(["scenario", "episode"]).cost_eur
        y, lo, hi = [], [], []
        for sc, _ in PEN:
            try:
                r = 100.0 * (mm.loc[sc] / ref.loc[sc] - 1.0)
            except KeyError:
                r = pd.Series(dtype=float)
            r = r.dropna()
            if r.empty:
                y.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                continue
            mval, l_, h_ = boot_ci(r.values, rng)
            y.append(mval); lo.append(l_); hi.append(h_)
        y, lo, hi = map(np.asarray, (y, lo, hi))
        axes[0].errorbar(x, y, yerr=[y - lo, hi - y], color=col, ls=ls, marker=mk, ms=4.2,
                         lw=1.8 if m == "nested" else 1.0, elinewidth=0.6, capsize=1.5, label=label(m),
                         zorder=3 if m == "nested" else 2, mec="white", mew=0.4)
    axes[0].axhline(0.0, color="#52514e", lw=0.6)
    axes[0].set_ylabel("Daily cost above LP-OPF$^\\ast$ (%)")
    axes[0].set_title("(a) Cost relative to LP-OPF$^\\ast$ (paired by day)",
                      fontsize=7.5, loc="left")
    for m in ["uncoordinated", "tou", "flat_ddpg", "ppo_lag", "cpo", "hrl"]:
        line(axes[1], m, "violation_rate_pct")
    zero = [row(s, "main", m, sc)["violation_rate_pct"] for m in
            ["nested", "uncoordinated+L3", "tou+L3", "price_aware+L3", "lp_opf+L3"]
            for sc, _ in PEN if row(s, "main", m, sc) is not None]
    if zero:
        axes[1].text(0.02, 0.96, f"nested and all +L3 controllers: max {max(zero):.2f}%",
                     transform=axes[1].transAxes, fontsize=6.5, va="top", color="#52514e")
    axes[1].set_ylabel("Violation rate (%)")
    axes[1].set_title("(b) Voltage violations without Level 3", fontsize=7.5, loc="left")
    for ax in axes:
        ax.set_xticks(x, [f"{p}%\n{sc}" for sc, p in PEN])
        ax.set_xlabel("EV penetration")
        ax.grid(axis="y", lw=0.3, color="#d9d8d4")
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = [], []
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.02), handlelength=2.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "fig_scenarios.pdf", bbox_inches="tight")
    plt.close(fig)

    runs = sorted((art / "runs").glob("*_residential_ieee33_none_s*"))
    if runs:
        fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.0))
        for name in ["flat_ddpg", "ppo_lag", "cpo", "hrl", "nested"]:
            c = STYLE[name][0]
            logs = [pd.read_csv(r / "train_log.csv") for r in runs if r.name.startswith(name + "_residential")]
            if not logs:
                continue
            n = min(len(lg) for lg in logs)
            cpk = np.mean([lg.cost_per_kwh.values[:n] for lg in logs], axis=0)
            sq = np.mean([lg.service_quality.values[:n] for lg in logs], axis=0)
            k = 25
            lw, ls = (1.6, "-") if name == "nested" else (0.9, STYLE[name][1])
            ax[0].plot(pd.Series(cpk).rolling(k, min_periods=1).mean(), color=c, lw=lw, ls=ls, label=label(name))
            ax[1].plot(pd.Series(sq).rolling(k, min_periods=1).mean(), color=c, lw=lw, ls=ls)
        ax[0].set(xlabel="Training episode", ylabel="Cost per delivered kWh (EUR)")
        ax[1].set(xlabel="Training episode", ylabel="Service quality")
        for a in ax:
            a.grid(axis="y", lw=0.3, color="#d9d8d4")
            a.spines[["top", "right"]].set_visible(False)
        fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False, fontsize=6.5,
                   bbox_to_anchor=(0.5, -0.02), handlelength=2.6)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        fig.savefig(out / "fig_training.pdf", bbox_inches="tight")
        plt.close(fig)


# ------------------------------------------------------------------ numbers
MACRO_METRICS = [("cost", "cost_eur", 1), ("cpk", "cost_per_kwh", 4), ("sq", "service_quality", 3),
                 ("viol", "violation_rate_pct", 2), ("vmin", "min_voltage_pu", 4),
                 ("curt", "curtailed_kwh", 1), ("qact", "q_activation_pct", 2),
                 ("retail", "retail_price_paid", 3), ("unmet", "unmet_kwh", 1)]
LEARNED_ALL = ["flat_ddpg", "flat_ddpg+L3", "ppo_lag", "ppo_lag+L3", "cpo", "cpo+L3", "hrl", "hrl+L3"]


def numbers(s, d, t_cost, t_sq, compute_df, art, out, cfg):
    N, C = Numbers(), Claims()
    default = ("residential", "ieee33", "test")
    for _, r in s.iterrows():                       # every summary row -> macros
        key = [r["method"], r["scenario"]]
        if r["variant"] != "main" or (r["fleet"], r["network"], r["split"]) != default:
            key = [r["variant"], r["method"], r["scenario"], r["fleet"], r["network"], r["split"]]
        for short, col, nd in MACRO_METRICS:
            N.add(" ".join(key + [short]), r[col], nd)
        N.add(" ".join(key + ["sq pct"]), 100 * r["service_quality"], 1)

    def get(m, sc, v="main"):
        return row(s, v, m, sc)

    scen = [sc for sc in ["S1", "S2", "S3", "S4", "S5"] if get("nested", sc) is not None]
    sav, gap_lp, gap_pa, gap_l = {}, {}, {}, {}
    for sc in scen:
        nest, unc = get("nested", sc), get("uncoordinated", sc)
        lp, pa = get("lp_opf+L3", sc), get("price_aware+L3", sc)
        if unc is not None:
            sav[sc] = 100 * (1 - nest["cost_eur"] / unc["cost_eur"])
            N.add(f"saving nested unc {sc}", sav[sc], 1)
        if lp is not None:
            gap_lp[sc] = 100 * (nest["cost_eur"] / lp["cost_eur"] - 1)
            N.add(f"gap nested lp {sc}", abs(gap_lp[sc]), 1)
        if pa is not None:
            gap_pa[sc] = 100 * (nest["cost_eur"] / pa["cost_eur"] - 1)
            N.add(f"gap nested pa {sc}", abs(gap_pa[sc]), 1)
        learned = [(m, get(m, sc)) for m in LEARNED_ALL if get(m, sc) is not None]
        if learned:
            best_m, best = min(learned, key=lambda x: x[1]["cost_eur"])
            gap_l[sc] = 100 * (1 - nest["cost_eur"] / best["cost_eur"])
            N.add(f"saving nested best learned {sc}", gap_l[sc], 1)
            N.add(f"best learned {sc}", label(best_m), 0)
    if sav:
        N.add("saving nested unc min", min(sav.values()), 1)
        N.add("saving nested unc max", max(sav.values()), 1)
    if gap_lp:
        N.add("gap nested lp max", max(gap_lp.values()), 1)
    if gap_l:
        N.add("saving nested best learned min", min(gap_l.values()), 1, rnd="down")
        N.add("saving nested best learned max", max(gap_l.values()), 1)

    def maxv(methods, metric="violation_rate_pct", scs=scen):
        vals = [get(m, sc)[metric] for m in methods for sc in scs if get(m, sc) is not None]
        return max(vals) if vals else float("nan")
    N.add("viol unc max", maxv(["uncoordinated"]), 2)
    N.add("viol tou max", maxv(["tou"]), 2)
    N.add("viol learned noL max", maxv([m for m in LEARNED_ALL if "+L3" not in m]), 2)
    N.add("viol l3 rules max", maxv(["uncoordinated+L3", "tou+L3", "price_aware+L3", "lp_opf+L3"]), 2)
    N.add("viol nested max", maxv(["nested"]), 2)

    # claims evaluated on the data (referenced in the text by % claim: name)
    C.add("nested_zero_viol_main", maxv(["nested"]) == 0.0, f"max {maxv(['nested'])}")
    C.add("nested_cheapest_learned_main", bool(gap_l) and min(gap_l.values()) > 0, json.dumps({k: round(float(v), 2) for k, v in gap_l.items()}))
    C.add("nested_cheaper_than_unc_main", bool(sav) and min(sav.values()) > 0, json.dumps({k: round(float(v), 2) for k, v in sav.items()}))
    C.add("nested_cheaper_than_tou_main", all(get("nested", sc)["cost_eur"] < get("tou+L3", sc)["cost_eur"]
                                              for sc in scen if get("tou+L3", sc) is not None))
    C.add("nested_cheaper_than_pa_main", bool(gap_pa) and max(gap_pa.values()) < 0, json.dumps({k: round(float(v), 2) for k, v in gap_pa.items()}))
    C.add("nested_above_lp_main", bool(gap_lp) and min(gap_lp.values()) > 0, json.dumps({k: round(float(v), 2) for k, v in gap_lp.items()}))
    ref_price = cfg["behavior"]["lambda_ref"]
    rp = {sc: float(get("nested", sc)["retail_price_paid"]) for sc in scen}
    if rp:
        N.add("nested retail min", min(rp.values()), 3)
        N.add("nested retail max", max(rp.values()), 3)
        N.add("ref price", ref_price, 2)
    C.add("nested_revenue_neutral_main", bool(rp) and max(abs(v - ref_price) for v in rp.values()) <= 0.01,
          json.dumps({k: round(v, 3) for k, v in rp.items()}))
    C.add("nested_retail_below_ref_main", bool(rp) and max(rp.values()) < ref_price - 0.01,
          json.dumps({k: round(v, 3) for k, v in rp.items()}))
    nonoracle = ["uncoordinated", "uncoordinated+L3", "tou", "tou+L3", "price_aware", "price_aware+L3",
                 "nested"] + LEARNED_ALL
    for sc in scen:
        costs = {m: get(m, sc)["cost_eur"] for m in ORDER if get(m, sc) is not None}
        if costs:
            C.add(f"lp_cheapest_{sc}", min(costs, key=costs.get) in ("lp_opf", "lp_opf+L3"), str(min(costs, key=costs.get)))
            no = {m: c for m, c in costs.items() if m in nonoracle}
            C.add(f"pa_cheapest_nonoracle_{sc}", min(no, key=no.get) in ("price_aware", "price_aware+L3"),
                  str(min(no, key=no.get)))
    C.add("pa_cheaper_than_nested_all_main", bool(gap_pa) and min(gap_pa.values()) > 0,
          json.dumps({k: round(float(v), 2) for k, v in gap_pa.items()}))
    C.add("tou_violates_main", maxv(["tou"]) > 0)
    if gap_pa:
        N.add("gap nested pa min", min(gap_pa.values()), 1)
        N.add("gap nested pa max", max(gap_pa.values()), 1)
    C.add("l3_zero_viol_rules_main", maxv(["uncoordinated+L3", "tou+L3", "price_aware+L3", "lp_opf+L3"]) == 0.0)
    C.add("unc_violates_main", maxv(["uncoordinated"]) > 0)
    l3_all = ["nested", "uncoordinated+L3", "tou+L3", "price_aware+L3", "lp_opf+L3"]
    C.add("s6_l3_zero_viol", maxv(l3_all, scs=["S6"]) == 0.0, f"max {maxv(l3_all, scs=['S6'])}")
    C.add("s7_l3_violations_remain", maxv(l3_all, scs=["S7"]) > 0, f"max {maxv(l3_all, scs=['S7'])}")
    nominal = [sc for sc in scen if sc != "S4"]           # S4 adds base-load forecast error
    l3_methods = ["uncoordinated+L3", "tou+L3", "price_aware+L3", "lp_opf+L3"]
    C.add("nested_zero_viol_nominal", maxv(["nested"], scs=nominal) == 0.0,
          f"max {maxv(['nested'], scs=nominal)}")
    C.add("l3_zero_viol_rules_nominal", maxv(l3_methods, scs=nominal) == 0.0, f"max {maxv(l3_methods, scs=nominal)}")
    C.add("l3_zero_viol_learned_nominal", maxv([m for m in LEARNED_ALL if "+L3" in m], scs=nominal) == 0.0)
    N.add("viol nested max S4", maxv(["nested"], scs=["S4"]), 3, rnd="up")
    # S4: are the remaining violations on days where the base load alone violates?
    sel = d[(d.variant.isin(["main", "reference"])) & (d.fleet == "residential") & (d.network == "ieee33")
            & (d.split == "test") & (d.scenario == "S4")]
    noev = sel[sel.method == "noev"].set_index("episode").violation_rate_pct
    if len(noev):
        base_days = set(noev[noev > 0].index)
        N.add("noev viol days S4", len(base_days), 0)
        bad = {}
        for m in ["nested"] + l3_methods + [x for x in LEARNED_ALL if "+L3" in x]:
            v = sel[sel.method == m].set_index("episode").violation_rate_pct
            extra = set(v[v > 0].index) - base_days
            if extra:
                bad[m] = sorted(extra)
            N.add(f"viol days S4 {m}", int((v > 0).sum()), 0)
        C.add("s4_l3_violations_only_base_load_days", not bad, json.dumps({k: [int(e) for e in v] for k, v in bad.items()}))
    l3rel = [abs(get(m + "+L3", sc)["cost_eur"] / get(m, sc)["cost_eur"] - 1) * 100
             for m in ("uncoordinated", "tou", "price_aware", "lp_opf") for sc in scen
             if get(m, sc) is not None and get(m + "+L3", sc) is not None]
    if l3rel:
        N.add("l3 cost change max", max(l3rel), 2, rnd="up")
    C.add("l3_negligible_cost_main", bool(l3rel) and max(l3rel) < 1.0, f"max {max(l3rel) if l3rel else None}")
    C.add("l3_zero_viol_learned_main", maxv([m for m in LEARNED_ALL if "+L3" in m]) == 0.0,
          f"max {maxv([m for m in LEARNED_ALL if '+L3' in m])}")
    C.add("learned_noL3_violate", maxv([m for m in LEARNED_ALL if "+L3" not in m]) > 0)
    sig = t_cost[t_cost.p_holm < 0.05].method.tolist() if len(t_cost) else []
    tc = t_cost.set_index("method") if len(t_cost) else None
    for m in ("price_aware+L3", "lp_opf+L3", "uncoordinated"):
        if tc is not None and m in tc.index:
            sign = "costlier" if tc.loc[m, "mean_diff"] > 0 else "cheaper"
            C.add(f"nested_sig_{sign}_than_{m.replace('+', '_')}_S3", bool(tc.loc[m, "p_holm"] < 0.05),
                  f"diff {tc.loc[m, 'mean_diff']:.1f}, p {tc.loc[m, 'p_holm']:.3g}")
    C.add("nested_sig_cheaper_all_learned_S3",
          all(m in sig and float(t_cost.set_index("method").loc[m, "mean_diff"]) < 0 for m in LEARNED_ALL
              if m in set(t_cost.method)), str(sig))

    # ablations: change relative to the complete framework (same days, same seeds)
    for sc in ("S3", "S5"):
        base = get("nested", sc)
        if base is None:
            continue
        for v in ABL_LABELS:
            if v == "main":
                continue
            m = "nested-noL3" if v in ("abl_no_l3", "ablation") else "nested"
            r = get(m, sc, v)
            if r is None:
                continue
            N.add(f"abl delta cost {v} {sc}", 100 * (r["cost_eur"] / base["cost_eur"] - 1), 1)
            N.add(f"abl delta cost abs {v} {sc}", abs(100 * (r["cost_eur"] / base["cost_eur"] - 1)), 1)
            C.add(f"abl_{v}_costlier_{sc}", r["cost_eur"] > base["cost_eur"],
                  f"{r['cost_eur']:.1f} vs {base['cost_eur']:.1f}")
            C.add(f"abl_{v}_cheaper_{sc}", r["cost_eur"] < base["cost_eur"],
                  f"{r['cost_eur']:.1f} vs {base['cost_eur']:.1f}")
    # decomposition: each configuration relative to the plan executed without learning
    for sc in ("S3", "S7"):
        plan = get("plan+L3", sc, "decomp")
        if plan is None:
            continue
        for v, m in DECOMP_ROWS:
            r = get(m, sc, v)
            if r is None or m == "plan+L3":
                continue
            N.add(f"decomp cost vs plan {m} {sc}", 100 * (r["cost_eur"] / plan["cost_eur"] - 1), 1)
            N.add(f"decomp cost vs plan abs {m} {sc}", abs(100 * (r["cost_eur"] / plan["cost_eur"] - 1)), 1)
            for k, col in (("curt", "curtailed_kwh"), ("unmet", "unmet_kwh")):
                if plan[col] > 0:
                    N.add(f"decomp {k} vs plan {m} {sc}", 100 * (1 - r[col] / plan[col]), 0)
            mm = m.replace("-", "_").replace("+", "_")
            C.add(f"decomp_{mm}_cheaper_than_plan_{sc}", r["cost_eur"] < plan["cost_eur"])
            C.add(f"decomp_{mm}_costlier_than_plan_{sc}", r["cost_eur"] > plan["cost_eur"])
            C.add(f"decomp_{mm}_less_curt_than_plan_{sc}", r["curtailed_kwh"] < plan["curtailed_kwh"])
            C.add(f"decomp_{mm}_less_unmet_than_plan_{sc}", r["unmet_kwh"] < plan["unmet_kwh"])
        # plan (no learning) versus the per-vehicle heuristic under stress
    plan7, pa7 = get("plan+L3", "S7", "decomp"), get("price_aware+L3", "S7")
    if plan7 is not None and pa7 is not None:
        C.add("plan_less_unmet_than_pa_S7", plan7["unmet_kwh"] < pa7["unmet_kwh"],
              f"{plan7['unmet_kwh']:.1f} vs {pa7['unmet_kwh']:.1f}")
        N.add("reduction unmet plan pa S7", 100 * (1 - plan7["unmet_kwh"] / pa7["unmet_kwh"]), 0)
    # generalisation settings (one row per setting, S3)
    gen_ok_v, gen_between, gen_sq = True, True, {}
    for variant, fleet, net, split, name in GENERAL:
        g = {m: row(s, variant, m, "S3", fleet, net, split) for m in ("uncoordinated", "price_aware+L3", "nested")}
        if any(v is None for v in g.values()):
            continue
        gen_ok_v &= g["nested"]["violation_rate_pct"] == 0.0
        gen_between &= g["price_aware+L3"]["cost_eur"] < g["nested"]["cost_eur"] < g["uncoordinated"]["cost_eur"]
        key = name.replace(",", "").replace("-", " ")
        N.add(f"gen saving nested unc {key}", 100 * (1 - g["nested"]["cost_eur"] / g["uncoordinated"]["cost_eur"]), 1)
        N.add(f"gen gap nested pa {key}", 100 * (g["nested"]["cost_eur"] / g["price_aware+L3"]["cost_eur"] - 1), 1)
        N.add(f"gen sq pa {key}", g["price_aware+L3"]["service_quality"], 3)
        gen_sq[key] = (round(float(g["nested"]["service_quality"]), 3), round(float(g["price_aware+L3"]["service_quality"]), 3))
    sav_g = [100 * (1 - row(s, v, "nested", "S3", f, n, sp)["cost_eur"] / row(s, v, "uncoordinated", "S3", f, n, sp)["cost_eur"])
             for v, f, n, sp, _ in GENERAL if row(s, v, "nested", "S3", f, n, sp) is not None
             and row(s, v, "uncoordinated", "S3", f, n, sp) is not None]
    gap_g = [100 * (row(s, v, "nested", "S3", f, n, sp)["cost_eur"] / row(s, v, "price_aware+L3", "S3", f, n, sp)["cost_eur"] - 1)
             for v, f, n, sp, _ in GENERAL if row(s, v, "nested", "S3", f, n, sp) is not None
             and row(s, v, "price_aware+L3", "S3", f, n, sp) is not None]
    if sav_g:
        N.add("gen saving min", min(sav_g), 1)
        N.add("gen saving max", max(sav_g), 1)
    if gap_g:
        N.add("gen gap min", min(gap_g), 1)
        N.add("gen gap max", max(gap_g), 1)
    C.add("general_nested_zero_viol", gen_ok_v)
    C.add("general_nested_between_pa_and_unc", gen_between)
    C.add("general_nested_sq_below_pa_caltech", any(k.endswith("ACN Caltech") and v[0] < v[1] for k, v in gen_sq.items()),
          json.dumps(gen_sq))
    # Level-3 sensitivity: inverter rating
    s10, s12, s14 = get("nested", "S3", "sens_S10"), get("nested", "S3"), get("nested", "S3", "sens_S14")
    if s10 is not None and s14 is not None:
        C.add("sens_smaller_rating_more_curtailment", s10["curtailed_kwh"] > s12["curtailed_kwh"] > s14["curtailed_kwh"],
              f"{s10['curtailed_kwh']:.1f} > {s12['curtailed_kwh']:.1f} > {s14['curtailed_kwh']:.1f}")
        costs = [r["cost_eur"] for r in (s10, s12, s14, get("nested", "S3", "sens_delta0.0005"),
                                         get("nested", "S3", "sens_delta0.002")) if r is not None]
        N.add("sens cost range", 100 * (max(costs) / min(costs) - 1), 2, rnd="up")
        C.add("sens_cost_insensitive", 100 * (max(costs) / min(costs) - 1) < 0.5)
        sens_rows = [r for r in (s10, s14, get("nested", "S3", "sens_delta0.0005"), get("nested", "S3", "sens_delta0.002"))
                     if r is not None]
        C.add("sens_zero_viol", all(r["violation_rate_pct"] == 0.0 for r in sens_rows))
        d05, d20 = get("nested", "S3", "sens_delta0.0005"), get("nested", "S3", "sens_delta0.002")
        if d05 is not None and d20 is not None:
            C.add("sens_larger_margin_more_activation",
                  d05["q_activation_pct"] < s12["q_activation_pct"] < d20["q_activation_pct"])
    # ablations: violations without Level 3
    for v in ("abl_no_l3", "ablation"):
        r = get("nested-noL3", "S3", v)
        if r is not None:
            C.add(f"{v}_violates_S3", r["violation_rate_pct"] > 0)
    nb = get("nested", "S3", "abl_no_behavior")
    if nb is not None:
        C.add("abl_no_behavior_higher_sq_S3", nb["service_quality"] > get("nested", "S3")["service_quality"])
        # stress: nested versus the price-aware heuristic with Level 3
    for sc in ("S5", "S7"):
        nst, pa = get("nested", sc), get("price_aware+L3", sc)
        if nst is not None and pa is not None:
            C.add(f"nested_less_curtailment_than_pa_{sc}", nst["curtailed_kwh"] < pa["curtailed_kwh"],
                  f"{nst['curtailed_kwh']:.1f} vs {pa['curtailed_kwh']:.1f}")
            C.add(f"nested_fewer_viol_than_pa_{sc}", nst["violation_rate_pct"] < pa["violation_rate_pct"],
                  f"{nst['violation_rate_pct']:.3f} vs {pa['violation_rate_pct']:.3f}")
            C.add(f"nested_higher_sq_than_pa_{sc}", nst["service_quality"] > pa["service_quality"],
                  f"{nst['service_quality']:.4f} vs {pa['service_quality']:.4f}")
            C.add(f"nested_cheaper_than_pa_{sc}", nst["cost_eur"] < pa["cost_eur"],
                  f"{nst['cost_eur']:.1f} vs {pa['cost_eur']:.1f}")
            C.add(f"pa_cheaper_than_nested_{sc}", pa["cost_eur"] < nst["cost_eur"],
                  f"{pa['cost_eur']:.1f} vs {nst['cost_eur']:.1f}")
            C.add(f"nested_less_unmet_than_pa_{sc}", nst["unmet_kwh"] < pa["unmet_kwh"],
                  f"{nst['unmet_kwh']:.1f} vs {pa['unmet_kwh']:.1f}")
            for k, col in (("curt", "curtailed_kwh"), ("unmet", "unmet_kwh"), ("viol", "violation_rate_pct")):
                if pa[col] > 0:
                    N.add(f"reduction {k} nested pa {sc}", 100 * (1 - nst[col] / pa[col]), 0)

    for _, r in t_cost.iterrows():
        N.add(f"diff cost {r['method']}", abs(r["mean_diff"]), 1)
        N.add(f"pholm cost {r['method']}", "<0.001" if r["p_holm"] < 0.001 else f"{r['p_holm']:.3f}")
        N.add(f"wins cost {r['method']}", r["wins"], 0)
    calib = json.loads((art / "calibration.json").read_text()) if (art / "calibration.json").exists() else {}
    for net, c in calib.items():
        N.add(f"scale {net}", c["base_load_scale"], 2)
        N.add(f"noev vmin {net}", c["no_ev_min_voltage_pu"], 4)
    offs = {}
    for f in sorted((art / "runs").glob("*/price_calibration.json")):
        name = f.parent.name
        if "_residential_ieee33_none_s" in name:
            offs.setdefault(name.split("_residential")[0], []).append(json.loads(f.read_text())["price_offset"])
    if offs:
        allv = [v for vs in offs.values() for v in vs]
        N.add("calib offset absmax", max(abs(v) for v in allv), 3, rnd="up")
        if "nested" in offs:
            N.add("calib offset nested min", min(offs["nested"]), 3)
            N.add("calib offset nested max", max(offs["nested"]), 3)
    tune_f = art / "tuning" / "summary.csv"                 # model selection (TUNING.md)
    if tune_f.exists():
        tu = pd.read_csv(tune_f).groupby(["candidate", "scenario"]).mean(numeric_only=True)
        sel = json.loads((art / "tuning" / "selected.json").read_text())
        alt = [c for c in sorted({c for c, _ in tu.index}) if c != "A"]
        rel = lambda col, sc: {c: 100 * (tu.loc[(c, sc), col] / tu.loc[("A", sc), col] - 1) for c in alt}
        cost3, un3, un7 = rel("cost", "S3"), rel("unmet", "S3"), rel("unmet", "S7")
        N.add("tune val days", int(pd.read_csv(tune_f).days.max()), 0)
        N.add("tune seeds", len(sel["seeds"]), 0)
        N.add("tune cost reduction max", -min(cost3.values()), 1)
        N.add("tune unmet increase min S3", min(un3.values()), 0)
        N.add("tune unmet increase max S3", max(un3.values()), 0)
        N.add("tune unmet increase min S7", min(un7.values()), 0)
        N.add("tune unmet increase max S7", max(un7.values()), 0)
        C.add("tune_llf_selected", sel["selected"] == "A" and sel["disaggregation"] == "llf", json.dumps(sel["J"]))
        C.add("tune_plan_cheaper_S3", any(v < 0 for v in cost3.values()), json.dumps(cost3))
        C.add("tune_plan_more_unmet", all(v > 0 for v in [*un3.values(), *un7.values()]),
              json.dumps({"S3": un3, "S7": un7}))
    N.add("eval days", int(d.episode.nunique()), 0)
    N.add("train seeds", int(len(list((art / "runs").glob("nested_residential_ieee33_none_s*")))), 0)
    N.add("train episodes", int(cfg["training"]["episodes"]), 0)
    N.add("aux seeds", int(len(list((art / "runs").glob("nested_residential_ieee33_no_l1_s*")))), 0)
    hh = cfg["aggregators"]["households"]
    if hh:
        for sc, spec in cfg["evaluation"]["scenarios"].items():      # penetration = EVs / households
            N.add(f"pen {sc}", round(100 * spec["n_ev"] / hh), 0)
    if compute_df is not None and len(compute_df):
        nest = compute_df[(compute_df.method == "nested") & (compute_df.ablation == "none")
                          & (compute_df.fleet == "residential") & (compute_df.network == "ieee33")]
        if len(nest):
            N.add("train hours nested", nest.total_h.mean(), 1)
    N.write(out / "numbers.tex")
    C.write(out / "claims.json")
    return N


def build_all(art: pathlib.Path, out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df = load(art)
    d = per_day(df)
    s = summarise(d)
    s.to_csv(out / "summary.csv", index=False)
    seed_spread(df).to_csv(out / "seed_spread.csv", index=False)
    table_main(s, "S3", out / "tab_main_s3.tex")
    table_scenarios(s, ["uncoordinated", "tou", "uncoordinated+L3", "tou+L3", "price_aware+L3", "lp_opf+L3",
                        "flat_ddpg", "ppo_lag", "cpo", "hrl", "nested"],
                    ["S1", "S2", "S3", "S4", "S5"], out / "tab_scenarios.tex")
    others = [m for m in ORDER if m != "nested"]
    t_cost = paired_tests(d, "nested", others, "cost_eur", "S3")
    t_sq = paired_tests(d, "nested", others, "service_quality", "S3")
    t_cost.to_csv(out / "tests_cost_s3.csv", index=False)
    t_sq.to_csv(out / "tests_sq_s3.csv", index=False)
    table_stats(t_cost, t_sq, out / "tab_stats.tex")
    table_ablation(s, ["S3", "S5"], out / "tab_ablation.tex")
    table_general(s, out / "tab_general.tex")
    table_stress(s, out / "tab_stress.tex")
    table_sensitivity(s, out / "tab_sensitivity.tex")
    table_decomp(s, out / "tab_decomp.tex")
    compute_df = table_compute(art, out / "tab_compute.tex")
    figures(s, d, art, out)
    import yaml
    from .diagrams import build_diagrams
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml"))
    table_hyper(cfg, out / "tab_hyper.tex")
    build_diagrams(cfg, out)
    numbers(s, d, t_cost, t_sq, compute_df, art, out, cfg)
    print(f"== analysis: tables, figures and numbers written to {out}")
