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
import pathlib

import numpy as np
import pandas as pd
from scipy import stats

from .methods import label

ROOT = pathlib.Path(__file__).resolve().parents[3]
BOOT = 10_000
RULE_ORDER = ["uncoordinated", "uncoordinated+L3", "tou", "tou+L3", "price_aware", "price_aware+L3",
              "lp_opf", "lp_opf+L3"]
LEARN_ORDER = ["flat_ddpg", "flat_ddpg+L3", "ppo_lag", "ppo_lag+L3", "cpo", "cpo+L3", "hrl", "hrl+L3"]
ORDER = RULE_ORDER + LEARN_ORDER + ["nested"]
METRICS = ["cost_eur", "cost_per_kwh", "service_quality", "violation_rate_pct", "min_voltage_pu",
           "curtailed_kwh", "q_activation_pct", "peak_ev_kw", "retail_price_paid", "energy_delivered_kwh"]


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
    """'uncoordinated+L3 S1 cost' -> 'UncoordinatedLThreeSOneCost' (LaTeX macro names
    may contain letters only)."""
    import re
    words = re.split(r"[^A-Za-z0-9]+", " ".join(str(p) for p in parts))
    s = "".join(w[:1].upper() + w[1:] for w in words if w)
    for d, w in zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven",
                                   "Eight", "Nine"]):
        s = s.replace(d, w)
    return s


class Numbers:
    def __init__(self):
        self.lines = ["% Auto-generated by nflev.eval.analysis -- do not edit by hand."]
        self.names = set()

    def add(self, name, value, nd=2):
        name = macro_name(name)
        if name in self.names:
            raise ValueError(f"duplicate macro {name}")
        self.names.add(name)
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
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"Method & Cost (\euro) & Cost (\euro/kWh) & SQ & Viol.\ (\%) & $V_{\min}$ (p.u.) & Curt.\ (kWh) \\",
             r"\midrule"]
    for i, m in enumerate(ORDER):
        r = row(s, "main", m, scenario)
        if r is None:
            continue
        if m in ("flat_ddpg", "nested"):
            lines.append(r"\midrule")
        name = label(m)
        cells = [fmt_ci(r, "cost_eur", 1), fmt(r["cost_per_kwh"], 4), fmt(r["service_quality"], 3),
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
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Comparison (nested $-$ method) & $\Delta$Cost (\euro) & $p_{\mathrm{Holm}}$ & $\Delta$SQ & $p_{\mathrm{Holm}}$ \\",
             r"\midrule"]
    sq = t_sq.set_index("method")
    for _, r in t_cost.iterrows():
        q = sq.loc[r["method"]]
        lines.append(f"{label(r['method'])} & {r['mean_diff']:+.1f} {{\\scriptsize[{r['lo']:+.1f}, {r['hi']:+.1f}]}}"
                     f" & {fmt_p(r['p_holm'])} & {q['mean_diff']:+.4f} & {fmt_p(q['p_holm'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


ABL_LABELS = {"main": "Complete framework", "abl_no_l1": "No Level 1 (fixed corridor)",
              "abl_flat_timescale": "Single timescale (L1 every 15 min)",
              "abl_no_behavior": "No behavioural model (L3a)", "abl_no_l3": "No Level 3 (trained without)",
              "ablation": "Level 3 removed at evaluation", "abl_no_curriculum": "No curriculum",
              "abl_no_guard": "No deadline guard", "abl_proportional": "Proportional instead of LLF"}


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


def table_general(s, path):
    settings = [("general", "residential", "ieee33", "test", "S3", "IEEE 33-bus, residential (main)"),
                ("general", "residential", "ieee69", "test", "S3", "IEEE 69-bus, residential"),
                ("general", "acn_caltech", "ieee33", "test", "S3", "IEEE 33-bus, ACN Caltech (real sessions)"),
                ("general", "acn_jpl", "ieee33", "test", "S3", "IEEE 33-bus, ACN JPL (real sessions)"),
                ("regime", "residential", "ieee33", "alt", "S3", "IEEE 33-bus, 2019 price regime")]
    methods = ["uncoordinated", "uncoordinated+L3", "price_aware+L3", "lp_opf+L3", "nested"]
    lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"Setting & Method & Cost (\euro) & \euro/kWh & SQ & Viol.\ (\%) \\", r"\midrule"]
    for variant, fleet, net, split, sc, name in settings:
        first = True
        for m in methods:
            v = "main" if (variant == "general" and fleet == "residential" and net == "ieee33") else variant
            r = row(s, v, m, sc, fleet, net, split)
            if r is None:
                continue
            lines.append((name if first else "") + f" & {label(m)} & {fmt(r['cost_eur'], 1)} & "
                         f"{fmt(r['cost_per_kwh'], 4)} & {fmt(r['service_quality'], 3)} & "
                         f"{fmt(r['violation_rate_pct'], 2)} \\\\")
            first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n")


def table_stress(s, path):
    methods = ["uncoordinated", "tou", "uncoordinated+L3", "tou+L3", "price_aware+L3", "lp_opf+L3", "nested"]
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"Method & \multicolumn{3}{c}{S6 (no charging 19--20 h)} & \multicolumn{3}{c}{S7 (120 \%, inverters 50 \%)} \\",
             r" & Viol.\ (\%) & $V_{\min}$ & Curt.\ (kWh) & Viol.\ (\%) & $V_{\min}$ & Curt.\ (kWh) \\", r"\midrule"]
    for m in methods:
        cells = []
        for sc in ("S6", "S7"):
            r = row(s, "main", m, sc)
            cells += ["--"] * 3 if r is None else [fmt(r["violation_rate_pct"], 2), fmt(r["min_voltage_pu"], 4),
                                                   fmt(r["curtailed_kwh"], 1)]
        lines.append(label(m) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def table_sensitivity(s, path):
    rows = [("sens_delta0.0005", r"$\delta = 0.0005$"), ("main", r"$\delta = 0.001$ (default), $S = 12$ kVA (default)"),
            ("sens_delta0.002", r"$\delta = 0.002$"), ("sens_S10", r"$S = 10$ kVA"), ("sens_S14", r"$S = 14$ kVA")]
    lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Setting & Cost (\euro) & SQ & Viol.\ (\%) & Q act.\ (\%) & Curt.\ (kWh) \\", r"\midrule"]
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


# -------------------------------------------------------------------- figures
def figures(s, d, art, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "pdf.fonttype": 42, "axes.linewidth": 0.6})
    methods = ["uncoordinated", "tou", "uncoordinated+L3", "price_aware+L3", "lp_opf+L3",
               "flat_ddpg", "ppo_lag", "cpo", "hrl", "nested"]
    colors = ["#7f7f7f", "#e69f00", "#bdbdbd", "#56b4e9", "#0072b2", "#009e73", "#f0e442", "#cc79a7",
              "#d55e00", "#c00000"]
    scen = ["S1", "S2", "S3", "S4", "S5"]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2))
    w = 0.8 / len(methods)
    for ax, (met, lab) in zip(axes, [("cost_per_kwh", "Cost (EUR/kWh)"), ("service_quality", "Service quality"),
                                     ("violation_rate_pct", "Violation rate (%)")]):
        for j, (m, c) in enumerate(zip(methods, colors)):
            vals, err = [], []
            for sc in scen:
                r = row(s, "main", m, sc)
                vals.append(np.nan if r is None else r[met])
                err.append([0, 0] if r is None else [r[met] - r[met + "_lo"], r[met + "_hi"] - r[met]])
            x = np.arange(len(scen)) + (j - len(methods) / 2 + 0.5) * w
            ax.bar(x, vals, w, color=c, yerr=np.array(err).T, error_kw={"lw": 0.4, "capsize": 0.8},
                   label=label(m), linewidth=0)
        ax.set_xticks(range(len(scen)), scen)
        ax.set_ylabel(lab)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    axes[1].set_ylim(0.8, 1.01)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False,
               fontsize=6, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "fig_scenarios.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for m, c in zip(methods, colors):
        r = row(s, "main", m, "S3")
        if r is None:
            continue
        ax.scatter(r["violation_rate_pct"], r["cost_per_kwh"], color=c, s=40 if m == "nested" else 22,
                   marker="*" if m == "nested" else "o", edgecolor="k", linewidth=0.3, zorder=3)
        ax.annotate(label(m).replace(" (proposed)", ""), (r["violation_rate_pct"], r["cost_per_kwh"]),
                    fontsize=5.5, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("Violation rate at S3 (%)")
    ax.set_ylabel("Cost per delivered kWh (EUR)")
    ax.grid(lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out / "fig_tradeoff.pdf")
    plt.close(fig)

    runs = sorted((art / "runs").glob("*_residential_ieee33_none_s*"))
    if runs:
        fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.0))
        for name, c in [("nested", "#c00000"), ("flat_ddpg", "#009e73"), ("ppo_lag", "#f0e442"),
                        ("cpo", "#cc79a7"), ("hrl", "#d55e00")]:
            logs = [pd.read_csv(r / "train_log.csv") for r in runs if r.name.startswith(name + "_residential")]
            if not logs:
                continue
            n = min(len(lg) for lg in logs)
            cpk = np.mean([lg.cost_per_kwh.values[:n] for lg in logs], axis=0)
            sq = np.mean([lg.service_quality.values[:n] for lg in logs], axis=0)
            k = 25
            ax[0].plot(pd.Series(cpk).rolling(k, min_periods=1).mean(), color=c, lw=0.9, label=label(name))
            ax[1].plot(pd.Series(sq).rolling(k, min_periods=1).mean(), color=c, lw=0.9)
        ax[0].set(xlabel="Training episode", ylabel="Cost (EUR/kWh)")
        ax[1].set(xlabel="Training episode", ylabel="Service quality")
        ax[0].legend(fontsize=6, frameon=False)
        for a in ax:
            a.grid(lw=0.3, alpha=0.5)
        fig.tight_layout()
        fig.savefig(out / "fig_training.pdf")
        plt.close(fig)


# ------------------------------------------------------------------ numbers
def numbers(s, d, t_cost, t_sq, compute_df, art, out):
    N = Numbers()
    for sc in ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]:
        for m in ORDER:
            r = row(s, "main", m, sc)
            if r is None:
                continue
            N.add(f"{m} {sc} cost", r["cost_eur"], 1)
            N.add(f"{m} {sc} cpk", r["cost_per_kwh"], 4)
            N.add(f"{m} {sc} sq", r["service_quality"], 3)
            N.add(f"{m} {sc} viol", r["violation_rate_pct"], 2)
            N.add(f"{m} {sc} vmin", r["min_voltage_pu"], 4)
            N.add(f"{m} {sc} curt", r["curtailed_kwh"], 1)
            N.add(f"{m} {sc} qact", r["q_activation_pct"], 2)
    for _, r in t_cost.iterrows():
        N.add(f"diff cost {r['method']}", abs(r["mean_diff"]), 1)
        N.add(f"pholm cost {r['method']}", "<0.001" if r["p_holm"] < 0.001 else f"{r['p_holm']:.3f}")
        N.add(f"wins cost {r['method']}", r["wins"], 0)
    calib = json.loads((art / "calibration.json").read_text()) if (art / "calibration.json").exists() else {}
    for net, c in calib.items():
        N.add(f"scale {net}", c["base_load_scale"], 2)
        N.add(f"noev vmin {net}", c["no_ev_min_voltage_pu"], 4)
    N.add("eval days", int(d.episode.nunique()), 0)
    N.add("train seeds", int(len([p for p in (art / "runs").glob("nested_residential_ieee33_none_s*")])), 0)
    N.write(out / "numbers.tex")
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
    compute_df = table_compute(art, out / "tab_compute.tex")
    figures(s, d, art, out)
    import yaml
    from .diagrams import build_diagrams
    build_diagrams(yaml.safe_load(open(ROOT / "configs" / "base.yaml")), out)
    numbers(s, d, t_cost, t_sq, compute_df, art, out)
    print(f"== analysis: tables, figures and numbers written to {out}")
