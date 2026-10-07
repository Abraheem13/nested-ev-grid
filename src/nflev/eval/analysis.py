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

from .diagrams import save_checked
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
EXTRA = ["q_kvarh", "peak_feeder_mw", "pf_solves"]   # grid-side means (no interval; RNG stream unchanged)


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
    return df.groupby(keys, as_index=False)[METRICS + EXTRA].mean()


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
        for m in EXTRA:
            r[m] = float(g[m].mean())
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
        if isinstance(value, str):
            v = value
        elif nd == 0:
            v = f"{int(math.copysign(math.floor(abs(float(value)) + 0.5), value))}"
        else:
            v = f"{value:.{nd}f}"
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
             r"Method & Cost (\euro) & Cost/kWh & Retail & SQ & Viol.\ (\%) & Mean $V^{\min}$ & Curt.\ (kWh) \\",
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
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def table_scenarios(s, methods, scenarios, pen, path):
    cols = "".join("rr" for _ in scenarios)
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{sc} ({pen[sc]}\%)}}" for sc in scenarios)
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


def signed(x, nd):
    v = f"{abs(x):.{nd}f}"
    return ("$-$" if x < 0 and float(v) != 0 else "$+$") + v


def table_stats(t_cost, t_sq, path):
    """Rows: every baseline with Level 3 attached (the nested controller includes it).
    The Holm correction runs over all comparisons, with and without Level 3."""
    lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Nested vs. & $\Delta$Cost (\euro) & $p_{\mathrm{Holm}}$ & Days & $\Delta$SQ & $p_{\mathrm{Holm}}$ \\",
             r"\midrule"]
    sq = t_sq.set_index("method")
    for _, r in t_cost[t_cost.method.str.endswith("+L3")].iterrows():
        q = sq.loc[r["method"]]
        lines.append(f"{short_label(r['method'])} & {signed(r['mean_diff'], 1)} {{\\scriptsize[{signed(r['lo'], 1)}, "
                     f"{signed(r['hi'], 1)}]}} & {fmt_p(r['p_holm'])} & {int(r['wins'])} & {signed(q['mean_diff'], 4)}"
                     f" & {fmt_p(q['p_holm'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


ABL_LABELS = {"abl_base": "Complete framework", "abl_no_prior": "No planning prior",
              "abl_no_l1": "No Level 1", "abl_flat_timescale": "Single timescale",
              "abl_no_behavior": "No behavior model", "abl_no_l3": "No Level 3 (trained)",
              "ablation": "Level 3 removed at test", "abl_no_curriculum": "No curriculum",
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
                cells += [fmt(r["cost_eur"], 1), fmt(r["service_quality"], 3), fmt(r["violation_rate_pct"], 2)]
        if any_row:
            lines.append(lab + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


GENERAL = [("main", "residential", "ieee33", "test", "33-bus, residential"),
           ("general", "residential", "ieee69", "test", "69-bus, residential"),
           ("general", "acn_caltech", "ieee33", "test", "33-bus, ACN Caltech"),
           ("general", "acn_jpl", "ieee33", "test", "33-bus, ACN JPL"),
           ("regime", "residential", "ieee33", "alt", "33-bus, 2019 days")]


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
             r" & Viol. & $V^{\min}$ & Viol. & Curt. & Unmet & SQ \\", r"\midrule"]
    for m in methods:
        a, b = row(s, "main", m, "S6"), row(s, "main", m, "S7")
        cells = (["--"] * 2 if a is None else [fmt(a["violation_rate_pct"], 2), fmt(a["min_voltage_pu"], 3)]) + \
                (["--"] * 4 if b is None else [fmt(b["violation_rate_pct"], 2), fmt(b["curtailed_kwh"], 0),
                                              fmt(b["unmet_kwh"], 0), fmt(b["service_quality"], 3)])
        lines.append(short_label(m) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def compute_times(art):
    """Training episodes and wall-clock time of every stored run (from train_log.csv)."""
    rows = []
    for d in sorted((art / "runs").glob("*")):
        log = d / "train_log.csv"
        if not log.exists():
            continue
        _, fleet, net, abl, _ = d.name.rsplit("_", 4)
        lg = pd.read_csv(log)
        rows.append({"run": d.name, "method": d.name.split("_residential")[0].split("_acn")[0],
                     "fleet": fleet, "network": net, "ablation": abl,
                     "episodes": len(lg), "wall_s_ep": lg.wall_s.mean(), "total_h": lg.wall_s.sum() / 3600})
    return pd.DataFrame(rows)


def table_setup(cfg, calib, path):
    """Simulation setup, read from configs/base.yaml, the calibration report and the
    fleet module (cannot drift from the code)."""
    from ..data.fleets import BATTERY_CLASSES_KWH, EPISODE_START_HOUR
    fl, bh, rp, v = cfg["fleets"]["residential"], cfg["behavior"], cfg["reactive_power"], cfg["voltage"]
    ag, sim, d = cfg["aggregators"], cfg["simulation"], cfg["data"]
    yrs = lambda k: ", ".join(str(y) for y in d[k])  # noqa: E731
    n = lambda mu, sd: f"$\\mathcal{{N}}({mu:g}, {sd:g}^2)$"  # noqa: E731
    sc = {k: c["base_load_scale"] for k, c in calib.items()}
    rows = [
        ("Feeders", "IEEE 33-/69-bus, 12.66\\,kV, $V_0={:g}$\\,p.u., floor {:g}\\,p.u.".format(
            cfg["network"]["substation_vm_pu"], v["v_min"])),
        ("Aggregators", "buses {} (33-bus), {} (69-bus); {:g}\\,kW each; {} households".format(
            ", ".join(map(str, ag["buses"]["ieee33"])), ", ".join(map(str, ag["buses"]["ieee69"])),
            ag["transformer_cap_kw"], ag["households"])),
        ("Base load", "Pecan Street profile, scale {} / {}, noise $\\sigma$ {:g}".format(
            fmt(sc.get("ieee33", float("nan")), 2), fmt(sc.get("ieee69", float("nan")), 2), sim["load_noise_sigma"])),
        ("Prices", f"ENTSO-E NL day-ahead; train {yrs('train_years')}, test {yrs('test_years')}, "
                   f"regime {yrs('alt_regime_years')}"),
        ("Arrival, dwell", f"{n(fl['arrival_mu_h'], fl['arrival_sigma_h'])}\\,h in "
                           f"[{fl['arrival_min_h']:g}, {fl['arrival_max_h']:g}]; "
                           f"{n(fl['dwell_mu_h'], fl['dwell_sigma_h'])}\\,h in [{fl['dwell_min_h']:g}, {fl['dwell_max_h']:g}]"),
        ("SoC, battery", f"{n(fl['soc_init_mu'], fl['soc_init_sigma'])} to {n(fl['soc_target_mu'], fl['soc_target_sigma'])}; "
                         + "/".join(f"{b:g}" for b in BATTERY_CLASSES_KWH) + "\\,kWh"),
        ("Charger", f"{rp['charger_p_max_kw']:g}\\,kW, $\\eta$ {rp['charger_efficiency']:g}, $S_i$ {rp['s_rated_kva']:g}\\,kVA"),
        ("Acceptance", f"$\\lambda^{{\\mathrm{{ref}}}}$ \\euro{bh['lambda_ref']:.2f}/kWh, $w^{{\\mathrm{{c}}}}_i$ "
                       f"{n(*bh['w_cost'])}, $w^{{\\mathrm{{n}}}}_i$ {n(*bh['w_norm'])}, $b_i$ {n(*bh['bias'])}"),
        ("Timing", "power flow {:g}\\,s, dispatch {:g}\\,min, pricing {:g}\\,h; day from {:02d}:00; "
                   "look-ahead {}\\,h".format(sim["resolution_s"], sim["dispatch_interval_s"] / 60,
                                              sim["pricing_interval_s"] / 3600, EPISODE_START_HOUR["residential"],
                                              sim["price_lookahead_h"])),
    ]
    lines = [r"\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{0.74\columnwidth}@{}}", r"\toprule",
             r"Item & Value \\", r"\midrule"]
    lines += [f"{a} & {b} \\\\" for a, b in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


SCEN_TEXT = {"S1": "nominal", "S2": "nominal", "S3": "nominal (main case)", "S5": "nominal",
             "S4": r"base load $\pm$15\% hourly", "S6": "no charging 19:00--20:00",
             "S7": r"inverter rating $\times$0.5"}


def table_scen_def(cfg, path):
    hh = cfg["aggregators"]["households"]
    lines = [r"\begin{tabular}{@{}lrrl@{}}", r"\toprule", r"Scenario & EVs & Pen.\ (\%) & Condition \\", r"\midrule"]
    for sc, spec in cfg["evaluation"]["scenarios"].items():
        lines.append(f"{sc} & {spec['n_ev']} & {round(100 * spec['n_ev'] / hh)} & {SCEN_TEXT.get(sc, '')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


GRID_METHODS = ["uncoordinated", "uncoordinated+L3", "tou+L3", "price_aware+L3", "lp_opf+L3", "hrl+L3", "nested"]


def table_grid(s, path):
    """Grid-side view of S3: peak loads, how often and how hard Level 3 acts, and the
    power flows it costs."""
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"Method & Peak EV & Peak feeder & L3 act. & $Q$ energy & Curt. & PF \\",
             r" & (kW) & (MW) & (\%) & (kvarh) & (kWh) & per day \\", r"\midrule"]
    for m in GRID_METHODS:
        r = row(s, "main", m, "S3")
        if r is None:
            continue
        lines.append(f"{short_label(m)} & {fmt(r['peak_ev_kw'], 0)} & {fmt(r['peak_feeder_mw'], 2)} & "
                     f"{fmt(r['q_activation_pct'], 2)} & {fmt(r['q_kvarh'], 1)} & {fmt(r['curtailed_kwh'], 1)} & "
                     f"{fmt(r['pf_solves'], 0)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def table_learned(s, spread, compute_df, path):
    """Learned methods at S3: cost, its spread over training seeds, and training cost."""
    names = {"nested": "nested", "flat": "flat_ddpg", "ppo": "ppo_lag", "cpo": "cpo", "hrl": "hrl"}
    main = compute_df[(compute_df.fleet == "residential") & (compute_df.network == "ieee33")
                      & (compute_df.ablation == "none")]
    hours = main.assign(m=main.method.map(lambda x: names.get(x.split("_")[0], x))).groupby("m").total_h.mean()
    sp = spread[(spread.variant == "main") & (spread.scenario == "S3") & (spread.fleet == "residential")
                & (spread.network == "ieee33") & (spread.split == "test")].set_index("method").cost_eur
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Method & Cost (\euro) & Seed s.d.\ (\euro) & SQ & Training (h) \\", r"\midrule"]
    for m in ["flat_ddpg", "ppo_lag", "cpo", "hrl", "nested"]:
        r = row(s, "main", m, "S3")
        if r is None:
            continue
        lines.append(f"{label(m)} & {fmt(r['cost_eur'], 1)} & {fmt(sp.get(m, float('nan')), 1)} & "
                     f"{fmt(r['service_quality'], 3)} & {fmt(hours.get(m, float('nan')), 2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


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
    """Hyperparameters, read from configs/base.yaml and the agent classes (cannot drift from the code)."""
    t, d, pp, l2 = cfg["training"], cfg["training"]["ddpg"], cfg["training"]["ppo"], cfg["level2"]
    rn = l2["revenue_neutral"]
    import inspect
    from ..agents.cpo import CPO
    from ..agents.ppo_lagrangian import PPOLagrangian
    defaults = lambda c: {k: v.default for k, v in inspect.signature(c.__init__).parameters.items()}  # noqa: E731
    lag, cpo = defaults(PPOLagrangian), defaults(CPO)

    def thin(x):
        return f"{int(x):,}".replace(",", "\\,")
    rows = [
        ("Level 1 (PPO)", "actor/critic 256--128--64, Beta policy, "
                          f"lr {pp['lr']:g}, $\\gamma$ {pp['gamma']}, $\\lambda_{{\\mathrm{{GAE}}}}$ {pp['gae_lambda']}, "
                          f"clip {pp['clip']}, {pp['epochs']} epochs per update, entropy {pp['entropy']:g}"),
        ("Level 2 (DDPG)", "actor/critic 256--128--64, shared, one-hot id"),
        ("", f"lr actor/critic {d['lr_actor']:g}/{d['lr_critic']:g}, $\\gamma$ {d['gamma']}, $\\tau$ {d['tau']}"),
        ("", f"batch {d['batch']}, buffer {thin(d['buffer'])}, warm-up {thin(d['warmup'])} transitions"),
        ("", f"Gaussian action noise, std {d['noise_start']}$\\to${d['noise_end']} (factor {d['noise_decay']} "
             f"per episode); residual scale $\\rho$ {l2.get('residual_scale', 1.0)}"),
        ("Multiplier", f"$K_p$ {rn['kp']}, $K_i$ {rn['ki']}, EMA weight $\\alpha$ {rn['ema']}, "
                       f"$\\omega\\in[{rn['w_min']}, {rn['w_max']}]$, $|K_i\\textstyle\\sum e|\\le{max(abs(rn['w_min']), abs(rn['w_max']))}$"),
        ("Training", f"{t['episodes']} episodes; curriculum window {cfg['curriculum']['window']}"),
        ("Baselines", "same networks, learning rates, episodes and curriculum; safe RL: "
                      f"$\\gamma$ {d['gamma']}, cost limit {lag['cost_limit']:g}, dual step {lag['lr_dual']:g} "
                      f"(PPO-Lag.), KL radius {cpo['delta_kl']:g} (CPO)"),
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


def figures(s, d, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .diagrams import FONT
    plt.rcParams.update({**FONT, "font.size": 8, "axes.linewidth": 0.6, "axes.edgecolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.0))
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
    # (a) mean daily cost relative to the perfect-foresight LP-OPF (ratio of means); the
    # interval resamples days, keeping both methods' costs of a day together (paired)
    rng = np.random.default_rng(2)
    sel = d[(d.variant == "main") & (d.fleet == "residential") & (d.network == "ieee33") & (d.split == "test")]
    ref = sel[sel.method == "lp_opf+L3"].set_index(["scenario", "episode"]).cost_eur
    for m in ["uncoordinated", "flat_ddpg", "ppo_lag", "cpo", "hrl", "price_aware+L3", "nested"]:
        col, ls, mk = STYLE[m]
        mm = sel[sel.method == m].set_index(["scenario", "episode"]).cost_eur
        y, lo, hi = [], [], []
        for sc, _ in PEN:
            if sc not in mm.index.get_level_values(0) or sc not in ref.index.get_level_values(0):
                y.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                continue
            pair = pd.concat([mm.loc[sc], ref.loc[sc]], axis=1, join="inner").values
            idx = rng.integers(0, len(pair), (BOOT, len(pair)))
            boot = 100.0 * (pair[idx, 0].mean(axis=1) / pair[idx, 1].mean(axis=1) - 1.0)
            y.append(100.0 * (pair[:, 0].mean() / pair[:, 1].mean() - 1.0))
            lo.append(float(np.percentile(boot, 2.5))); hi.append(float(np.percentile(boot, 97.5)))
        y, lo, hi = map(np.asarray, (y, lo, hi))
        axes[0].errorbar(x, y, yerr=[y - lo, hi - y], color=col, ls=ls, marker=mk, ms=4.2,
                         lw=1.8 if m == "nested" else 1.0, elinewidth=0.6, capsize=1.5, label=label(m),
                         zorder=3 if m == "nested" else 2, mec="white", mew=0.4)
    axes[0].axhline(0.0, color="#52514e", lw=0.6)
    axes[0].set_ylabel("Cost above LP-OPF$^\\ast$ (%)")
    axes[0].set_title("(a) Mean daily cost relative to LP-OPF$^\\ast$+L3", fontsize=8, loc="left")
    for m in ["uncoordinated", "tou", "flat_ddpg", "ppo_lag", "cpo", "hrl"]:
        line(axes[1], m, "violation_rate_pct")
    axes[1].set_yscale("symlog", linthresh=0.1, linscale=0.6)
    axes[1].set_ylim(-0.005, 15)
    axes[1].set_yticks([0, 0.1, 1, 10], ["0", "0.1", "1", "10"])
    axes[1].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    zero = [row(s, "main", m, sc)["violation_rate_pct"] for m in
            ["nested", "uncoordinated+L3", "tou+L3", "price_aware+L3", "lp_opf+L3"] + [m for m in LEARNED_ALL if "+L3" in m]
            for sc, _ in PEN if row(s, "main", m, sc) is not None]
    if zero:
        axes[1].text(0.02, 0.98, f"Nested and every +L3 controller: {max(zero):.2f}%",
                     transform=axes[1].transAxes, fontsize=7, va="top", color="#52514e")
    axes[1].set_ylabel("Violation rate (%, symlog)")
    axes[1].set_title("(b) Violation rate of controllers without Level 3", fontsize=8, loc="left")
    for ax in axes:
        ax.set_xticks(x, [f"{p}% ({sc})" for sc, p in PEN])
        ax.set_xlabel("EV penetration (scenario)", labelpad=2)
        ax.grid(axis="y", lw=0.3, color="#d9d8d4")
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = [], []                  # one legend: style and colour identify a method in both panels
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.0),
               handlelength=2.4, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=2.0)
    save_checked(fig, out / "fig_scenarios.pdf", bbox_inches="tight")
    plt.close(fig)


PROFILE = {  # method -> (label, colour, line style)
    "uncoordinated": ("Uncoordinated", PAL[3], "--"),
    "uncoordinated+L3": ("Uncoordinated + L3", "#8c6d1f", "-."),
    "price_aware+L3": ("Price-aware heuristic + L3", PAL[1], "--"),
    "nested": ("Nested (proposed)", PAL[0], "-"),
}


def load_profiles(art):
    out = {}
    for m in PROFILE:
        f = art / "profiles" / f"S3__{m}.csv"
        if f.exists():
            t = pd.read_csv(f)
            t["h"] = 12.0 + t.step / 60.0                 # hours since 00:00 of the first day
            out[m] = t
    return out


def fig_profile(prof, out):
    """60-s operating profile of the representative S3 day (scripts/day_profile.py)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .diagrams import FONT
    plt.rcParams.update({**FONT, "font.size": 8, "axes.linewidth": 0.6, "axes.edgecolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 1.85), gridspec_kw={"width_ratios": [1.25, 1.15, 0.8]})
    ticks = [12, 18, 24, 30, 36]
    for m in ("uncoordinated", "price_aware+L3", "nested"):
        if m in prof:
            lab, col, ls = PROFILE[m]
            ax[0].plot(prof[m].h, prof[m].ev_kw / 1000.0, color=col, ls=ls, lw=1.6 if m == "nested" else 1.0, label=lab)
    if prof:
        any_t = next(iter(prof.values()))
        a2 = ax[0].twinx()
        a2.step(any_t.h, any_t.price_eur_kwh, where="post", color=NEUTRAL, lw=0.8, label="Day-ahead price")
        a2.set_ylabel("Price (€/kWh)", color=NEUTRAL)
        a2.tick_params(axis="y", colors=NEUTRAL)
        a2.spines[["top"]].set_visible(False)
    ax[0].set_ylabel("EV charging power (MW)")
    ax[0].set_title("(a) Charging power and price", fontsize=8, loc="left")
    for m in ("uncoordinated", "uncoordinated+L3", "nested"):
        if m in prof:
            lab, col, ls = PROFILE[m]
            ax[1].plot(prof[m].h, prof[m].vmin_pu, color=col, ls=ls, lw=1.6 if m == "nested" else 1.0, label=lab)
    ax[1].axhline(0.95, color="#c00000", lw=0.7, ls=":")
    ax[1].text(35.8, 0.9495, "0.95 p.u.", color="#c00000", fontsize=7, ha="right", va="top")
    ax[1].set_ylabel("Lowest bus voltage (p.u.)")
    ax[1].set_title("(b) Lowest bus voltage", fontsize=8, loc="left")
    if "uncoordinated+L3" in prof:
        lab, col, ls = PROFILE["uncoordinated+L3"]
        ax[2].fill_between(prof["uncoordinated+L3"].h, prof["uncoordinated+L3"].q_kvar, color=col, alpha=0.35, lw=0)
        ax[2].plot(prof["uncoordinated+L3"].h, prof["uncoordinated+L3"].q_kvar, color=col, lw=0.8, label=lab)
    ax[2].set_ylabel("Reactive power (kvar)")
    ax[2].set_title("(c) Level-3 reactive power", fontsize=8, loc="left")
    for a in ax:
        tk, lim = (ticks, (ticks[0], ticks[-1])) if a is not ax[2] else ([18, 20, 22], (17, 23))
        a.set_xticks(tk, [f"{t % 24:02d}:00" for t in tk], fontsize=7)
        a.set_xlim(*lim)
        a.grid(axis="y", lw=0.3, color="#d9d8d4")
        a.spines[["top", "right"]].set_visible(False) if a is not ax[0] else a.spines[["top"]].set_visible(False)
    handles, labels = [], []
    for a in list(ax) + ([a2] if prof else []):
        for h_, l_ in zip(*a.get_legend_handles_labels()):
            if l_ not in labels:
                handles.append(h_)
                labels.append(l_)
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.0),
               handlelength=2.4, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.2)
    save_checked(fig, out / "fig_profile.pdf", bbox_inches="tight")
    plt.close(fig)


def profile_numbers(prof, N, C):
    if not prof:
        return
    t0 = next(iter(prof.values()))
    N.add("profile day", pd.Timestamp(t0.day.iloc[0]).strftime("%-d %B %Y"), 0)
    for m, t in prof.items():
        N.add(f"profile peak {m}", t.ev_kw.max(), 0)
        N.add(f"profile vmin {m}", t.vmin_pu.min(), 3)
        N.add(f"profile cost {m}", float((t.ev_kw * t.price_eur_kwh).sum() / 60.0), 1)
    if "uncoordinated+L3" in prof:
        u = prof["uncoordinated+L3"]
        N.add("profile vmin pre unc l3", u.vmin_pre_pu.min(), 3)
        N.add("profile q max", u.q_kvar.max(), 0)
        N.add("profile q steps", int((u.q_kvar > 0).sum()), 0)
        N.add("profile q steps pct", 100.0 * (u.q_kvar > 0).mean(), 1)
        C.add("profile_unc_l3_keeps_floor", u.vmin_pu.min() >= 0.95 - 1e-9, f"{u.vmin_pu.min():.4f}")
        C.add("profile_unc_l3_no_curtailment", float(u.curtailed_kw.sum()) == 0.0)
    if "uncoordinated" in prof:
        C.add("profile_unc_violates", prof["uncoordinated"].vmin_pu.min() < 0.95)
    co = [m for m in ("nested", "price_aware+L3") if m in prof]
    if co:
        C.add("profile_coordinated_no_l3", all((prof[m].q_kvar > 0).sum() == 0 for m in co))
        C.add("profile_coordinated_above_floor", all(prof[m].vmin_pu.min() >= 0.95 for m in co))
    if "nested" in prof and "uncoordinated" in prof:
        C.add("profile_nested_peak_off_price_peak",
              prof["nested"].price_eur_kwh[prof["nested"].ev_kw.idxmax()] <
              prof["uncoordinated"].price_eur_kwh[prof["uncoordinated"].ev_kw.idxmax()])


# ------------------------------------------------------------------ numbers
MACRO_METRICS = [("cost", "cost_eur", 1), ("cpk", "cost_per_kwh", 4), ("sq", "service_quality", 3),
                 ("viol", "violation_rate_pct", 2), ("vmin", "min_voltage_pu", 4),
                 ("curt", "curtailed_kwh", 1), ("qact", "q_activation_pct", 2),
                 ("retail", "retail_price_paid", 3), ("unmet", "unmet_kwh", 1)]
LEARNED_ALL = ["flat_ddpg", "flat_ddpg+L3", "ppo_lag", "ppo_lag+L3", "cpo", "cpo+L3", "hrl", "hrl+L3"]


DESIGN_ABL = ["abl_no_prior", "abl_no_l1", "abl_flat_timescale", "abl_no_curriculum", "abl_no_guard",
              "abl_proportional"]


def numbers(s, d, t_cost, t_sq, compute_df, art, out, cfg, raw=None, spread=None):
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
        base = get("nested", sc, "abl_base")
        if base is None:
            continue
        for v in ABL_LABELS:
            if v == "abl_base":
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
            C.add(f"decomp_{mm}_more_unmet_than_plan_{sc}", r["unmet_kwh"] > plan["unmet_kwh"],
                  f"{r['unmet_kwh']:.1f} vs {plan['unmet_kwh']:.1f}")
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
    # service quality: lowest at the Caltech workplace site for every reference method
    sq_set = {m: {name: row(s, v, m, "S3", f, n, sp)["service_quality"] for v, f, n, sp, name in GENERAL
                  if row(s, v, m, "S3", f, n, sp) is not None}
              for m in ("uncoordinated", "price_aware+L3", "lp_opf+L3", "nested")}
    cal = "33-bus, ACN Caltech"
    if all(cal in v for v in sq_set.values()):
        C.add("gen_caltech_sq_lowest_all", all(v[cal] < min(x for k, x in v.items() if k != cal) for v in sq_set.values()),
              json.dumps({m: {k: round(float(x), 4) for k, x in v.items()} for m, v in sq_set.items()}))
        N.add("gen sq caltech max", max(v[cal] for v in sq_set.values()), 3, rnd="up")
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
        C.add("abl_no_behavior_higher_sq_S3", nb["service_quality"] > get("nested", "S3", "abl_base")["service_quality"])
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

    # training-seed spread of the mean daily cost at S3 (learned methods)
    if spread is not None and len(spread):
        sp = spread[(spread.variant == "main") & (spread.scenario == "S3") & (spread.fleet == "residential")
                    & (spread.network == "ieee33") & (spread.split == "test")].set_index("method").cost_eur
        base_l = [m for m in LEARNED_ALL if "+L3" not in m and m in sp.index]
        if "nested" in sp.index and base_l:
            N.add("seed std nested S3", sp["nested"], 1)
            N.add("seed std learned min S3", min(sp[m] for m in base_l), 0)
            N.add("seed std learned max S3", max(sp[m] for m in base_l), 0)
            C.add("seed_std_nested_smallest_S3", sp["nested"] < min(sp[m] for m in base_l),
                  json.dumps({m: round(float(sp[m]), 2) for m in ["nested"] + base_l}))
    # ablations: do the per-seed mean costs separate from those of the complete framework?
    if raw is not None:
        tr = raw[(raw.fleet == "residential") & (raw.network == "ieee33") & (raw.split == "test")]
        over = []
        for sc in ("S3", "S5"):
            b = tr[(tr.variant == "abl_base") & (tr.scenario == sc)].groupby("train_seed").cost_eur.mean()
            if b.empty:
                continue
            for v in ABL_LABELS:
                if v == "abl_base":
                    continue
                m = "nested-noL3" if v in ("abl_no_l3", "ablation") else "nested"
                a = tr[(tr.variant == v) & (tr.method == m) & (tr.scenario == sc)].groupby("train_seed").cost_eur.mean()
                if a.empty:
                    continue
                sep = bool(a.min() > b.max() or a.max() < b.min())
                info = f"{[round(float(x), 1) for x in a]} vs {[round(float(x), 1) for x in b]}"
                C.add(f"abl_{v}_seed_separated_{sc}", sep, info)
                C.add(f"abl_{v}_seed_overlap_{sc}", not sep, info)
                if v in DESIGN_ABL and not sep:
                    over.append(abs(100 * (a.mean() / b.mean() - 1)))
        if over:
            N.add("abl overlap delta max", max(over), 1, rnd="up")
    # statistics table: +L3 rows shown, Holm over all comparisons
    if len(t_cost):
        tc_i, ts_i = t_cost.set_index("method"), t_sq.set_index("method")
        pairs = [(m, m + "+L3") for m in tc_i.index if not m.endswith("+L3") and m + "+L3" in tc_i.index]
        N.add("holm family", len(t_cost), 0)
        N.add("stats l3 diff max", max(abs(tc_i.loc[a, "mean_diff"] - tc_i.loc[b, "mean_diff"]) for a, b in pairs), 1,
              rnd="up")
        C.add("stats_l3_same_cost_significance",
              all((tc_i.loc[a, "p_holm"] < 0.05) == (tc_i.loc[b, "p_holm"] < 0.05) for a, b in pairs))
        lower = [m for m in ts_i.index if ts_i.loc[m, "mean_diff"] < 0 and ts_i.loc[m, "p_holm"] < 0.05]
        C.add("nested_sq_sig_lower_only_unc_cpo_S3",
              sorted(lower) == sorted(["uncoordinated", "uncoordinated+L3", "cpo", "cpo+L3"]), str(lower))
        N.add("sq diff absmax S3", float(ts_i.mean_diff.abs().max()), 3, rnd="up")
        N.add("sq pholm max unc cpo S3", float(ts_i.loc[["uncoordinated", "uncoordinated+L3", "cpo", "cpo+L3"],
                                                        "p_holm"].max()), 3, rnd="up")
    # stress: in S7 only the perfect-foresight LP-OPF keeps the floor
    s7 = {m: get(m, "S7") for m in ["uncoordinated", "tou", "uncoordinated+L3", "tou+L3", "price_aware+L3",
                                    "lp_opf+L3", "nested"]}
    if all(v is not None for v in s7.values()):
        C.add("s7_only_lp_zero_viol", s7["lp_opf+L3"]["violation_rate_pct"] == 0.0
              and all(v["violation_rate_pct"] > 0 for m, v in s7.items() if m != "lp_opf+L3"),
              json.dumps({m: round(float(v["violation_rate_pct"]), 3) for m, v in s7.items()}))
    # generalization: workplace sessions
    for fl in ("acn_caltech", "acn_jpl"):
        g = {m: row(s, "general", m, "S3", fl, "ieee33", "test") for m in ("uncoordinated", "price_aware+L3", "lp_opf+L3")}
        if all(v is not None for v in g.values()):
            C.add(f"gen_unc_zero_viol_{fl}", g["uncoordinated"]["violation_rate_pct"] == 0.0)
            C.add(f"gen_unc_viol_below_001_{fl}", g["uncoordinated"]["violation_rate_pct"] < 0.01,
                  f"{g['uncoordinated']['violation_rate_pct']:.4f}")
            if fl == "acn_jpl":
                N.add("gen unc viol jpl", g["uncoordinated"]["violation_rate_pct"], 3, rnd="up")
            C.add(f"gen_pa_cheaper_than_lp_{fl}", g["price_aware+L3"]["cost_eur"] < g["lp_opf+L3"]["cost_eur"],
                  f"{g['price_aware+L3']['cost_eur']:.1f} vs {g['lp_opf+L3']['cost_eur']:.1f}")
    # reactive capability of one charger at full active power
    rpc = cfg["reactive_power"]
    N.add("q full power", math.sqrt(max(rpc["s_rated_kva"] ** 2 - rpc["charger_p_max_kw"] ** 2, 0.0)), 1)
    N.add("s rated", rpc["s_rated_kva"], 0)
    N.add("s rated S7", rpc["s_rated_kva"] * cfg["evaluation"]["scenarios"]["S7"]["s_rated_derate"], 0)
    N.add("p charger", rpc["charger_p_max_kw"], 0)
    N.add("transformer cap", cfg["aggregators"]["transformer_cap_kw"], 0)
    rt = cfg["retail"]                        # corridor of the neutral Level-1 action (0.5, 0.5)
    half = 0.5 * (rt["min_corridor_width"] + 0.5 * (rt["max_corridor_width"] - rt["min_corridor_width"]))
    N.add("neutral lo", cfg["behavior"]["lambda_ref"] - half, 3)
    N.add("neutral hi", cfg["behavior"]["lambda_ref"] + half, 3)
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
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    counts = {"train seeds": len(list((art / "runs").glob("nested_residential_ieee33_none_s*"))),
              "aux seeds": len(list((art / "runs").glob("nested_residential_ieee33_no_l1_s*")))}
    if tune_f.exists():
        counts["tune seeds"] = len(json.loads((art / "tuning" / "selected.json").read_text())["seeds"])
    for k, n in counts.items():                       # IEEE style: small counts spelled out in the text
        N.add(f"{k} word", words[n], 0)
        N.add(f"{k} word cap", words[n].capitalize(), 0)
    hh = cfg["aggregators"]["households"]
    if hh:
        for sc, spec in cfg["evaluation"]["scenarios"].items():      # penetration = EVs / households
            N.add(f"pen {sc}", round(100 * spec["n_ev"] / hh), 0)
    if compute_df is not None and len(compute_df):
        nest = compute_df[(compute_df.method == "nested") & (compute_df.ablation == "none")
                          & (compute_df.fleet == "residential") & (compute_df.network == "ieee33")]
        if len(nest):
            N.add("train hours nested", nest.total_h.mean(), 1)
    # grid side at S3: peaks and the power-flow cost of Level 3
    g = {m: get(m, "S3") for m in GRID_METHODS}
    if all(v is not None for v in g.values()):
        nst, unc, pa = g["nested"], g["uncoordinated"], g["price_aware+L3"]
        N.add("peak feeder red nested unc S3", 100 * (1 - nst["peak_feeder_mw"] / unc["peak_feeder_mw"]), 0)
        N.add("peak feeder red nested pa S3", 100 * (1 - nst["peak_feeder_mw"] / pa["peak_feeder_mw"]), 0)
        C.add("nested_lower_peak_feeder_than_unc_pa_S3",
              nst["peak_feeder_mw"] < min(unc["peak_feeder_mw"], pa["peak_feeder_mw"]))
        C.add("tou_highest_peak_ev_S3", max(g, key=lambda m: g[m]["peak_ev_kw"]) == "tou+L3")
        C.add("hrl_lowest_peak_feeder_S3", min(g, key=lambda m: g[m]["peak_feeder_mw"]) == "hrl+L3")
        base_pf = unc["pf_solves"]
        over = {m: 100 * (g[m]["pf_solves"] / base_pf - 1) for m in GRID_METHODS if m != "uncoordinated"}
        N.add("pf base S3", base_pf, 0)
        N.add("pf overhead nested S3", over["nested"], 1, rnd="up")
        N.add("pf overhead max S3", max(over.values()), 1, rnd="up")
        C.add("tou_l3_most_pf_S3", max(over, key=over.get) == "tou+L3")
    profile_numbers(load_profiles(art), N, C)
    N.write(out / "numbers.tex")
    C.write(out / "claims.json")
    return N


def build_all(art: pathlib.Path, out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df = load(art)
    # ablations use fewer seeds than the main comparison: compare them with the
    # complete framework on exactly the same training seeds ("abl_base")
    abl_seeds = sorted(df[df.variant.str.startswith("abl_")].train_seed.unique())
    base = df[(df.variant == "main") & (df.method == "nested") & df.train_seed.isin(abl_seeds)]
    df = pd.concat([df, base.assign(variant="abl_base")], ignore_index=True)
    d = per_day(df)
    s = summarise(d)
    s.to_csv(out / "summary.csv", index=False)
    spread = seed_spread(df)
    spread.to_csv(out / "seed_spread.csv", index=False)
    table_main(s, "S3", out / "tab_main_s3.tex")
    import yaml
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml"))
    hh = cfg["aggregators"]["households"]
    pen = {sc: round(100 * spec["n_ev"] / hh) for sc, spec in cfg["evaluation"]["scenarios"].items()}
    table_scenarios(s, ["uncoordinated", "tou", "uncoordinated+L3", "tou+L3", "price_aware+L3", "lp_opf+L3",
                        "flat_ddpg", "ppo_lag", "cpo", "hrl", "nested"],
                    ["S1", "S2", "S3", "S4", "S5"], pen, out / "tab_scenarios.tex")
    others = [m for m in ORDER if m != "nested"]
    t_cost = paired_tests(d, "nested", others, "cost_eur", "S3")
    t_sq = paired_tests(d, "nested", others, "service_quality", "S3")
    t_cost.to_csv(out / "tests_cost_s3.csv", index=False)
    t_sq.to_csv(out / "tests_sq_s3.csv", index=False)
    table_stats(t_cost, t_sq, out / "tab_stats.tex")
    table_ablation(s, ["S3", "S5"], out / "tab_ablation.tex")
    table_general(s, out / "tab_general.tex")
    table_stress(s, out / "tab_stress.tex")
    table_decomp(s, out / "tab_decomp.tex")
    compute_df = compute_times(art)
    calib = json.loads((art / "calibration.json").read_text()) if (art / "calibration.json").exists() else {}
    table_setup(cfg, calib, out / "tab_setup.tex")
    table_scen_def(cfg, out / "tab_scen_def.tex")
    table_grid(s, out / "tab_grid.tex")
    table_learned(s, spread, compute_df, out / "tab_learned.tex")
    figures(s, d, out)
    fig_profile(load_profiles(art), out)
    from .diagrams import build_diagrams
    table_hyper(cfg, out / "tab_hyper.tex")
    build_diagrams(cfg, out)
    from .diagrams import fig_pipeline
    fig_pipeline(cfg, out, len(list((art / "runs").glob("nested_residential_ieee33_none_s*"))),
                 len(list((art / "runs").glob("nested_residential_ieee33_no_l1_s*"))))
    numbers(s, d, t_cost, t_sq, compute_df, art, out, cfg, raw=df, spread=spread)
    print(f"== analysis: tables, figures and numbers written to {out}")
