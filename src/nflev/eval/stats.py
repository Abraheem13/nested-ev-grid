"""Statistical analysis of evaluation results.

Produces the Table IV / Table VI content for the paper:
  - per scenario x method: mean +/- 95% CI over episodes
  - Welch two-sample t-tests against the nested reference
  - Holm-Bonferroni correction across the comparison family (Q1 reviewers
    increasingly expect multiplicity control; v1 reported raw p-values)
  - Cohen's d effect sizes

Usage:
  python -m nflev.eval.stats results/eval_ieee33_nrel.csv --out results/tables
"""
from __future__ import annotations
import argparse, pathlib
import numpy as np
import pandas as pd
from scipy import stats as sps

METRICS = ["daily_cost_usd", "min_voltage_pu", "violation_rate_pct",
           "service_quality"]
REF = "nested"


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2 or (a.std() == 0 and b.std() == 0):
        return float("nan"), float("nan")
    t, p = sps.ttest_ind(a, b, equal_var=False)
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    d = (a.mean() - b.mean()) / sp if sp > 0 else float("inf")
    return float(p), float(d)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (NaNs passed through)."""
    idx = [i for i, p in enumerate(pvals) if not np.isnan(p)]
    m = len(idx)
    order = sorted(idx, key=lambda i: pvals[i])
    adj = list(pvals)
    prev = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        prev = max(prev, val)
        adj[i] = prev
    return adj


def ci95(x: np.ndarray) -> float:
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def analyze(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scen, g in df.groupby("scenario"):
        ref = g[g.method == REF]
        for method, gm in g.groupby("method"):
            row = {"scenario": scen, "method": method, "n": len(gm)}
            for m in METRICS:
                x = gm[m].dropna().values
                row[f"{m}_mean"] = x.mean() if len(x) else np.nan
                row[f"{m}_ci"] = ci95(x) if len(x) else np.nan
                if method != REF and len(ref):
                    p, d = welch(x, ref[m].dropna().values)
                    row[f"{m}_p_raw"] = p
                    row[f"{m}_d"] = d
            rows.append(row)
    out = pd.DataFrame(rows)
    # Holm correction per metric across all (scenario, method) comparisons
    for m in METRICS:
        col = f"{m}_p_raw"
        if col in out:
            out[f"{m}_p_holm"] = holm(out[col].fillna(np.nan).tolist())
    return out


def fmt_p(p: float) -> str:
    if np.isnan(p):
        return "--"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def latex_table_iv(res: pd.DataFrame, path: pathlib.Path):
    """Aggregate performance table (Table IV style)."""
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Scenario & Method & Daily Cost (\$) & Min V (p.u.) & "
             r"Viol.\ (\%) & Service Quality \\", r"\midrule"]
    for scen, g in res.groupby("scenario"):
        for _, r in g.sort_values("daily_cost_usd_mean", ascending=False).iterrows():
            bold = r.method == REF
            def cell(m, prec):
                v, c = r[f"{m}_mean"], r[f"{m}_ci"]
                if np.isnan(v):
                    return "--"
                s = f"{v:.{prec}f} $\\pm$ {c:.{prec}f}"
                return f"\\textbf{{{s}}}" if bold else s
            lines.append(f"{scen} & {r.method} & {cell('daily_cost_usd',2)} & "
                         f"{cell('min_voltage_pu',4)} & "
                         f"{cell('violation_rate_pct',1)} & "
                         f"{cell('service_quality',3)} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines))


def latex_table_vi(res: pd.DataFrame, scenario: str, path: pathlib.Path):
    """Statistical comparison table (Table VI style) with Holm-adjusted p."""
    g = res[res.scenario == scenario]
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Method & Cost (\$) & $p$ & Min V & $p$ & Viol.\ \% & $p$ \\",
             r"\midrule"]
    for _, r in g.iterrows():
        if r.method == REF:
            continue
        lines.append(
            f"{r.method} & {r.daily_cost_usd_mean:.2f} & "
            f"{fmt_p(r.get('daily_cost_usd_p_holm', np.nan))} & "
            f"{r.min_voltage_pu_mean:.4f} & "
            f"{fmt_p(r.get('min_voltage_pu_p_holm', np.nan))} & "
            f"{r.violation_rate_pct_mean:.1f} & "
            f"{fmt_p(r.get('violation_rate_pct_p_holm', np.nan))} \\\\")
    rr = g[g.method == REF]
    if len(rr):
        r = rr.iloc[0]
        lines.append(f"\\textbf{{{REF} (ref.)}} & \\textbf{{{r.daily_cost_usd_mean:.2f}}} & -- & "
                     f"\\textbf{{{r.min_voltage_pu_mean:.4f}}} & -- & "
                     f"\\textbf{{{r.violation_rate_pct_mean:.1f}}} & -- \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="results/tables")
    ap.add_argument("--table6_scenario", default="S3")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.csv)
    res = analyze(df)
    res.to_csv(out / "stats_full.csv", index=False)
    latex_table_iv(res, out / "table_iv.tex")
    if (df.scenario == a.table6_scenario).any():
        latex_table_vi(res, a.table6_scenario, out / "table_vi.tex")
    print(res[["scenario", "method", "daily_cost_usd_mean", "min_voltage_pu_mean",
               "violation_rate_pct_mean", "service_quality_mean"]]
          .to_string(index=False))
    print(f"\nwrote {out}/stats_full.csv, table_iv.tex, table_vi.tex")