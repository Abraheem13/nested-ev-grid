#!/usr/bin/env python3
"""Verify every headline claim of the manuscript against fresh evaluation
CSVs, and regenerate the paper's LaTeX tables from those CSVs only.

Reads (produced by scripts/run_campaign.py):
    results/eval/main_ieee33_nrel.csv     8 methods x S1-S5 x 50 episodes
    results/eval/stress_ieee33_nrel.csv   nested x {S6, S7}
    results/eval/abl_*.csv                ablation rows on S3
    results/eval/ieee69_nrel.csv          cross-network block
    results/eval/dataset_{acn,elaad}.csv  cross-dataset block
    results/eval/sens_*.csv               sensitivity sweeps

Writes:
    results/tables/paper_table_main.tex        (Table IV)
    results/tables/paper_table_pareto.tex      (Table Pareto)
    results/tables/paper_table_stats.tex       (Table VI)
    results/tables/paper_table_ablation.tex    (Table V)
    results/tables/paper_table_general.tex     (Table VII)
    results/tables/paper_table_sensitivity.tex (Table VIII, delta & S rows)
    results/claims_report.md                   PASS/FAIL per claim

Exit status is non-zero if any MUST-HOLD claim fails, so this script can act
as the final gate of the campaign. The numeric VALUES in the paper must then
be updated from these .tex files - the claims here check the paper's
QUALITATIVE assertions (zero violations, delivery >= 0.95, Pareto uniqueness,
statistical significance), which is what reviewers will probe.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from nflev.eval.stats import welch, holm, ci95  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL = ROOT / "results" / "eval"
TABLES = ROOT / "results" / "tables"
REPORT = ROOT / "results" / "claims_report.md"

LABELS = {"uncoordinated": "Uncoordinated", "tou": "TOU Pricing",
          "milp": "Centralized OPF", "flat_ddpg": "Flat DDPG",
          "ppo_lag": "PPO-Lagrangian", "cpo": "CPO",
          "hrl": "Hierarchical RL", "nested": "Nested Learning (Proposed)"}
ORDER = ["uncoordinated", "tou", "milp", "flat_ddpg", "ppo_lag", "cpo",
         "hrl", "nested"]
SCEN_TITLES = {"S1": "S1: 40\\%", "S2": "S2: 67\\%", "S3": "S3: 80\\%",
               "S4": "S4: 80\\%+noise", "S5": "S5: 100\\%"}

PARETO_SQ = 0.95        # "complete" threshold used in the paper
PARETO_VIOL = 0.05      # "safe" threshold (%): violation rate < 0.05%


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
class Claims:
    def __init__(self):
        self.rows: list[tuple[str, str, bool, str]] = []

    def check(self, cid: str, desc: str, ok: bool, detail: str = ""):
        self.rows.append((cid, desc, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {cid}: {desc}"
              + (f"  ({detail})" if detail else ""))

    @property
    def failed(self) -> list[tuple]:
        return [r for r in self.rows if not r[2]]

    def markdown(self) -> str:
        lines = ["# Claims verification report", "",
                 "| ID | Claim | Result | Detail |",
                 "|----|-------|--------|--------|"]
        for cid, desc, ok, det in self.rows:
            lines.append(f"| {cid} | {desc} | "
                         f"{'PASS' if ok else '**FAIL**'} | {det} |")
        return "\n".join(lines) + "\n"


def agg(df: pd.DataFrame) -> pd.DataFrame:
    """Per (scenario, method): mean and 95% CI of the four paper metrics."""
    out = []
    for (sc, me), g in df.groupby(["scenario", "method"]):
        row = {"scenario": sc, "method": me, "n": len(g)}
        for m in ["daily_cost_usd", "min_voltage_pu", "violation_rate_pct",
                  "service_quality"]:
            x = g[m].astype(float).values
            row[m], row[m + "_ci"] = x.mean(), ci95(x)
        out.append(row)
    return pd.DataFrame(out)


def need(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"missing evaluation file: {path} - run the campaign first")
    df = pd.read_csv(path)
    return df


def f2(x): return f"{x:.2f}"
def f4(x): return f"{x:.4f}"
def f3(x): return f"{x:.3f}"


# --------------------------------------------------------------------------- #
# LaTeX emitters (formats match main.tex exactly)
# --------------------------------------------------------------------------- #
def emit_table_main(a: pd.DataFrame) -> str:
    body = []
    for sc in ["S1", "S2", "S3", "S4", "S5"]:
        first = True
        for me in ORDER:
            r = a[(a.scenario == sc) & (a.method == me)]
            if r.empty:
                continue
            r = r.iloc[0]
            cells = (f"{f2(r.daily_cost_usd)}$\\pm${f2(r.daily_cost_usd_ci)} & "
                     f"{f4(r.min_voltage_pu)}$\\pm${f4(r.min_voltage_pu_ci)} & "
                     f"{r.violation_rate_pct:.1f}$\\pm${r.violation_rate_pct_ci:.1f} & "
                     f"{f3(r.service_quality)}$\\pm${f3(r.service_quality_ci)}")
            if me == "nested":
                cells = " & ".join(f"\\textbf{{{c.strip()}}}"
                                   for c in cells.split("&"))
            lead = SCEN_TITLES[sc] if first else ""
            name = (f"\\textbf{{{LABELS[me]}}}" if me == "nested"
                    else LABELS[me])
            body.append(f"{lead} & {name} & {cells} \\\\")
            first = False
        body.append("\\midrule")
    body = body[:-1]
    return "\n".join(body) + "\n"


def emit_table_pareto(a: pd.DataFrame) -> str:
    rows = []
    for me in ORDER:
        r = a[(a.scenario == "S3") & (a.method == me)].iloc[0]
        safe = r.violation_rate_pct < PARETO_VIOL
        complete = r.service_quality >= PARETO_SQ
        c = lambda b: "\\checkmark" if b else "$\\times$"
        name = ("\\textbf{Nested (Proposed)}" if me == "nested"
                else LABELS[me])
        v = (f"\\textbf{{{r.violation_rate_pct:.1f}}}" if me == "nested"
             else f"{r.violation_rate_pct:.1f}")
        s = (f"\\textbf{{{f3(r.service_quality)}}}" if me == "nested"
             else f3(r.service_quality))
        both = (f"\\textbf{{{c(safe and complete)}}}" if me == "nested"
                else c(safe and complete))
        rows.append(f"{name} & {v} & {s} & {c(safe)} & {c(complete)} & {both} \\\\")
    return "\n".join(rows) + "\n"


def emit_table_stats(df: pd.DataFrame, a: pd.DataFrame) -> str:
    g3 = df[df.scenario == "S3"]
    ref = g3[g3.method == "nested"]["daily_cost_usd"].astype(float).values
    ps, rows = [], []
    for me in ORDER[:-1]:
        x = g3[g3.method == me]["daily_cost_usd"].astype(float).values
        p, _ = welch(x, ref)
        ps.append(p)
    p_adj = holm(ps)
    for me, p in zip(ORDER[:-1], p_adj):
        r = a[(a.scenario == "S3") & (a.method == me)].iloc[0]
        ptxt = "$<0.001$" if p < 0.001 else f"${p:.3f}$"
        rows.append(f"{LABELS[me]} & {f2(r.daily_cost_usd)} & "
                    f"{f4(r.min_voltage_pu)} & "
                    f"{r.violation_rate_pct:.1f} & {ptxt} \\\\")
    rn = a[(a.scenario == "S3") & (a.method == "nested")].iloc[0]
    rows.append("\\midrule")
    rows.append(f"\\textbf{{Nested (ref.)}} & \\textbf{{{f2(rn.daily_cost_usd)}}}"
                f" & \\textbf{{{f4(rn.min_voltage_pu)}}} & "
                f"\\textbf{{{rn.violation_rate_pct:.1f}}} & -- \\\\")
    return "\n".join(rows) + "\n", dict(zip(ORDER[:-1], p_adj))


def emit_table_ablation(main_s3: pd.Series, abl: dict[str, pd.Series]) -> str:
    def objectives(r):
        cost_ok = r.daily_cost_usd <= main_s3.daily_cost_usd * 1.05
        v_ok = r.min_voltage_pu >= 0.95 - 1e-9 and r.violation_rate_pct < PARETO_VIOL
        sq_ok = r.service_quality >= PARETO_SQ
        return sum([cost_ok, v_ok, sq_ok])

    label = {"abl_no_l1": "No Level 1 (fixed price)",
             "abl_no_l3": "No Level 3 (no reactive)",
             "abl_no_user_model": "No user model (L3a)",
             "abl_flat_timescale": "Flat timescale (all 15m)",
             "abl_no_curriculum": "No curriculum"}
    rows = [f"\\textbf{{Complete framework}} & "
            f"\\textbf{{{main_s3.daily_cost_usd:.1f}}} & -- & "
            f"\\textbf{{{f4(main_s3.min_voltage_pu)}}} & \\textbf{{3/3}} \\\\"]
    for tag in ["abl_no_l1", "abl_no_l3", "abl_no_user_model",
                "abl_flat_timescale", "abl_no_curriculum"]:
        r = abl[tag]
        d = 100 * (r.daily_cost_usd / main_s3.daily_cost_usd - 1)
        rows.append(f"{label[tag]} & {r.daily_cost_usd:.1f} & "
                    f"{'+' if d >= 0 else ''}{d:.1f} & "
                    f"{f4(r.min_voltage_pu)} & {objectives(r)}/3 \\\\")
    return "\n".join(rows) + "\n"


def emit_table_general(a69: pd.DataFrame, ds: dict[str, pd.Series],
                       main_s3: pd.Series) -> str:
    rows = ["\\multicolumn{5}{@{}l}{\\emph{IEEE 69-bus feeder}}\\\\"]
    for me in ["uncoordinated", "tou", "nested"]:
        r = a69[a69.method == me].iloc[0]
        name = ("Nested Learning (Proposed)" if me == "nested" else LABELS[me])
        rows.append(f"~~{name} & & {r.daily_cost_usd:.1f} & "
                    f"{f4(r.min_voltage_pu)} & {r.violation_rate_pct:.1f} \\\\")
    rows.append("\\midrule")
    rows.append("\\multicolumn{5}{@{}l}{\\emph{IEEE 33-bus, "
                "alternate fleets (proposed)}}\\\\")
    rows.append(f"~~NREL EV Project & & {main_s3.daily_cost_usd:.1f} & "
                f"{f4(main_s3.min_voltage_pu)} & "
                f"{main_s3.violation_rate_pct:.1f} \\\\")
    for name, key in [("ACN-Data (Caltech)", "acn"), ("ElaadNL", "elaad")]:
        r = ds[key]
        rows.append(f"~~{name} & & {r.daily_cost_usd:.1f} & "
                    f"{f4(r.min_voltage_pu)} & {r.violation_rate_pct:.1f} \\\\")
    return "\n".join(rows) + "\n"


def emit_table_sensitivity(main_s3: pd.Series,
                           sens: dict[str, pd.Series]) -> str:
    def row(val, r, default=False):
        v = f"{val} (default)" if default else val
        return (f" & ${v}$ & {r.daily_cost_usd:.1f} & "
                f"{f4(r.min_voltage_pu)} & {f3(r.service_quality)} \\\\")
    rows = ["\\multirow{3}{*}{Reserve margin $\\delta$}",
            row("0.0005", sens["delta_0p0005"]),
            row("0.001", main_s3, default=True),
            row("0.002", sens["delta_0p002"]),
            "\\midrule",
            "\\multirow{3}{*}{Inverter rating $S$ (kVA)}",
            row("10", sens["srated_10"]),
            row("12", main_s3, default=True),
            row("14", sens["srated_14"])]
    return "\n".join(rows) + "\n"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    C = Claims()

    # ---------------- load everything ----------------------------------------
    df_main = need(EVAL / "main_ieee33_nrel.csv")
    a_main = agg(df_main)
    df_stress = need(EVAL / "stress_ieee33_nrel.csv")
    a_stress = agg(df_stress)
    a69 = agg(need(EVAL / "ieee69_nrel.csv"))
    abl = {t: agg(need(EVAL / f"{t}.csv")).iloc[0]
           for t in ["abl_no_l1", "abl_no_l3", "abl_no_user_model",
                     "abl_flat_timescale", "abl_no_curriculum"]}
    ds = {k: agg(need(EVAL / f"dataset_{k}.csv")).iloc[0]
          for k in ["acn", "elaad"]}
    sens = {t: agg(need(EVAL / f"sens_{t}.csv")).iloc[0]
            for t in ["delta_0p0005", "delta_0p002", "srated_10", "srated_14"]}
    main_s3 = a_main[(a_main.scenario == "S3")
                     & (a_main.method == "nested")].iloc[0]

    # ---------------- protocol integrity -------------------------------------
    print("\n== Protocol integrity ==")
    n_ok = (a_main["n"] == 50).all()
    C.check("P1", "every (scenario, method) cell has exactly 50 episodes",
            n_ok, f"min n = {a_main['n'].min()}")
    C.check("P2", "evaluation seeds follow the stated 7000..7049 protocol",
            set(df_main["seed"].unique()) == set(range(7000, 7050)))
    C.check("P3", "no infeasible/failed episodes in the main grid",
            not df_main.get("infeasible", pd.Series(False)).fillna(False)
                .astype(bool).any())

    # ---------------- headline claims (abstract / results) -------------------
    print("\n== Headline claims ==")
    nested = a_main[a_main.method == "nested"]
    C.check("H1", "nested: ZERO recorded violations across S1-S5",
            (nested.violation_rate_pct < 1e-9).all(),
            f"max viol = {nested.violation_rate_pct.max():.3f}%")
    C.check("H2", "nested: min voltage >= 0.95 p.u. in every scenario mean",
            (nested.min_voltage_pu >= 0.95 - 1e-6).all(),
            f"min = {nested.min_voltage_pu.min():.4f}")
    sq_lo, sq_hi = nested.service_quality.min(), nested.service_quality.max()
    C.check("H3", "nested: delivers 97-98% of requested energy (paper claim)",
            sq_lo >= 0.97 - 5e-3,
            f"SQ range {sq_lo:.3f}-{sq_hi:.3f}; if outside 0.97-0.98, "
            "update the abstract wording")
    others = a_main[a_main.method != "nested"]
    pareto_others = others[(others.violation_rate_pct < PARETO_VIOL)
                           & (others.service_quality >= PARETO_SQ)]
    C.check("H4", "Pareto uniqueness: NO baseline is both safe and complete "
                  "in any scenario",
            pareto_others.empty,
            "" if pareto_others.empty else
            pareto_others[["scenario", "method"]].to_string(index=False))
    unc = a_main[a_main.method.isin(["uncoordinated", "tou"])]
    C.check("H5", "uncoordinated/TOU violate at high penetration (S3-S5)",
            (unc[unc.scenario.isin(["S3", "S4", "S5"])]
             .violation_rate_pct > 1.0).all())

    # ---------------- statistics ----------------------------------------------
    print("\n== Statistical separation (S3, Welch + Holm) ==")
    stats_tex, p_adj = emit_table_stats(df_main, a_main)
    C.check("T1", "all S3 cost comparisons Holm-significant at p<0.05",
            all(p < 0.05 for p in p_adj.values() if not np.isnan(p)),
            "; ".join(f"{m}:{p:.4g}" for m, p in p_adj.items()))
    v_sd = (df_main[(df_main.scenario == "S3") & (df_main.method == "nested")]
            ["min_voltage_pu"].std(ddof=1))
    C.check("T2", "near-deterministic voltage outcome (sigma <= 0.0003 p.u.)",
            v_sd <= 3e-4, f"sigma = {v_sd:.2e}")

    # ---------------- stress scenarios ---------------------------------------
    print("\n== Stress scenarios ==")
    s6 = a_stress[a_stress.scenario == "S6_deadlock"].iloc[0]
    s7 = a_stress[a_stress.scenario == "S7_q_exhaustion"].iloc[0]
    C.check("S6", "deadlock scenario: zero violations with forced P=0 window",
            s6.violation_rate_pct < 1e-9,
            f"viol = {s6.violation_rate_pct:.3f}%, SQ = {s6.service_quality:.3f}")
    C.check("S7", "capacity exhaustion: graceful degradation "
                  "(0 < viol <= 2%, vmin >= 0.94)",
            0 < s7.violation_rate_pct <= 2.0 and s7.min_voltage_pu >= 0.94,
            f"viol = {s7.violation_rate_pct:.2f}%, "
            f"vmin = {s7.min_voltage_pu:.4f}")

    # ---------------- ablations ----------------------------------------------
    print("\n== Ablations (S3) ==")
    C.check("A1", "removing Level 3 breaks the voltage floor "
                  "(vmin < 0.95, violations reappear)",
            abl["abl_no_l3"].min_voltage_pu < 0.95
            and abl["abl_no_l3"].violation_rate_pct > 1.0,
            f"vmin = {abl['abl_no_l3'].min_voltage_pu:.4f}, "
            f"viol = {abl['abl_no_l3'].violation_rate_pct:.1f}%")
    C.check("A2", "every ablation is strictly worse than the complete "
                  "framework on cost",
            all(abl[t].daily_cost_usd > main_s3.daily_cost_usd
                for t in abl))

    # ---------------- generalization -----------------------------------------
    print("\n== Generalization ==")
    n69 = a69[a69.method == "nested"].iloc[0]
    C.check("G1", "IEEE 69-bus: nested keeps zero violations",
            n69.violation_rate_pct < 1e-9,
            f"vmin = {n69.min_voltage_pu:.4f}")
    C.check("G2", "cross-dataset (ACN, ElaadNL): zero violations, SQ >= 0.95",
            all(ds[k].violation_rate_pct < 1e-9
                and ds[k].service_quality >= PARETO_SQ for k in ds),
            "; ".join(f"{k}: SQ={ds[k].service_quality:.3f}" for k in ds))

    # ---------------- sensitivity --------------------------------------------
    print("\n== Sensitivity ==")
    C.check("V1", "zero violations across the delta and S-rating sweeps",
            all(sens[t].violation_rate_pct < 1e-9 for t in sens))

    # ---------------- emit LaTeX ----------------------------------------------
    outputs = {
        "paper_table_main.tex": emit_table_main(a_main),
        "paper_table_pareto.tex": emit_table_pareto(a_main),
        "paper_table_stats.tex": stats_tex,
        "paper_table_ablation.tex": emit_table_ablation(main_s3, abl),
        "paper_table_general.tex": emit_table_general(a69, ds, main_s3),
        "paper_table_sensitivity.tex": emit_table_sensitivity(main_s3, sens),
    }
    for name, tex in outputs.items():
        (TABLES / name).write_text(tex)
        print(f"wrote {TABLES / name}")

    REPORT.write_text(C.markdown())
    print(f"\nwrote {REPORT}")

    if C.failed:
        print(f"\n{len(C.failed)} claim(s) FAILED - the manuscript text must "
              "be revised to match the data before submission:")
        for cid, desc, _, det in C.failed:
            print(f"  - {cid}: {desc}  ({det})")
        sys.exit(1)
    print("\nAll manuscript claims verified against fresh evaluation data.")
    print("Next step: replace the numeric cells in main.tex with the "
          "generated rows in results/tables/paper_table_*.tex.")


if __name__ == "__main__":
    main()