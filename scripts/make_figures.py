#!/usr/bin/env python3
"""Publication figures — all vector PDF, colorblind-safe (Okabe-Ito), >=8pt.

Modes:
  training  : learning curves from results/*/train_log.csv
  aggregate : Table-IV-style grouped bars with 95% CI from an eval CSV
  profile   : 24-h temporal profile (Fig. 5 replacement) — runs ONE
              deterministic nested episode with a checkpoint and dumps +
              plots power/voltage/corridor traces

Usage:
  python scripts/make_figures.py training  --runs results/nested_v3_s0 results/flat_ddpg_ieee33_nrel_s0 --out figs
  python scripts/make_figures.py aggregate --csv results/eval_ieee33_nrel.csv --out figs
  python scripts/make_figures.py profile   --checkpoint results/nested_v3_s0 --out figs
"""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
             "#E69F00", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 200, "savefig.bbox": "tight",
    "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
    "pdf.fonttype": 42,  # embed TrueType (IEEE/ACM compliance)
})

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOAD_SHAPE = np.array([.55,.5,.48,.47,.48,.55,.65,.75,.72,.70,.68,.67,.66,.68,.72,.80,.92,1.0,1.0,.97,.88,.78,.68,.60])


def smooth(x, w=15):
    if len(x) < w:
        return x
    return pd.Series(x).rolling(w, min_periods=1).mean().values


def fig_training(runs, out):
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2))  # \textwidth double col
    for run in runs:
        df = pd.read_csv(pathlib.Path(run) / "train_log.csv")
        label = pathlib.Path(run).name.split("_")[0]
        axes[0].plot(smooth(df.daily_cost_usd), label=label, lw=1.2)
        axes[1].plot(smooth(df.service_quality), lw=1.2)
        axes[2].plot(smooth(df.min_voltage_pu), lw=1.2)
    axes[0].set(xlabel="Episode", ylabel="Daily cost (\\$)")
    axes[1].set(xlabel="Episode", ylabel="Service quality", ylim=(0, 1.02))
    axes[2].set(xlabel="Episode", ylabel="Min voltage (p.u.)")
    axes[2].axhline(0.95, color="k", ls="--", lw=0.8)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_training.pdf")
    print("wrote fig_training.pdf")


def fig_aggregate(csv, out):
    df = pd.read_csv(csv)
    metrics = [("daily_cost_usd", "Daily cost (\\$)"),
               ("min_voltage_pu", "Min voltage (p.u.)"),
               ("violation_rate_pct", "Violation rate (\\%)"),
               ("service_quality", "Service quality")]
    scens = sorted(df.scenario.unique())
    methods = [m for m in ["uncoordinated", "tou", "milp", "flat_ddpg",
                           "ppo_lag", "cpo", "hrl", "nested"]
               if m in df.method.unique()]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.4))
    w = 0.8 / len(methods)
    for ax, (m, lab) in zip(axes.flat, metrics):
        for j, meth in enumerate(methods):
            means, cis = [], []
            for s in scens:
                x = df[(df.scenario == s) & (df.method == meth)][m].dropna()
                means.append(x.mean() if len(x) else np.nan)
                cis.append(1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0)
            pos = np.arange(len(scens)) + (j - len(methods) / 2 + 0.5) * w
            ax.bar(pos, means, w, yerr=cis, capsize=1.5, label=meth,
                   error_kw={"lw": 0.7})
        ax.set_xticks(range(len(scens)), scens)
        ax.set_ylabel(lab)
        if m == "min_voltage_pu":
            ax.axhline(0.95, color="k", ls="--", lw=0.8)
            ax.set_ylim(0.82, 0.97)
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "fig_aggregate.pdf")
    print("wrote fig_aggregate.pdf")


def fig_profile(checkpoint, out, scenario="S3"):
    import yaml
    from nflev.env.charging_env import ChargingEnv
    from nflev.training.trainer import NestedTrainer

    cfg = yaml.safe_load(open(ROOT / "configs/base.yaml"))
    ds = yaml.safe_load(open(ROOT / "configs/dataset/nrel.yaml"))
    scen = cfg["evaluation"]["scenarios"][scenario]
    price = np.array([28,26,25,25,26,30,45,60,55,50,48,47,46,48,52,60,80,110,120,105,85,60,45,35], float)

    env = ChargingEnv(cfg, "ieee33", ds, n_ev=scen["n_ev"], seed=7)
    ck = pathlib.Path(checkpoint)
    tr = NestedTrainer(cfg, lambda st, sd: env)
    tr.l1.load(ck / "l1_final.pt")
    for k, ag in enumerate(tr.l2):
        ag.load(ck / f"l2_{k}_final.pt")

    corridor_trace, ev_kw_trace = [], []
    orig_set_corridor = env.set_corridor
    def rec_corridor(pmin, pmax):
        orig_set_corridor(pmin, pmax)
        corridor_trace.append((env.t_s / 3600.0, *env.corridor))
    env.set_corridor = rec_corridor
    orig_run = env.run_dispatch_interval
    def rec_run():
        ev_kw_trace.append((env.t_s / 3600.0,
                            sum(sum(r.values()) for r in env.agg_rates.values())))
        return orig_run()
    env.run_dispatch_interval = rec_run

    tr.run_episode(seed=7, train=False, scenario=scen,
                   price_profile=price, load_profile=LOAD_SHAPE)

    logs = pd.DataFrame([vars(l) for l in env.logs])
    logs["wall_h"] = (logs.t_h + env.wall_offset_h) % 24
    logs = logs.sort_values("t_h")
    ckw = pd.DataFrame(ev_kw_trace, columns=["t_h", "ev_kw"])
    ckw["wall_h"] = (ckw.t_h + env.wall_offset_h) % 24
    cor = pd.DataFrame(corridor_trace, columns=["t_h", "pmin", "pmax"])
    cor["wall_h"] = (cor.t_h + env.wall_offset_h) % 24
    logs.to_csv(out / "profile_logs.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(3.5, 4.6), sharex=True)
    o = np.argsort(logs.t_h.values)
    axes[0].plot(logs.t_h, logs.p_total_mw, lw=1.0, label="Total load")
    ax0b = axes[0].twinx()
    ax0b.plot(ckw.t_h, ckw.ev_kw / 1000, lw=1.0, color=OKABE_ITO[1], label="EV charging")
    axes[0].set_ylabel("System load (MW)")
    ax0b.set_ylabel("EV charging (MW)", color=OKABE_ITO[1])
    axes[1].plot(logs.t_h, logs.v_min, lw=1.0)
    axes[1].axhline(0.95, color="k", ls="--", lw=0.8)
    axes[1].set_ylabel("Min voltage (p.u.)")
    axes[2].step(cor.t_h, cor.pmin, where="post", lw=1.0, label="$p^{min}$")
    axes[2].step(cor.t_h, cor.pmax, where="post", lw=1.0, label="$p^{max}$")
    axes[2].set(xlabel="Episode hour (0 = 12:00)", ylabel="Corridor (\\$/kWh)")
    axes[2].legend(frameon=False)
    q = logs[logs.q_injected_kvar > 0]
    if len(q):
        axes[1].scatter(q.t_h, q.v_min, s=6, color=OKABE_ITO[1], zorder=3,
                        label="Q active")
        axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_profile.pdf")
    print("wrote fig_profile.pdf + profile_logs.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["training", "aggregate", "profile"])
    ap.add_argument("--runs", nargs="+", default=[])
    ap.add_argument("--csv", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--scenario", default="S3")
    ap.add_argument("--out", default="figs")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == "training":
        fig_training(a.runs, out)
    elif a.mode == "aggregate":
        fig_aggregate(a.csv, out)
    else:
        fig_profile(a.checkpoint, out, a.scenario)