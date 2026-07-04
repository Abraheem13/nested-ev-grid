"""Dataset calibration utilities.

Fits the arrival/duration/energy parameters consumed by FleetModel from raw
public datasets, writing a configs/dataset/*.yaml. Run once per dataset:

  ACN-Data (Caltech):  register at https://ev.caltech.edu, download sessions
      JSON/CSV for caltech or jpl site, then:
      python -m nflev.data.loaders acn  path/to/acn_sessions.csv
  ElaadNL open data:   https://platform.elaad.io (open charging transactions),
      python -m nflev.data.loaders elaad path/to/elaadnl_transactions.csv
  Prices: PJM DataMiner2 (da_hrl_lmps) / CAISO OASIS (PRC_LMP DAM) hourly CSV:
      python -m nflev.data.loaders prices path/to/lmp.csv --name pjm

The paper reports the FITTED parameters; raw data is never shipped in-repo.
"""
from __future__ import annotations
import argparse, pathlib, sys
import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]


def _truncnorm_fit(x: np.ndarray) -> tuple[float, float]:
    return float(np.mean(x)), float(np.std(x))


def fit_sessions(df: pd.DataFrame, name: str,
                 arrival_col: str, departure_col: str, kwh_col: str) -> dict:
    arr = pd.to_datetime(df[arrival_col])
    dep = pd.to_datetime(df[departure_col])
    arr_h = arr.dt.hour + arr.dt.minute / 60.0
    dur_h = (dep - arr).dt.total_seconds() / 3600.0
    kwh = df[kwh_col].astype(float)
    ok = (dur_h > 0.25) & (dur_h < 48) & (kwh > 0.5)
    arr_h, dur_h, kwh = arr_h[ok], dur_h[ok], kwh[ok]

    a_mu, a_sd = _truncnorm_fit(arr_h.values)
    d_mu, d_sd = _truncnorm_fit(dur_h.values)
    # map delivered energy to SoC-need distribution assuming 60 kWh median pack
    need_frac = np.clip(kwh.values / 60.0, 0.02, 0.95)
    soc_init_mu = float(np.clip(0.85 - need_frac.mean(), 0.1, 0.7))
    soc_init_sd = float(np.clip(need_frac.std(), 0.05, 0.25))

    cfg = {
        "name": name,
        "arrival_mu_h": round(a_mu, 2), "arrival_sigma_h": round(a_sd, 2),
        "arrival_min_h": round(float(np.percentile(arr_h, 1)), 2),
        "arrival_max_h": round(float(np.percentile(arr_h, 99)), 2),
        "duration_mu_h": round(d_mu, 2), "duration_sigma_h": round(d_sd, 2),
        "duration_min_h": round(float(np.percentile(dur_h, 1)), 2),
        "duration_max_h": round(float(np.percentile(dur_h, 99)), 2),
        "soc_init": [round(soc_init_mu, 2), round(soc_init_sd, 2)],
        "soc_target": [0.85, 0.10],
        "battery_kwh_classes": [40, 60, 75, 100],
        "price_source": "pjm" if name != "acn" else "caiso",
        "fitted_from_n_sessions": int(ok.sum()),
    }
    out = ROOT / f"configs/dataset/{name}.yaml"
    out.write_text(yaml.dump(cfg, sort_keys=False))
    print(f"wrote {out} (n={ok.sum()})")
    return cfg


def fit_acn(path: str) -> dict:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    return fit_sessions(df, "acn",
                        cols.get("connectiontime", "connectionTime"),
                        cols.get("disconnecttime", "disconnectTime"),
                        cols.get("kwhdelivered", "kWhDelivered"))


def fit_elaad(path: str) -> dict:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    start = cols.get("utctransactionstart", cols.get("started", "TransactionStartDT"))
    stop = cols.get("utctransactionstop", cols.get("ended", "TransactionStopDT"))
    kwh = cols.get("totalenergy", cols.get("connectedenergy", "TotalEnergy"))
    return fit_sessions(df, "elaad", start, stop, kwh)


def fit_prices(path: str, name: str) -> np.ndarray:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    price_col = next((cols[c] for c in
                      ["total_lmp_da", "lmp", "price", "mw", "lmp_prc"]
                      if c in cols), df.columns[-1])
    ts_col = next((cols[c] for c in
                   ["datetime_beginning_ept", "intervalstarttime_gmt",
                    "datetime", "time", "opr_hr"] if c in cols), df.columns[0])
    try:
        hours = pd.to_datetime(df[ts_col]).dt.hour
    except Exception:
        hours = df[ts_col].astype(int) - 1
    prof = df.groupby(hours)[price_col].mean().reindex(range(24)).interpolate()
    arr = prof.values.astype(float)
    out = ROOT / f"configs/prices_{name}.yaml"
    out.write_text(yaml.dump({"name": name, "hourly_lmp_usd_mwh":
                              [round(float(v), 2) for v in arr]}))
    print(f"wrote {out}")
    return arr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["acn", "elaad", "prices"])
    ap.add_argument("path")
    ap.add_argument("--name", default="pjm")
    a = ap.parse_args()
    if a.kind == "acn":
        fit_acn(a.path)
    elif a.kind == "elaad":
        fit_elaad(a.path)
    else:
        fit_prices(a.path, a.name)