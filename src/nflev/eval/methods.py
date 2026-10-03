"""Registry of every evaluated method.

A method name is `<base>` or `<base>+L3`. `+L3` evaluates the same policy with
the Level-3 reactive controller switched on. Rule-based and LP methods are
evaluated in both forms; learned baselines are trained without Level 3 and
evaluated without it (main comparison) and with it (deployment variant). The
nested framework includes Level 3 by design; `nested-noL3` switches it off at
evaluation time only.
"""
from __future__ import annotations

import pathlib

import torch

from ..baselines.flat_ddpg import FlatDDPG
from ..baselines.hrl import HRL
from ..baselines.milp import LPOPF
from ..baselines.safe_rl import SafeRL
from ..baselines.simple import NoCharging, PlanLLF, PriceAware, TOUTimer, Uncoordinated
from ..training.trainer import ablated_cfg, load_nested

RULES = {"uncoordinated": Uncoordinated, "tou": TOUTimer, "price_aware": PriceAware, "lp_opf": LPOPF,
         "plan": PlanLLF, "noev": NoCharging}
LEARNED_BASELINES = {"flat_ddpg", "ppo_lag", "cpo", "hrl"}
LABELS = {
    "uncoordinated": "Uncoordinated", "tou": "TOU timer", "price_aware": "Price-aware heuristic",
    "lp_opf": "MPC LP-OPF (perfect foresight)", "flat_ddpg": "Flat DDPG", "ppo_lag": "PPO-Lagrangian",
    "cpo": "CPO", "hrl": "Hierarchical RL", "nested": "Nested (proposed)", "plan": "Plan only (no learning)", "noev": "No EV charging (reference)",
}
# Decomposition of the nested controller (what the learned levels add); Level 3 on.
DECOMP = {"nested-flatprice": "Learned dispatch, flat price", "nested-planprice": "Plan dispatch, learned prices"}


SHORT = {
    "uncoordinated": "Uncoord.", "tou": "TOU", "price_aware": "Price-aware", "lp_opf": "LP-OPF$^\\ast$",
    "flat_ddpg": "Flat DDPG", "ppo_lag": "PPO-Lag.", "cpo": "CPO", "hrl": "HRL", "nested": "Nested",
    "plan": "Plan", "noev": "No EVs", "nested-flatprice": "Dispatch only", "nested-planprice": "Prices only",
}


def short_label(method: str) -> str:
    """Compact label for column-width tables (LP-OPF$^\\ast$: perfect foresight)."""
    base, l3 = split_name(method)
    if method in DECOMP:
        return SHORT[method]
    lab = SHORT[base]
    if base != "nested" and l3:
        lab += "+L3"
    if method == "nested-noL3":
        lab += " w/o L3"
    return lab


def split_name(method: str) -> tuple[str, bool]:
    if method in DECOMP:
        return method, True
    if method.endswith("+L3"):
        return method[:-3], True
    if method == "nested-noL3":
        return "nested", False
    return method, method == "nested"


def label(method: str) -> str:
    if method in DECOMP:
        return DECOMP[method]
    base, l3 = split_name(method)
    lab = LABELS[base]
    if base != "nested" and l3:
        lab += " + L3"
    if method == "nested-noL3":
        lab += " without L3"
    return lab


def price_offset(checkpoint: pathlib.Path | None) -> float:
    """Tariff calibration written next to a learned policy by scripts/calibrate.py."""
    import json
    f = checkpoint.parent / "price_calibration.json" if checkpoint is not None else None
    return float(json.loads(f.read_text())["price_offset"]) if f is not None and f.exists() else 0.0


def build(method: str, cfg: dict, n_agg: int, checkpoint: pathlib.Path | None = None,
          calibrated: bool = True):
    """Returns (policy, q_control, cfg_for_env). Learned policies carry their
    tariff calibration (price offset) unless calibrated=False."""
    base, l3 = split_name(method)
    if base in RULES:
        return RULES[base](), l3, cfg
    if base in DECOMP:
        ctl = load_nested(checkpoint, cfg)
        ctl.price_offset = price_offset(checkpoint) if calibrated else 0.0
        wrap = FlatPrice if base == "nested-flatprice" else PlanDispatch
        return wrap(ctl), True, ablated_cfg(cfg, ctl.ablation)
    if base == "nested":
        ctl = load_nested(checkpoint, cfg)
        ctl.price_offset = price_offset(checkpoint) if calibrated else 0.0
        return ctl, l3, ablated_cfg(cfg, ctl.ablation)
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    if base == "flat_ddpg":
        ctl = FlatDDPG(cfg, n_agg, 0)
    elif base == "hrl":
        ctl = HRL(cfg, n_agg, 0)
    elif base in ("ppo_lag", "cpo"):
        ctl = SafeRL(cfg, n_agg, base)
    else:
        raise ValueError(method)
    ctl.load_state_dict(sd)
    ctl.explore = False
    ctl.price_offset = price_offset(checkpoint) if calibrated else 0.0
    return ctl, l3, cfg


class FlatPrice:
    """Nested controller with its learned dispatch but the flat reference tariff."""
    name = "nested-flatprice"

    def __init__(self, inner):
        self.inner = inner

    def reset(self, env) -> None:
        self.inner.reset(env)
        env.price_offset = 0.0

    def act(self, env) -> None:
        self.inner.act(env)
        env.set_corridor(env.flat_price, env.flat_price + env.min_width)
        env.actions = {k: (a[0], a[1], 0.0) for k, a in env.actions.items()}


class PlanDispatch:
    """Nested controller with its learned (calibrated) prices but plan dispatch u0_k."""
    name = "nested-planprice"

    def __init__(self, inner):
        self.inner = inner

    def reset(self, env) -> None:
        self.inner.reset(env)

    def act(self, env) -> None:
        self.inner.act(env)
        env.actions = {k: ("agg", env.prior_u(k), a[2]) for k, a in env.actions.items()}
