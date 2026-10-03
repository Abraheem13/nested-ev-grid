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
from ..baselines.simple import PriceAware, TOUTimer, Uncoordinated
from ..training.trainer import ablated_cfg, load_nested

RULES = {"uncoordinated": Uncoordinated, "tou": TOUTimer, "price_aware": PriceAware, "lp_opf": LPOPF}
LEARNED_BASELINES = {"flat_ddpg", "ppo_lag", "cpo", "hrl"}
LABELS = {
    "uncoordinated": "Uncoordinated", "tou": "TOU timer", "price_aware": "Price-aware heuristic",
    "lp_opf": "MPC LP-OPF (perfect foresight)", "flat_ddpg": "Flat DDPG", "ppo_lag": "PPO-Lagrangian",
    "cpo": "CPO", "hrl": "Hierarchical RL", "nested": "Nested (proposed)",
}


SHORT = {
    "uncoordinated": "Uncoord.", "tou": "TOU", "price_aware": "Price-aware", "lp_opf": "LP-OPF$^\\ast$",
    "flat_ddpg": "Flat DDPG", "ppo_lag": "PPO-Lag.", "cpo": "CPO", "hrl": "HRL", "nested": "Nested",
}


def short_label(method: str) -> str:
    """Compact label for column-width tables (LP-OPF$^\\ast$: perfect foresight)."""
    base, l3 = split_name(method)
    lab = SHORT[base]
    if base != "nested" and l3:
        lab += "+L3"
    if method == "nested-noL3":
        lab += " w/o L3"
    return lab


def split_name(method: str) -> tuple[str, bool]:
    if method.endswith("+L3"):
        return method[:-3], True
    if method == "nested-noL3":
        return "nested", False
    return method, method == "nested"


def label(method: str) -> str:
    base, l3 = split_name(method)
    lab = LABELS[base]
    if base != "nested" and l3:
        lab += " + L3"
    if method == "nested-noL3":
        lab += " without L3"
    return lab


def build(method: str, cfg: dict, n_agg: int, checkpoint: pathlib.Path | None = None):
    """Returns (policy, q_control, cfg_for_env)."""
    base, l3 = split_name(method)
    if base in RULES:
        return RULES[base](), l3, cfg
    if base == "nested":
        ctl = load_nested(checkpoint, cfg)
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
    return ctl, l3, cfg
