"""Multi-timescale EV charging environment on a radial feeder (v3).

Timescales
  Level 1  (1 h)     DSO sets the retail price corridor [p_min, p_max].
  Level 2  (15 min)  each aggregator sets its execution price and its charging
                     dispatch (aggregate set point or per-vehicle rates).
  Level 3a (15 min)  vehicles accept or decline the execution price.
  Level 3b (60 s)    AC power flow and reactive correction every QSTS step.

Energy accounting: a vehicle draws power only while it is connected, has
accepted the current price, and still needs energy; within a step it never
draws more than completes its need. Wholesale cost is charged on the energy
actually drawn from the grid (after any curtailment).
"""
from __future__ import annotations

import numpy as np

from ..grid.network import load_network
from ..grid.powerflow import RadialPowerFlow
from .allocation import laxity_h, llf_allocate, project
from .behavior import FishbeinBehavior
from .calibration import base_load_scale
from .episode import EpisodeSpec
from .qcontrol import ReactiveController

L2_OBS_DIM = 28
L1_OBS_DIM = 11


class ChargingEnv:
    def __init__(self, cfg: dict, network: str = "ieee33", q_control: bool = True):
        self.cfg = cfg
        self.network_name = network
        self.net = load_network(network)
        self.pf = RadialPowerFlow(self.net, v0=cfg["network"]["substation_vm_pu"])
        self.agg_bus = np.asarray(cfg["aggregators"]["buses"][network], int) - 1
        self.n_agg = len(self.agg_bus)
        self.scale = cfg["network"]["base_load_scale"] or base_load_scale(cfg, network)
        self.q_enabled = q_control
        sim, rp = cfg["simulation"], cfg["reactive_power"]
        self.dt_s = sim["resolution_s"]
        self.dt_h = self.dt_s / 3600.0
        self.interval_h = sim["dispatch_interval_s"] / 3600.0
        self.steps_per_interval = sim["dispatch_interval_s"] // self.dt_s
        self.look = sim["price_lookahead_h"]
        self.p_max = rp["charger_p_max_kw"]
        self.eff = rp["charger_efficiency"]
        self.p_cap = cfg["aggregators"]["transformer_cap_kw"]
        self.v_min = cfg["voltage"]["v_min"]
        r = cfg["retail"]
        self.price_floor, self.price_ceil = r["price_floor"], r["price_ceil"]
        self.min_width, self.flat_price = r["min_corridor_width"], r["flat_price"]
        self.guard = cfg["level2"]["deadline_guard"]
        self.disagg = cfg["level2"].get("disaggregation", "llf")
        self.prior_mode = cfg["level2"].get("prior", "none")

    # ================================================================ reset
    def reset(self, spec: EpisodeSpec) -> None:
        self.spec, self.mods = spec, dict(spec.mods)
        evs = spec.evs
        self.n_ev = len(evs)
        self.arr = np.array([e.arrival_h for e in evs])
        self.dep = np.array([e.departure_h for e in evs])
        self.need0 = np.array([e.need_kwh for e in evs])
        self.bat = np.array([e.battery_kwh for e in evs])
        self.soc0 = np.array([e.soc_init for e in evs])
        self.agg = np.asarray(spec.ev_agg, int)
        self.delivered = np.zeros(self.n_ev)
        self.connected = np.zeros(self.n_ev, bool)
        self.departed = np.zeros(self.n_ev, bool)
        self.accepted = np.zeros(self.n_ev, bool)
        self.alloc = np.zeros(self.n_ev)
        self.behavior = FishbeinBehavior(self.cfg, self.n_ev, spec.behavior_seed)
        self.qctl = ReactiveController(self.cfg, self.cfg["reactive_power"]["s_rated_kva"]
                                       * float(self.mods.get("s_rated_derate", 1.0)))
        self.t_step = 0
        self.n_steps = spec.horizon_h * 3600 // self.dt_s
        self.corridor = np.array([self.flat_price, self.flat_price + self.min_width])
        self.exec_price = np.full(self.n_agg, self.flat_price)
        self.actions: dict[int, tuple] = {}
        self.applied_u = np.zeros(self.n_agg)
        self.accept_rate = 0.7
        self.m = dict(cost=0.0, drawn_kwh=0.0, curtailed_kwh=0.0, viol_steps=0, q_steps=0,
                      vmin=np.inf, peak_ev_kw=0.0, peak_feeder_mw=0.0, exhausted=False,
                      unmet_kwh=0.0, q_kvarh=0.0, pf_solves=0, revenue=0.0)
        self.p_nom = self.net.p_load_mw * self.scale
        self.q_nom = self.net.q_load_mvar * self.scale
        self._unmet_acc = np.zeros(self.n_agg)
        self._prior_cache = (-1, None)
        self._update_connections(0.0)
        self.last_res = self.pf.solve(self.p_nom * spec.load_actual[0], self.q_nom * spec.load_actual[0],
                                      warm=False)

    # ============================================================== helpers
    @property
    def t_h(self) -> float:
        return self.t_step * self.dt_h

    @property
    def done(self) -> bool:
        return self.t_step >= self.n_steps

    def wall_hour(self, t_h: float | None = None) -> float:
        return (self.spec.start_hour + (self.t_h if t_h is None else t_h)) % 24.0

    def lmp(self, t_h: float | None = None) -> float:
        t = self.t_h if t_h is None else t_h
        return float(self.spec.prices[min(int(t), len(self.spec.prices) - 1)])

    def need(self) -> np.ndarray:
        return np.maximum(self.need0 - self.delivered, 0.0)

    def hours_left(self) -> np.ndarray:
        return self.dep - self.t_h

    def soc(self) -> np.ndarray:
        return self.soc0 + self.delivered / self.bat

    def _update_connections(self, t_h: float) -> None:
        """Plug vehicles in/out; unmet energy of departing vehicles is booked
        to the aggregator's running `_unmet_acc` and to the episode total."""
        arriving = (~self.connected) & (~self.departed) & (self.arr <= t_h + 1e-9)
        self.connected |= arriving
        leaving = np.flatnonzero(self.connected & (self.dep <= t_h + 1e-9))
        if len(leaving):
            um = self.need()[leaving]
            np.add.at(self._unmet_acc, self.agg[leaving], um)
            self.m["unmet_kwh"] += float(um.sum())
            self.connected[leaving] = False
            self.departed[leaving] = True
            self.alloc[leaving] = 0.0

    # ============================================================== actions
    def set_corridor(self, p_min: float, p_max: float) -> None:
        p_min = float(np.clip(p_min, self.price_floor, self.price_ceil - self.min_width))
        p_max = float(np.clip(p_max, p_min + self.min_width, self.price_ceil))
        self.corridor = np.array([p_min, p_max])

    def set_aggregate(self, k: int, u: float, price_frac: float) -> None:
        self.actions[k] = ("agg", float(u), float(np.clip(price_frac, 0, 1)))

    def set_rates(self, k: int, rates: dict, price_frac: float = 0.0) -> None:
        """rates: vehicle index -> requested kW (missing vehicles get 0)."""
        self.actions[k] = ("rate", rates, float(np.clip(price_frac, 0, 1)))

    # ============================================================= stepping
    def run_interval(self) -> dict:
        t0 = self.t_h
        price_now, price_ahead = self.lmp(t0), float(self._lmp_ahead(self.look).mean())
        self._update_connections(t0)
        lo, hi = self.corridor
        for k in range(self.n_agg):
            self.exec_price[k] = lo + self.actions.get(k, ("rate", {}, 0.0))[2] * (hi - lo)
        need = self.need()
        live = np.flatnonzero(self.connected & (need > 1e-6))
        self.accepted[:] = False
        if self.mods.get("disable_behavior"):
            self.accepted[live] = True
        elif len(live):
            self.accepted[live] = self.behavior.evaluate(
                live, self.exec_price[self.agg[live]], need[live],
                self.dep[live] - t0, self.p_max)
        self.accept_rate = float(self.accepted[live].mean()) if len(live) else self.accept_rate
        self.alloc[:] = 0.0
        forced = self.mods.get("force_zero_charging_window")
        zero_now = bool(forced) and forced[0] <= self.wall_hour(t0) < forced[1]
        for k in range(self.n_agg):
            mem = live[(self.agg[live] == k) & self.accepted[live]]
            if len(mem) == 0:
                self.applied_u[k] = 0.0
                continue
            cap = np.minimum(self.p_max, need[mem] / (self.eff * self.interval_h))
            kind, a, _ = self.actions.get(k, ("rate", {}, 0.0))
            if kind == "agg":
                lax = laxity_h(self.dep[mem] - t0, need[mem], self.p_max, self.eff)
                rates, self.applied_u[k] = llf_allocate(a, cap, lax, self.p_cap, self.interval_h,
                                                        self.guard, self.disagg)
            else:
                req = np.array([a.get(int(i), 0.0) for i in mem])
                rates = project(req, cap, self.p_cap)
                avail = min(self.p_cap, cap.sum())
                self.applied_u[k] = rates.sum() / avail if avail > 0 else 0.0
            if zero_now:
                rates = np.zeros_like(rates)
            self.alloc[mem] = rates
        self.actions = {}

        per = {key: np.zeros(self.n_agg) for key in
               ("drawn_kwh", "delivered_kwh", "cost", "curtailed_kwh", "unmet_kwh")}
        info = dict(vmin=np.inf, viol_steps=0, q_steps=0, cost=0.0, curtailed_kwh=0.0)
        load = self.spec.load_actual
        for _ in range(self.steps_per_interval):
            t = self.t_h
            self._update_connections(t)
            need = self.need()
            ok = self.connected & self.accepted & (self.alloc > 0)
            drawn = np.where(ok, np.minimum(self.alloc, need / (self.eff * self.dt_h)), 0.0)
            ev_kw = np.bincount(self.agg, weights=drawn, minlength=self.n_agg)
            mult = load[min(self.t_step, len(load) - 1)]
            p_bus = self.p_nom * mult
            q_bus = self.q_nom * mult
            np.add.at(p_bus, self.agg_bus, ev_kw / 1000.0)
            res = self.pf.solve(p_bus, q_bus)
            self.m["pf_solves"] += 1
            shed = np.zeros(self.n_agg)
            if self.q_enabled:
                conn_by_k = [drawn[(self.agg == k) & self.connected] for k in range(self.n_agg)]
                qr = self.qctl.correct(self.pf, p_bus, q_bus, self.agg_bus, conn_by_k, res)
                res, shed = qr.res, qr.shed_frac
                self.m["pf_solves"] += qr.iterations
                if qr.activated:
                    info["q_steps"] += 1
                    self.m["q_steps"] += 1
                    self.m["q_kvarh"] += float(qr.q_kvar.sum()) * self.dt_h
                self.m["exhausted"] |= qr.exhausted
            eff_rate = drawn * (1.0 - shed[self.agg])
            self.delivered += eff_rate * self.eff * self.dt_h
            price = self.lmp(t) / 1000.0                       # EUR/kWh
            e_k = np.bincount(self.agg, weights=eff_rate * self.dt_h, minlength=self.n_agg)
            c_k = np.bincount(self.agg, weights=(drawn - eff_rate) * self.dt_h, minlength=self.n_agg)
            per["drawn_kwh"] += e_k
            per["delivered_kwh"] += e_k * self.eff
            per["cost"] += e_k * price
            self.m["revenue"] += float((e_k * self.eff * self.exec_price).sum())
            per["curtailed_kwh"] += c_k
            self.m["cost"] += float(e_k.sum() * price)
            self.m["drawn_kwh"] += float(e_k.sum())
            self.m["curtailed_kwh"] += float(c_k.sum())
            viol = res.vmin < self.v_min - 1e-9
            self.m["viol_steps"] += int(viol)
            info["viol_steps"] += int(viol)
            info["vmin"] = min(info["vmin"], res.vmin)
            self.m["vmin"] = min(self.m["vmin"], res.vmin)
            self.m["peak_ev_kw"] = max(self.m["peak_ev_kw"], float(eff_rate.sum()))
            self.m["peak_feeder_mw"] = max(self.m["peak_feeder_mw"], res.s_sub_mva.real)
            self.last_res = res
            self.t_step += 1
        if self.done:                                   # vehicles still plugged in at the end
            self._update_connections(np.inf)
        per["unmet_kwh"] = self._unmet_acc.copy()
        self._unmet_acc[:] = 0.0
        info.update(per=per, cost=float(per["cost"].sum()),
                    curtailed_kwh=float(per["curtailed_kwh"].sum()),
                    applied_u=self.applied_u.copy(), exec_price=self.exec_price.copy(),
                    price_now=price_now, price_ahead_mean=price_ahead)
        return info

    # ====================================================== planning prior
    def price_plan(self, k: int) -> np.ndarray:
        """Deadline-aware cheapest-slot plan: the plugged-in vehicles of
        aggregator k that still need energy and for which the current dispatch
        interval is among the ceil(need / slot energy) cheapest intervals before
        departure. Prices are known `price_lookahead_h` ahead; later intervals
        are valued at the mean known price (naive forecast)."""
        tau = int(round(self.t_h / self.interval_h))
        n_tau = int(round(self.spec.horizon_h / self.interval_h))
        known = tau + int(round(self.look / self.interval_h))
        slot_kwh = self.p_max * self.eff * self.interval_h
        lmps_all = np.array([self.lmp(s * self.interval_h) for s in range(tau, n_tau)])
        unknown = np.arange(tau, n_tau) >= known
        if unknown.any():
            lmps_all[unknown] = lmps_all[~unknown].mean()
        need = self.need()
        chosen = []
        for i in np.flatnonzero(self.connected & (self.agg == k)):
            k_need = int(np.ceil(need[i] / slot_kwh - 1e-9))
            if k_need <= 0:
                continue
            last = min(n_tau, int(np.floor(self.dep[i] / self.interval_h + 1e-9)))
            lmps = lmps_all[: max(1, last - tau)]
            if 0 in np.argsort(lmps, kind="stable")[:k_need]:          # slot 0 = now
                chosen.append(int(i))
        return np.asarray(chosen, int)

    def prior_u(self, k: int) -> float:
        """Aggregate set point (fraction of available power, as in
        `llf_allocate`) that the cheapest-slot plan dispatches now."""
        if self._prior_cache[0] != self.t_step:
            need = self.need()
            cap = np.minimum(self.p_max, need / (self.eff * self.interval_h))
            live = self.connected & (need > 1e-6)
            u = np.zeros(self.n_agg)
            for j in range(self.n_agg):
                avail = min(self.p_cap, float(cap[live & (self.agg == j)].sum()))
                plan = self.price_plan(j)
                u[j] = min(1.0, float(cap[plan].sum()) / avail) if avail > 0 and len(plan) else 0.0
            self._prior_cache = (self.t_step, u)
        return float(self._prior_cache[1][k])

    # ========================================================= observations
    def _lmp_ahead(self, hours: int) -> np.ndarray:
        i = int(self.t_h)
        p = self.spec.prices
        return np.array([p[min(i + j, len(p) - 1)] for j in range(hours)])

    def l1_obs(self) -> np.ndarray:
        res = self.last_res
        conn = self.connected
        h = self.wall_hour()
        return np.array([
            res.vm[1:].mean(), res.s_sub_mva.real / 5.0, res.s_sub_mva.imag / 3.0,
            self.lmp() / 200.0, conn.sum() / max(1, self.n_ev),
            float(self.soc()[conn].mean()) if conn.any() else 0.0,
            self.pf.line_margin(res), self.pf.substation_loading(res),
            np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
            self._lmp_ahead(self.look).mean() / 200.0,
        ], dtype=np.float32)

    def l2_obs(self, k: int) -> np.ndarray:
        res, t = self.last_res, self.t_h
        mem = np.flatnonzero(self.connected & (self.agg == k))
        need = self.need()[mem]
        left = self.dep[mem] - t
        lax = laxity_h(left, need, self.p_max, self.eff)
        bins = [(-np.inf, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, np.inf)]
        scale = self.p_cap * 1.0
        lax_feat = [need[(lax >= a) & (lax < b)].sum() / scale for a, b in bins]
        cap = np.minimum(self.p_max, need / (self.eff * self.interval_h))
        h = self.wall_hour()
        obs = [res.vm[self.agg_bus[k]], np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
               self.lmp() / 200.0, *(self._lmp_ahead(self.look) / 200.0),
               self.corridor[0] / self.price_ceil, self.corridor[1] / self.price_ceil,
               len(mem) / 100.0, need.sum() / scale, min(self.p_cap, cap.sum()) / self.p_cap,
               *lax_feat, (left.mean() / 12.0) if len(mem) else 0.0, self.accept_rate,
               self.prior_u(k) if self.prior_mode != "none" else 0.0]
        return np.asarray(obs, dtype=np.float32)

    # ============================================================== metrics
    def episode_metrics(self) -> dict:
        m = self.m
        req = float(self.need0.sum())
        deliv = float(np.minimum(self.delivered, self.need0).sum())
        return {
            "cost_eur": m["cost"],
            "energy_drawn_kwh": m["drawn_kwh"],
            "energy_delivered_kwh": deliv,
            "energy_requested_kwh": req,
            "service_quality": deliv / req if req > 0 else 1.0,
            "unmet_kwh": m["unmet_kwh"],
            "cost_per_kwh": m["cost"] / deliv if deliv > 0 else float("nan"),
            "retail_price_paid": m["revenue"] / deliv if deliv > 0 else float("nan"),
            "min_voltage_pu": m["vmin"],
            "violation_rate_pct": 100.0 * m["viol_steps"] / self.n_steps,
            "q_activation_pct": 100.0 * m["q_steps"] / self.n_steps,
            "q_kvarh": m["q_kvarh"],
            "curtailed_kwh": m["curtailed_kwh"],
            "q_exhausted_any": bool(m["exhausted"]),
            "peak_ev_kw": m["peak_ev_kw"],
            "peak_feeder_mw": m["peak_feeder_mw"],
            "pf_solves": m["pf_solves"],
            "day": str(self.spec.day.date()),
        }
