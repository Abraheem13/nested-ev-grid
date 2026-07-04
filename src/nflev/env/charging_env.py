"""Multi-timescale EV charging coordination environment (v2).

Timescale structure:
  Level 1 (3600 s): DSO price corridor [p_min, p_max]  (+ R3.4 network features)
  Level 2 (900 s):  per-aggregator charging dispatch + execution price
                    selection within the corridor (R1.3 fix)
  Level 3a (900 s): Fishbein acceptance at dispatch boundaries (R1.4 fix)
  Level 3b (<= resolution): reactive correction loop inside each QSTS step

Simulation: quasi-static time series (QSTS) at `resolution_s` (default 60 s)
with within-step corrective convergence; a 1-s mode exists for the
high-resolution validation experiment.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandapower as pp

from .network import NETWORKS, constraint_features
from .ev_fleet import FleetModel
from .behavior import FishbeinBehavior
from .qcontrol import ReactivePowerController
from ..agents.projection import PROJECTIONS

AGG_BUSES = {"ieee33": [5, 11, 17, 24, 29],   # 0-indexed buses 6,12,18,25,30
             "ieee69": [8, 20, 34, 48, 60]}   # strong->weak spread incl. bus 61 area


@dataclass
class StepLog:
    t_h: float = 0.0
    v_min: float = 1.0
    q_activated: bool = False
    q_injected_kvar: float = 0.0
    curtailed_kw: float = 0.0
    q_exhausted: bool = False
    cost_usd: float = 0.0
    p_total_mw: float = 0.0
    violation: bool = False
    line_margin: float = 1.0
    sub_frac: float = 0.0


class ChargingEnv:
    def __init__(self, cfg: dict, network: str, dataset_cfg: dict,
                 n_ev: int, seed: int = 0, scenario_mods: dict | None = None):
        self.cfg = cfg
        self.network_name = network
        self.rng = np.random.default_rng(seed)
        self.mods = scenario_mods or {}

        sim = cfg["simulation"]
        self.dt_s = int(self.mods.get("resolution_s", sim["resolution_s"]))
        self.dispatch_s = sim["dispatch_interval_s"]
        self.pricing_s = sim["pricing_interval_s"]
        self.episode_h = self.mods.get("episode_hours", sim["episode_hours"])
        self.load_noise = sim["load_noise_sigma"]
        # Noon-to-noon horizon: evening arrivals need the overnight valley
        # inside the optimization window; a 00:00-24:00 episode strands them.
        self.wall_offset_h = self.mods.get("wall_offset_h", 12.0)

        self.agg_buses = AGG_BUSES[network]
        self.n_agg = len(self.agg_buses)
        self.fleet = FleetModel(cfg, dataset_cfg, n_ev, self.n_agg, self.rng)
        self.behavior = FishbeinBehavior(cfg["level3"]["behavior"]["lambda_ref"], self.rng)

        v = cfg["voltage"]
        rp = cfg["reactive_power"]
        derate = self.mods.get("s_rated_derate", 1.0)
        self.s_rated = rp["s_rated_kva"] * derate
        self.eff = rp["charger_efficiency"]
        self.qctl = ReactivePowerController(
            v["v_min"], v["v_critical"], v["correction_margin"],
            v["max_correction_iters"], rp["curtailment_fallback"],
            rp["curtailment_step"])
        self.v_min_limit = v["v_min"]
        self.project = PROJECTIONS[cfg["level2"]["action_projection"]]
        self.p_cap = cfg["level2"]["transformer_cap_kw"]

        self._net_template = NETWORKS[network]()
        self.base_load_p = self._net_template.load.p_mw.values.copy()
        self.base_load_q = self._net_template.load.q_mvar.values.copy()

    # ---------------------------------------------------------------- reset
    def reset(self, price_profile: np.ndarray, load_profile: np.ndarray):
        """price_profile: hourly wholesale LMP [$/MWh], len == episode_h.
           load_profile: hourly multiplier on base load, len == episode_h."""
        self.net = NETWORKS[self.network_name]()
        self.price_profile = price_profile
        self.load_profile = load_profile
        self.t_s = 0
        self.evs = self.fleet.reset()
        for ev in self.evs:  # wall clock -> episode time
            dur = ev.departure_h - ev.arrival_h
            ev.arrival_h = (ev.arrival_h - self.wall_offset_h) % 24
            ev.departure_h = ev.arrival_h + dur
        self.logs: list[StepLog] = []
        # per-aggregator EV load elements + sgen for Q injection
        self.ev_load_idx, self.ev_sgen_idx = {}, {}
        for k, b in enumerate(self.agg_buses):
            self.ev_load_idx[k] = pp.create_load(self.net, b, p_mw=0.0, q_mvar=0.0,
                                                 name=f"agg{k}_ev")
            self.ev_sgen_idx[k] = pp.create_sgen(self.net, b, p_mw=0.0, q_mvar=0.0,
                                                 name=f"agg{k}_v2g_q")
        self.corridor = np.array([0.08, 0.20])
        self.exec_prices = np.full(self.n_agg, 0.12)
        self.agg_rates: dict[int, dict[int, float]] = {k: {} for k in range(self.n_agg)}
        self._episode_cost = 0.0
        return self._l1_state()

    # --------------------------------------------------------------- states
    def _solved_pf(self):
        try:
            pp.runpp(self.net, init="results", numba=True)
        except Exception:
            pp.runpp(self.net, init="auto", numba=True)

    def _l1_state(self) -> np.ndarray:
        self._apply_loads(self.t_s / 3600.0)
        self._solved_pf()
        lm, sf = constraint_features(self.net)
        vm = self.net.res_bus.vm_pu
        n_conn = sum(e.connected for e in self.evs)
        socs = [e.soc for e in self.evs if e.connected]
        return np.array([
            vm.mean(), self.net.res_load.p_mw.sum() * 1000.0 / 5000.0,
            self.net.res_load.q_mvar.sum() * 1000.0 / 3000.0,
            self._lmp(self.t_s / 3600.0) / 100.0,
            n_conn / max(1, self.fleet.n_ev),
            float(np.mean(socs)) if socs else 0.0,
            lm, sf,  # R3.4 features
        ], dtype=np.float32)

    def l2_state(self, k: int, max_evs: int = 40) -> np.ndarray:
        evs = self.fleet.connected_by_aggregator(k)[:max_evs]
        b = self.agg_buses[k]
        head = [self.net.res_bus.vm_pu.at[b],
                float(self.net.load.at[self.ev_load_idx[k], "p_mw"]) * 1000 / self.p_cap,
                float(self.net.sgen.at[self.ev_sgen_idx[k], "q_mvar"]) * 1000 / 250.0,
                self.exec_prices[k] / 0.3,
                self.corridor[0] / 0.3, self.corridor[1] / 0.3]
        evf = np.zeros(max_evs * 3, dtype=np.float32)
        t_h = self.t_s / 3600.0
        for j, e in enumerate(evs):
            evf[j*3:(j+1)*3] = [e.soc, e.soc_target,
                                min(1.0, (e.departure_h - t_h) / 12.0)]
        return np.concatenate([np.array(head, dtype=np.float32), evf])

    def _lmp(self, t_h: float) -> float:
        wall = (t_h + self.wall_offset_h) % 24 + 24 * (t_h // 24)
        return float(self.price_profile[min(int(wall), len(self.price_profile) - 1)])

    # --------------------------------------------------------------- actions
    def set_corridor(self, p_min: float, p_max: float):
        l1 = self.cfg["level1"]
        p_min = float(np.clip(p_min, l1["price_floor"], l1["price_ceil"]))
        p_max = float(np.clip(p_max, p_min + l1["min_corridor_width"], l1["price_ceil"]))
        self.corridor = np.array([p_min, p_max])

    def set_dispatch(self, k: int, rates_raw: np.ndarray, price_frac: float):
        """R1.3: price_frac in [0,1] selects execution price inside corridor.
           R1.5: rates projected onto feasible set by construction."""
        self.exec_prices[k] = self.corridor[0] + price_frac * (self.corridor[1] - self.corridor[0])
        evs = self.fleet.connected_by_aggregator(k)
        n = len(evs)
        if n == 0:
            self.agg_rates[k] = {}
            return
        c_max = np.array([e.p_max_kw for e in evs])
        c = self.project(rates_raw[:n], c_max, self.p_cap)
        # zero-charging forcing window (deadlock experiment S6)
        w = self.mods.get("force_zero_charging_window")  # wall-clock hours
        wall_h = (self.t_s / 3600.0 + self.wall_offset_h) % 24
        if w and w[0] <= wall_h < w[1]:
            c = np.zeros_like(c)
        self.agg_rates[k] = {e.idx: float(c[i]) * float(e.accepted) for i, e in enumerate(evs)}

    # ------------------------------------------------------------- stepping
    def run_dispatch_interval(self) -> dict:
        """Advance one 15-min interval at QSTS resolution. Returns interval metrics."""
        t0_h = self.t_s / 3600.0
        self.fleet.step_connections(t0_h)
        # L3a: behavioral response at the dispatch boundary
        mean_price = float(np.mean(self.exec_prices))
        accept_rate = self.behavior.evaluate(self.evs, mean_price, t_h=t0_h)

        steps = self.dispatch_s // self.dt_s
        interval_cost = 0.0
        v_mins, q_acts, curts, exhausted = [], 0, 0.0, False
        for s in range(steps):
            t_h = (self.t_s + s * self.dt_s) / 3600.0
            self.fleet.step_connections(t_h)
            self._apply_loads(t_h)
            self._solved_pf()
            charger_map = self._charger_map()
            if self.mods.get("disable_q_control"):
                from .qcontrol import QControlResult
                v = float(self.net.res_bus.vm_pu.min())
                res = QControlResult(v_min_pre=v, v_min_post=v)
            else:
                res = self.qctl.correct(self.net, charger_map)
            log = StepLog(
                t_h=t_h, v_min=res.v_min_post, q_activated=res.activated,
                q_injected_kvar=sum(res.q_injected_kvar.values()),
                curtailed_kw=sum(res.curtailed_kw.values()),
                q_exhausted=res.q_exhausted,
                p_total_mw=float(self.net.res_load.p_mw.sum()),
                violation=res.v_min_post < self.v_min_limit - 1e-9,
            )
            log.line_margin, log.sub_frac = constraint_features(self.net)
            dt_h = self.dt_s / 3600.0
            ev_p_kw = self._total_ev_kw(after_curtail=res)
            log.cost_usd = self._lmp(t_h) / 1000.0 * ev_p_kw * dt_h
            interval_cost += log.cost_usd
            self.logs.append(log)
            v_mins.append(res.v_min_post)
            q_acts += res.activated
            curts += log.curtailed_kw
            exhausted |= res.q_exhausted
            # charging progress (post-curtailment rates)
            eff_rates = self._effective_rates(res)
            self.fleet.apply_charging(eff_rates, dt_h, self.eff)
            self._reset_injections()
        self.t_s += self.dispatch_s
        self._episode_cost += interval_cost
        return {"cost": interval_cost, "v_min": min(v_mins), "q_activations": q_acts,
                "curtailed_kwh": curts * (self.dt_s / 3600.0),
                "q_exhausted": exhausted, "accept_rate": accept_rate}

    # -------------------------------------------------------------- helpers
    def _apply_loads(self, t_h: float):
        wall = (t_h + self.wall_offset_h) % 24 + 24 * (t_h // 24)
        mult = self.load_profile[min(int(wall), len(self.load_profile) - 1)]
        noise = self.rng.normal(1.0, self.load_noise)
        fnoise = self.mods.get("load_forecast_noise", 0.0)
        if fnoise:
            mult *= self.rng.normal(1.0, fnoise / 3)
        n_base = len(self.base_load_p)
        self.net.load.iloc[:n_base, self.net.load.columns.get_loc("p_mw")] = \
            self.base_load_p * mult * noise
        self.net.load.iloc[:n_base, self.net.load.columns.get_loc("q_mvar")] = \
            self.base_load_q * mult * noise
        for k in range(self.n_agg):
            p_kw = sum(self.agg_rates[k].values())
            self.net.load.at[self.ev_load_idx[k], "p_mw"] = p_kw / 1000.0

    def _charger_map(self) -> dict:
        m = {}
        for k, b in enumerate(self.agg_buses):
            evs = self.fleet.connected_by_aggregator(k)
            if not evs:
                continue
            p = np.array([self.agg_rates[k].get(e.idx, 0.0) for e in evs])
            m[b] = {"sgen_idx": self.ev_sgen_idx[k],
                    "s_rated_kva": np.full(len(evs), self.s_rated),
                    "p_kw": p, "load_idx": self.ev_load_idx[k]}
        return m

    def _effective_rates(self, res) -> dict[int, float]:
        rates = {}
        for k, b in enumerate(self.agg_buses):
            shed_frac = 0.0
            if b in res.curtailed_kw:
                tot = sum(self.agg_rates[k].values())
                shed_frac = min(1.0, res.curtailed_kw[b] / tot) if tot > 0 else 0.0
            for idx, r in self.agg_rates[k].items():
                rates[idx] = r * (1.0 - shed_frac)
        return rates

    def _total_ev_kw(self, after_curtail) -> float:
        return sum(self._effective_rates(after_curtail).values())

    def _reset_injections(self):
        for k in range(self.n_agg):
            self.net.sgen.at[self.ev_sgen_idx[k], "q_mvar"] = 0.0

    # -------------------------------------------------------------- metrics
    def episode_metrics(self) -> dict:
        v = np.array([l.v_min for l in self.logs])
        return {
            "daily_cost_usd": self._episode_cost,
            "min_voltage_pu": float(v.min()),
            "violation_rate_pct": 100.0 * float(np.mean([l.violation for l in self.logs])),
            "service_quality": self.fleet.service_quality(),
            "q_activation_freq": float(np.mean([l.q_activated for l in self.logs])),
            "curtailed_kwh": sum(l.curtailed_kw for l in self.logs) * self.dt_s / 3600.0,
            "q_exhausted_any": any(l.q_exhausted for l in self.logs),
            "peak_mw": float(max(l.p_total_mw for l in self.logs)),
        }