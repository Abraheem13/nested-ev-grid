"""Regression tests for the v3 environment, data layer and training plumbing.
Each test pins down one defect found in the v2 code or one modelling claim made
in the paper."""
import pathlib
import sys

import numpy as np
import pytest
import yaml
from scipy.optimize import minimize

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nflev.baselines.milp import LPOPF                     # noqa: E402
from nflev.baselines.simple import PriceAware, TOUTimer, Uncoordinated  # noqa: E402
from nflev.data.prices import load_prices                   # noqa: E402
from nflev.data.sources import FILES, ensure_data, _sha256  # noqa: E402
from nflev.env.allocation import llf_allocate, project      # noqa: E402
from nflev.env.charging_env import ChargingEnv               # noqa: E402
from nflev.env.episode import make_episode                   # noqa: E402
from nflev.env.qcontrol import ReactiveController            # noqa: E402
from nflev.eval.runner import run_policy_episode             # noqa: E402

CFG = yaml.safe_load(open(ROOT / "configs" / "base.yaml"))


def spec(n_ev=120, j=0, **scen):
    return make_episode(CFG, "residential", "test", n_ev, seed=7000 + j, scenario=scen, eval_index=j)


# ------------------------------------------------------------------ data
def test_data_checksums():
    root = ensure_data(verbose=False)
    for f, h in FILES.items():
        assert _sha256(root / f) == h


def test_price_series_complete_and_deduplicated():
    s = load_prices()
    assert s.index.is_unique and s.isna().sum() == 0
    counts = s.groupby(s.index.year).size()
    assert counts[2019] == 8760 and counts[2023] == 8760 and counts[2024] == 8784


# ------------------------------------------------------- feasibility layer
def test_projection_is_exact_euclidean():
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = 6
        raw = rng.uniform(-2, 14, n)
        cap = rng.uniform(1, 11, n)
        pc = rng.uniform(5, 40)
        c = project(raw, cap, pc)
        assert c.sum() <= pc + 1e-7 and np.all(c >= -1e-9) and np.all(c <= cap + 1e-9)
        res = minimize(lambda x: ((x - raw) ** 2).sum(), np.zeros(n), bounds=list(zip(np.zeros(n), cap)),
                       constraints=[{"type": "ineq", "fun": lambda x: pc - x.sum()}], method="SLSQP",
                       options={"ftol": 1e-12, "maxiter": 500})
        assert np.allclose(c, res.x, atol=1e-4)


def test_llf_guard_and_caps():
    cap = np.array([11.0, 11.0, 11.0, 5.0])
    lax = np.array([0.1, 5.0, 0.2, 9.0])
    rates, _ = llf_allocate(0.0, cap, lax, p_cap=500, interval_h=0.25, guard=True)
    assert rates[0] == 11.0 and rates[2] == 11.0 and rates[1] == 0.0      # urgent served at u = 0
    rates, _ = llf_allocate(1.0, cap, lax, p_cap=20, interval_h=0.25, guard=False)
    assert abs(rates.sum() - 20) < 1e-9 and rates[0] == 11.0              # least laxity first


# ---------------------------------------------------------- energy model
@pytest.mark.parametrize("policy", [Uncoordinated(), TOUTimer(), PriceAware()])
@pytest.mark.parametrize("q", [False, True])
def test_energy_conservation(policy, q):
    env = ChargingEnv(CFG, "ieee33", q_control=q)
    m = run_policy_episode(env, policy, spec())
    assert m["energy_delivered_kwh"] <= m["energy_requested_kwh"] + 1e-6
    assert abs(m["energy_delivered_kwh"] - env.eff * m["energy_drawn_kwh"]) < 1e-6 * m["energy_drawn_kwh"] + 1e-6
    assert abs(m["energy_requested_kwh"] - m["energy_delivered_kwh"] - m["unmet_kwh"]) < 1e-6 * m["energy_requested_kwh"]


def test_full_vehicles_draw_nothing():
    """v2 defect: chargers kept drawing 11 kW into full batteries."""
    env = ChargingEnv(CFG, "ieee33", q_control=False)
    env.reset(spec(60))
    pol = Uncoordinated()
    pol.reset(env)
    while not env.done:
        pol.act(env)
        env.run_interval()
        assert np.all(env.delivered <= env.need0 + 1e-9)
    m = env.episode_metrics()
    assert m["energy_drawn_kwh"] * env.eff <= m["energy_requested_kwh"] + 1e-6


def test_transformer_cap_respected():
    env = ChargingEnv(CFG, "ieee33", q_control=False)
    env.reset(spec(300))
    pol = Uncoordinated()
    pol.reset(env)
    while not env.done:
        pol.act(env)
        env.run_interval()
        per_agg = np.bincount(env.agg, weights=env.alloc, minlength=env.n_agg)
        assert np.all(per_agg <= env.p_cap + 1e-6)


# -------------------------------------------------------------- episodes
def test_episodes_are_reproducible_and_paired():
    a, b = spec(120, 3), spec(120, 3)
    assert a.day == b.day and np.array_equal(a.load_actual, b.load_actual)
    assert [e.need_kwh for e in a.evs] == [e.need_kwh for e in b.evs]
    assert len({spec(30, j).day for j in range(50)}) == 50   # 50 distinct held-out days


def test_departures_inside_horizon():
    sp = spec(300)
    assert all(e.departure_h <= sp.horizon_h - 0.25 + 1e-9 for e in sp.evs)
    assert all(e.need_kwh <= (e.departure_h - e.arrival_h) * 11.0 * 0.95 + 1e-9 for e in sp.evs)


# -------------------------------------------------------------- Level 3
def test_reactive_capacity_from_inverter_rating():
    q = ReactiveController(CFG, 12.0)
    assert abs(q.capacity(np.array([0.0])) - 12.0) < 1e-12             # full capacity at P = 0
    assert abs(q.capacity(np.array([11.0])) - np.sqrt(144 - 121)) < 1e-12


def test_level3_holds_floor_where_capacity_suffices():
    sp = spec(240, 0)
    off = run_policy_episode(ChargingEnv(CFG, "ieee33", q_control=False), TOUTimer(), sp)
    on = run_policy_episode(ChargingEnv(CFG, "ieee33", q_control=True), TOUTimer(), sp)
    assert off["violation_rate_pct"] > 0 and on["violation_rate_pct"] == 0.0


def test_lp_opf_runs_on_the_episode_fleet():
    """v2 defect: the LP was solved for one random fleet and replayed on another."""
    m = run_policy_episode(ChargingEnv(CFG, "ieee33", q_control=False), LPOPF(), spec(60, 1))
    assert m["service_quality"] > 0.97


# ------------------------------------------------- DDPG reward alignment
def test_ddpg_transitions_are_aligned(monkeypatch, tmp_path):
    """v2 defect: the reward of interval i was stored with the action of i-1."""
    import nflev.training.trainer as T
    stored = []
    orig_obs = ChargingEnv.l2_obs

    def tagged_obs(self, k):
        o = orig_obs(self, k)
        o[0] = self.t_step // self.steps_per_interval          # interval index in slot 0
        return o

    def fake_rewards(cfg, env, info, w=None):                  # reward = index of the interval just run
        return np.full(env.n_agg, float(env.t_step // env.steps_per_interval - 1))

    monkeypatch.setattr(ChargingEnv, "l2_obs", tagged_obs)
    monkeypatch.setattr(T, "aggregator_rewards", fake_rewards)
    orig_store = T.DDPGAgent.store
    monkeypatch.setattr(T.DDPGAgent, "store",
                        lambda self, s, a, r, s2, d: (stored.append((s[0], r, s2[0], d)),
                                                      orig_store(self, s, a, r, s2, d)))
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml"))
    cfg["curriculum"]["stages"] = [{"n_ev": 20}]
    T.train_nested(cfg, "residential", "ieee33", 0, "none", tmp_path, episodes=1, log_every=99)
    assert stored
    for s, r, s2, d in stored:
        assert r == s                        # reward produced by the action taken in state s
        assert s2 == s + 1                   # next state is the following interval


def test_training_resume_is_exact(tmp_path, monkeypatch):
    """A run that dies right after a checkpoint and is restarted produces the same
    policy and training log as an uninterrupted run."""
    import copy
    import torch
    import nflev.training.trainer as T
    cfg = copy.deepcopy(CFG)
    cfg["training"].update(checkpoint_every=2, snapshot_every=1000)
    cfg["training"]["ddpg"].update(warmup=50, batch=16)
    cfg["simulation"]["episode_hours"] = 4
    full, part = tmp_path / "full", tmp_path / "part"
    T.train_nested(cfg, "residential", "ieee33", 0, "none", full, episodes=4, log_every=100)

    class Died(Exception):
        pass
    orig_save = T.TrainState.maybe_save

    def save_then_die(self, ep_done, total, objs):
        orig_save(self, ep_done, total, objs)
        if ep_done == 2:
            raise Died
    monkeypatch.setattr(T.TrainState, "maybe_save", save_then_die)
    with pytest.raises(Died):
        T.train_nested(cfg, "residential", "ieee33", 0, "none", part, episodes=4, log_every=100)
    monkeypatch.setattr(T.TrainState, "maybe_save", orig_save)
    assert (part / "checkpoint.pt").exists()
    T.train_nested(cfg, "residential", "ieee33", 0, "none", part, episodes=4, log_every=100)
    assert not (part / "checkpoint.pt").exists()
    a = torch.load(full / "model.pt", weights_only=False)["l2"][0]["actor"]
    b = torch.load(part / "model.pt", weights_only=False)["l2"][0]["actor"]
    assert all(torch.equal(a[k], b[k]) for k in a)

    def rows(path):        # drop the wall-clock column
        return [",".join(r.split(",")[:-1]) for r in path.read_text().splitlines()]
    assert rows(full / "train_log.csv") == rows(part / "train_log.csv")


def test_price_offset_shifts_execution_price_only_for_calibrated_policies():
    """Tariff calibration adds a constant to learned execution prices (clipped to the
    retail bounds); rule-based policies are unaffected because env.reset clears it."""
    import copy
    from nflev.baselines.simple import Uncoordinated
    cfg = copy.deepcopy(CFG)
    env = ChargingEnv(cfg, "ieee33", q_control=False)
    spec = make_episode(cfg, "residential", "test", 40, seed=7000, eval_index=0)
    env.reset(spec)
    env.price_offset = 0.03
    env.set_corridor(0.10, 0.30)
    for k in range(env.n_agg):
        env.set_aggregate(k, 0.5, 0.5)
    env.run_interval()
    assert np.allclose(env.exec_price, 0.20 + 0.03)
    env.reset(spec)
    assert env.price_offset == 0.0
    pol = Uncoordinated()
    pol.reset(env)
    pol.act(env)
    env.run_interval()
    assert np.allclose(env.exec_price, cfg["retail"]["flat_price"])


def test_gaussian_policy_stays_valid_for_extreme_log_std():
    """CPO's trust-region step once drove log-std to +-inf (crash); the policy now
    clamps it, so the distribution is always valid."""
    import torch
    from nflev.agents.ppo_lagrangian import GaussianPolicy
    pi = GaussianPolicy(4, 3)
    with torch.no_grad():
        pi.log_std.copy_(torch.tensor([float("inf"), -float("inf"), 0.0]))
    d = pi.dist(torch.zeros(2, 4))
    assert torch.isfinite(d.stddev).all() and (d.stddev > 0).all()


def test_plan_allocation_executes_plan_and_serves_urgent_first():
    cap = np.array([11.0, 11.0, 11.0, 11.0])
    lax = np.array([5.0, 0.1, 3.0, 8.0])            # vehicle 1 is urgent (laxity < 0.25 h)
    planned = np.array([False, True, False, True])
    avail = cap.sum()
    # u = u0 (power of the planned vehicles) executes exactly the plan
    rates, _ = llf_allocate(cap[planned].sum() / avail, cap, lax, 1e9, 0.25, True, "plan", planned)
    assert np.allclose(rates, cap * planned)
    # least-laxity-first would serve vehicle 2 (laxity 3) before planned vehicle 3
    rates_llf, _ = llf_allocate(cap[planned].sum() / avail, cap, lax, 1e9, 0.25, True, "llf")
    assert rates_llf[2] > 0 and rates_llf[3] == 0
    # the urgent vehicle is served first even when it is not planned and u is small
    rates, _ = llf_allocate(0.25, cap, lax, 1e9, 0.25, True, "plan", np.array([True, False, True, True]))
    assert rates[1] == pytest.approx(11.0) and rates.sum() == pytest.approx(11.0)


def test_validation_days_are_held_out_of_training():
    from nflev.env.episode import _days
    tr, va = _days(CFG, "train", "residential", 24), _days(CFG, "val", "residential", 24)
    assert len(va) == CFG["data"]["val_days"] and not set(tr) & set(va)
    assert len(tr) + len(va) == len(_days({**CFG, "data": {**CFG["data"], "val_days": 0}}, "train", "residential", 24))


def test_residual_penalty_pulls_actor_to_plan():
    import torch
    from nflev.agents.ddpg import DDPGAgent
    torch.manual_seed(0)
    ag = DDPGAgent(CFG, 4, 2, seed=0, zero_init=True, residual_penalty=100.0)
    with torch.no_grad():                            # start far from the plan
        ag.actor[-2].bias.fill_(2.0)
    rng = np.random.default_rng(0)
    for _ in range(ag.warmup + ag.batch):
        ag.store(rng.random(4).astype(np.float32), rng.random(2).astype(np.float32), 0.0,
                 rng.random(4).astype(np.float32), False)
    a0 = float(ag.act(np.zeros(4, np.float32), explore=False)[0])
    for _ in range(200):
        ag.update()
    a1 = float(ag.act(np.zeros(4, np.float32), explore=False)[0])
    assert abs(a1 - 0.5) < abs(a0 - 0.5) / 2
