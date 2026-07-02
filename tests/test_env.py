"""Day 1 smoke tests: environment physics, Q-controller behavior, deadlock fix,
projection layer correctness."""
import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import yaml

from nflev.env.charging_env import ChargingEnv
from nflev.agents.projection import simplex_scale, bounded_simplex_projection
from nflev.env.qcontrol import ReactivePowerController

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs/base.yaml"))
DS = yaml.safe_load(open(ROOT / "configs/dataset/nrel.yaml"))

# PJM-like hourly LMP ($/MWh) and IEEE residential load-shape multiplier
LMP = np.array([28,26,25,25,26,30,45,60,55,50,48,47,46,48,52,60,80,110,120,105,85,60,45,35], float)
LOAD = np.array([.55,.5,.48,.47,.48,.55,.65,.75,.72,.70,.68,.67,.66,.68,.72,.80,.92,1.0,1.0,.97,.88,.78,.68,.60])


def make_env(**mods):
    return ChargingEnv(CFG, "ieee33", DS, n_ev=60, seed=42, scenario_mods=mods)


def test_projection():
    c_max = np.full(5, 11.0)
    c = simplex_scale(np.array([10, 10, 10, 10, 10.0]), c_max, 30.0)
    assert abs(c.sum() - 30.0) < 1e-9
    c2 = bounded_simplex_projection(np.array([15, 2, 9, 9, 9.0]), c_max, 30.0)
    assert c2.sum() <= 30.0 + 1e-6 and (c2 <= c_max + 1e-9).all() and (c2 >= 0).all()
    print("[PASS] projection: sum<=P_cap, box bounds hold")


def test_q_capacity_at_zero_power():
    q = ReactivePowerController.q_capacity_kvar(np.array([12.0]), np.array([0.0]))
    assert q[0] > 11.9, "full Q at P=0 (deadlock fix)"
    q2 = ReactivePowerController.q_capacity_kvar(np.array([12.0]), np.array([11.0]))
    assert 4.0 < q2[0] < 5.0
    print(f"[PASS] Q capacity: P=0 -> {q[0]:.2f} kVAR (STATCOM mode), P=11kW -> {q2[0]:.2f} kVAR")


def run_episode(env, greedy=True, label=""):
    env.reset(LMP, LOAD)
    t0 = time.time()
    n_intervals = int(env.episode_h * 3600 // env.dispatch_s)
    for i in range(n_intervals):
        if env.t_s % env.pricing_s == 0:
            env.set_corridor(0.08, 0.20)
        for k in range(env.n_agg):
            n = len(env.fleet.connected_by_aggregator(k))
            raw = np.full(max(n, 1), 11.0 if greedy else 4.0)
            env.set_dispatch(k, raw, price_frac=0.5)
        env.run_dispatch_interval()
    m = env.episode_metrics()
    m["wall_s"] = time.time() - t0
    print(f"[{label}] cost=${m['daily_cost_usd']:.2f} vmin={m['min_voltage_pu']:.4f} "
          f"viol={m['violation_rate_pct']:.1f}% SQ={m['service_quality']:.3f} "
          f"Qact={m['q_activation_freq']:.2f} curt={m['curtailed_kwh']:.1f}kWh "
          f"peak={m['peak_mw']:.2f}MW wall={m['wall_s']:.1f}s")
    return m


def test_uncoordinated_with_q_control():
    m = run_episode(make_env(), greedy=True, label="S3 greedy + Q-control")
    assert m["violation_rate_pct"] == 0.0, "Q-control must hold the floor"
    assert m["min_voltage_pu"] >= 0.95 - 1e-9


def test_deadlock_scenario():
    """S6: charging forced to zero 19:00-20:00 during peak. v1 model would
    have zero Q capacity here; v2 must keep supporting voltage."""
    m = run_episode(make_env(force_zero_charging_window=[19, 20]),
                    greedy=True, label="S6 deadlock window")
    assert m["violation_rate_pct"] == 0.0, "S_rated-based Q must survive zero charging"


if __name__ == "__main__":
    test_projection()
    test_q_capacity_at_zero_power()
    test_uncoordinated_with_q_control()
    test_deadlock_scenario()
    print("\nAll Day-1 smoke tests passed.")
