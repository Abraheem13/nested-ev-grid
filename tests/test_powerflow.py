"""The radial solver must reproduce pandapower's Newton-Raphson on both feeders."""
import pathlib
import sys

import numpy as np
import pandapower as pp
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from nflev.grid.network import load_network          # noqa: E402
from nflev.grid.powerflow import RadialPowerFlow     # noqa: E402


def to_pandapower(rn, p_mw, q_mvar, v0):
    net = pp.create_empty_network(sn_mva=rn.base_mva)
    b = [pp.create_bus(net, vn_kv=rn.base_kv) for _ in range(rn.n_bus)]
    pp.create_ext_grid(net, b[0], vm_pu=v0)
    zbase = rn.base_kv ** 2 / rn.base_mva
    for i in range(1, rn.n_bus):
        z = rn.z_branch[i] * zbase
        pp.create_line_from_parameters(net, b[rn.parent[i]], b[i], length_km=1.0,
                                       r_ohm_per_km=z.real, x_ohm_per_km=z.imag,
                                       c_nf_per_km=0.0, max_i_ka=1.0)
    for i in range(1, rn.n_bus):
        if p_mw[i] != 0 or q_mvar[i] != 0:
            pp.create_load(net, b[i], p_mw=p_mw[i], q_mvar=q_mvar[i])
    return net


@pytest.mark.parametrize("name", ["ieee33", "ieee69"])
@pytest.mark.parametrize("scale,ev_mw,q_inj", [(1.0, 0.0, 0.0), (0.85, 0.15, 0.0),
                                                (1.1, 0.25, 0.08), (0.4, 0.0, 0.05)])
def test_matches_newton_raphson(name, scale, ev_mw, q_inj):
    rn = load_network(name)
    rng = np.random.default_rng(0)
    p = rn.p_load_mw * scale * rng.uniform(0.9, 1.1, rn.n_bus)
    q = rn.q_load_mvar * scale * rng.uniform(0.9, 1.1, rn.n_bus)
    p[0] = q[0] = 0.0
    weak = rn.order[-5:]                      # deepest buses
    p[weak] += ev_mw
    q[weak] -= q_inj                          # reactive injection = negative load
    pf = RadialPowerFlow(rn, v0=1.03)
    res = pf.solve(p, q)
    net = to_pandapower(rn, p, q, 1.03)
    pp.runpp(net, algorithm="nr", tolerance_mva=1e-11, max_iteration=60, init="flat")
    vm_nr = net.res_bus.vm_pu.values
    assert np.max(np.abs(res.vm - vm_nr)) < 1e-8
    p_sub_nr = net.res_ext_grid.p_mw.iloc[0]
    q_sub_nr = net.res_ext_grid.q_mvar.iloc[0]
    assert abs(res.s_sub_mva.real - p_sub_nr) < 1e-7
    assert abs(res.s_sub_mva.imag - q_sub_nr) < 1e-7


def test_network_sizes_and_totals():
    n33, n69 = load_network("ieee33"), load_network("ieee69")
    assert n33.n_bus == 33 and n69.n_bus == 69
    assert abs(n33.p_load_mw.sum() - 3.715) < 1e-9      # Baran & Wu 33-bus total load
    assert abs(n69.p_load_mw.sum() - 3.8021) < 1e-9     # MATPOWER case69 total load
