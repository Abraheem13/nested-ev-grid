"""Radial distribution feeders built from the MATPOWER case files in data/networks.

Both feeders are radial once the open tie switches (MATPOWER branch status 0)
are removed. `RadialNetwork` holds everything the power-flow solver and the
environment need as plain numpy arrays, in per unit on (base_mva, base_kv).

Assumed ratings (not part of the MATPOWER data, used only for the Level-1
loading features): every line has an ampacity of `LINE_AMPACITY_KA` and the
substation transformer is rated `SUBSTATION_MVA[name]`. Neither limit binds in
any experiment; both are documented as assumptions in the paper.
"""
from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
NETWORK_DIR = ROOT / "data" / "networks"
CASE_FILES = {"ieee33": "case33bw.m", "ieee69": "case69.m"}
LINE_AMPACITY_KA = 0.4
SUBSTATION_MVA = {"ieee33": 8.0, "ieee69": 8.0}


@dataclass
class RadialNetwork:
    name: str
    base_mva: float
    base_kv: float
    n_bus: int                 # buses indexed 0..n_bus-1, bus 0 is the substation
    parent: np.ndarray         # parent[i] for i>=1 (parent[0] = -1)
    z_branch: np.ndarray       # complex p.u. impedance of the branch feeding bus i (index i>=1)
    p_load_mw: np.ndarray      # nominal active load per bus (MW)
    q_load_mvar: np.ndarray    # nominal reactive load per bus (Mvar)
    order: np.ndarray          # buses in breadth-first order from the root
    depth: np.ndarray

    @property
    def i_base_ka(self) -> float:
        return self.base_mva / (np.sqrt(3.0) * self.base_kv)

    def z_path(self) -> np.ndarray:
        """Zpath[i, j] = sum of branch impedances on the common part of the
        root->i and root->j paths (buses 1..n-1). V = V0 - Zpath @ I."""
        n = self.n_bus
        anc = [set() for _ in range(n)]          # branches (identified by child bus) on root->i path
        for i in self.order[1:]:
            anc[i] = anc[self.parent[i]] | {int(i)}
        zp = np.zeros((n - 1, n - 1), dtype=complex)
        for a in range(1, n):
            for b in range(a, n):
                common = anc[a] & anc[b]
                val = sum(self.z_branch[c] for c in common)
                zp[a - 1, b - 1] = zp[b - 1, a - 1] = val
        return zp

    def downstream(self) -> np.ndarray:
        """T[c-1, j-1] = 1 if bus j is in the subtree rooted at bus c
        (branch currents I_branch = T @ I_bus)."""
        n = self.n_bus
        t = np.zeros((n - 1, n - 1))
        for j in range(1, n):
            k = j
            while k > 0:
                t[k - 1, j - 1] = 1.0
                k = self.parent[k]
        return t


def _matpower_block(text: str, name: str) -> np.ndarray:
    m = re.search(r"mpc\.%s\s*=\s*\[(.*?)\];" % name, text, re.S)
    if m is None:
        raise ValueError(f"block mpc.{name} not found")
    rows = []
    for line in m.group(1).strip().splitlines():
        line = line.split("%")[0].strip().rstrip(";").strip()
        if line:
            rows.append([float(x) for x in line.split()])
    return np.array(rows)


def _matpower_scalar(text: str, name: str) -> float:
    m = re.search(r"mpc\.%s\s*=\s*([0-9.eE+-]+)\s*;" % name, text)
    return float(m.group(1))


def load_network(name: str) -> RadialNetwork:
    text = (NETWORK_DIR / CASE_FILES[name]).read_text()
    base_mva = _matpower_scalar(text, "baseMVA")
    bus = _matpower_block(text, "bus")
    br = _matpower_block(text, "branch")
    br = br[br[:, 10] > 0]                       # drop open tie switches
    ids = bus[:, 0].astype(int)
    idx = {b: i for i, b in enumerate(ids)}
    n = len(ids)
    slack = [i for i in range(n) if int(bus[i, 1]) == 3]
    if slack != [0]:
        raise ValueError("expected the slack bus to be the first bus")
    base_kv = float(bus[0, 9])
    z_base = base_kv ** 2 / base_mva
    adj: dict[int, list[tuple[int, complex]]] = {i: [] for i in range(n)}
    for f, t, r, x in br[:, :4]:
        a, b = idx[int(f)], idx[int(t)]
        z = complex(r, x) / z_base               # MATPOWER v2 case files give ohms
        adj[a].append((b, z))
        adj[b].append((a, z))
    if len(br) != n - 1:
        raise ValueError(f"{name}: {len(br)} closed branches for {n} buses is not radial")
    parent = -np.ones(n, dtype=int)
    zb = np.zeros(n, dtype=complex)
    depth = np.zeros(n, dtype=int)
    order, seen = [0], {0}
    for u in order:                              # BFS from the substation
        for v, z in adj[u]:
            if v not in seen:
                seen.add(v)
                parent[v], zb[v], depth[v] = u, z, depth[u] + 1
                order.append(v)
    if len(order) != n:
        raise ValueError(f"{name}: network is not connected")
    return RadialNetwork(name=name, base_mva=base_mva, base_kv=base_kv, n_bus=n,
                         parent=parent, z_branch=zb,
                         p_load_mw=bus[:, 2] / 1000.0, q_load_mvar=bus[:, 3] / 1000.0,
                         order=np.array(order), depth=depth)
