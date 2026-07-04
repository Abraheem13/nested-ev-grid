"""PPO-Lagrangian: constrained RL baseline (R2.4).

Flat single-agent PPO with a learned cost critic and dual-ascent Lagrange
multiplier on the voltage-violation constraint:

    max_pi  J_r(pi) - lambda * (J_c(pi) - d),   lambda >= 0
    lambda <- [lambda + lr_dual * (J_c - d)]_+

Cost signal c_t = 1[violation in interval t] (+ graded voltage deficit),
constraint limit d = 0.  Q-control is DISABLED for this baseline: constraint
satisfaction must be achieved by the learned policy alone, which is exactly
the capability being compared against the physics-informed Level 3.

State/action interface identical to FlatDDPG (338-d state, 105-d action).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from ..baselines.flat_ddpg import MAX_EVS_FLAT


class GaussianPolicy(nn.Module):
    def __init__(self, s_dim, a_dim, hidden=(256, 256)):
        super().__init__()
        layers, d = [], s_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        self.body = nn.Sequential(*layers)
        self.mu = nn.Linear(d, a_dim)
        self.log_std = nn.Parameter(torch.full((a_dim,), -0.7))

    def dist(self, s):
        return torch.distributions.Normal(self.mu(self.body(s)),
                                          self.log_std.exp())


class Critic(nn.Module):
    def __init__(self, s_dim, hidden=(256, 256)):
        super().__init__()
        layers, d = [], s_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, s):
        return self.net(s).squeeze(-1)


def gae(r, v, gamma, lam):
    adv = np.zeros_like(r)
    last = 0.0
    for t in reversed(range(len(r))):
        nxt = v[t + 1] if t + 1 < len(r) else 0.0
        delta = r[t] + gamma * nxt - v[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + v[:len(r)]


class PPOLagrangian:
    name = "ppo_lagrangian"

    def __init__(self, cfg: dict, n_agg: int = 5, device: str = "cpu",
                 cost_limit: float = 0.0, lr_dual: float = 0.05):
        self.cfg, self.n_agg, self.device = cfg, n_agg, device
        self.s_dim = 8 + n_agg * (6 + MAX_EVS_FLAT * 3)
        self.a_dim = n_agg * (MAX_EVS_FLAT + 1)
        self.pi = GaussianPolicy(self.s_dim, self.a_dim).to(device)
        self.vr = Critic(self.s_dim).to(device)
        self.vc = Critic(self.s_dim).to(device)
        self.opt = torch.optim.Adam(
            [*self.pi.parameters(), *self.vr.parameters(), *self.vc.parameters()],
            lr=3e-4)
        self.lam = 0.0
        self.d, self.lr_dual = cost_limit, lr_dual
        self.gamma, self.lae = 0.99, 0.95
        self.clip, self.epochs = 0.2, 10
        self.traj = []

    @torch.no_grad()
    def act(self, s, deterministic=False):
        st = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        dist = self.pi.dist(st)
        raw = dist.mean if deterministic else dist.sample()
        a = torch.sigmoid(raw).cpu().numpy()
        return a, raw.cpu().numpy(), float(dist.log_prob(raw).sum())

    def store(self, s, raw, logp, r, c):
        self.traj.append((s, raw, logp, r, c))

    def update(self) -> dict:
        if not self.traj:
            return {}
        s = torch.as_tensor(np.array([t[0] for t in self.traj]), dtype=torch.float32, device=self.device)
        raw = torch.as_tensor(np.array([t[1] for t in self.traj]), dtype=torch.float32, device=self.device)
        logp_old = torch.as_tensor([t[2] for t in self.traj], dtype=torch.float32, device=self.device)
        r = np.array([t[3] for t in self.traj], np.float32)
        c = np.array([t[4] for t in self.traj], np.float32)

        with torch.no_grad():
            vr = self.vr(s).cpu().numpy()
            vc = self.vc(s).cpu().numpy()
        adv_r, ret_r = gae(r, vr, self.gamma, self.lae)
        adv_c, ret_c = gae(c, vc, self.gamma, self.lae)
        adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
        adv_c = (adv_c - adv_c.mean()) / (adv_c.std() + 1e-8)

        # dual ascent on episode cost
        Jc = float(c.sum())
        self.lam = max(0.0, self.lam + self.lr_dual * (Jc - self.d))

        adv = torch.as_tensor((adv_r - self.lam * adv_c) / (1 + self.lam),
                              dtype=torch.float32, device=self.device)
        ret_r_t = torch.as_tensor(ret_r, device=self.device)
        ret_c_t = torch.as_tensor(ret_c, device=self.device)

        for _ in range(self.epochs):
            dist = self.pi.dist(s)
            logp = dist.log_prob(raw).sum(-1)
            ratio = torch.exp(logp - logp_old)
            l_pi = -torch.min(ratio * adv,
                              ratio.clamp(1 - self.clip, 1 + self.clip) * adv).mean()
            loss = (l_pi + 0.5 * nn.functional.mse_loss(self.vr(s), ret_r_t)
                    + 0.5 * nn.functional.mse_loss(self.vc(s), ret_c_t))
            self.opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.pi.parameters(), 0.5)
            self.opt.step()
        self.traj.clear()
        return {"lambda": self.lam, "Jc": Jc}

    def save(self, path):
        torch.save({"pi": self.pi.state_dict(), "vr": self.vr.state_dict(),
                    "vc": self.vc.state_dict(), "lam": self.lam}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.pi.load_state_dict(ck["pi"]); self.vr.load_state_dict(ck["vr"])
        self.vc.load_state_dict(ck["vc"]); self.lam = ck.get("lam", 0.0)