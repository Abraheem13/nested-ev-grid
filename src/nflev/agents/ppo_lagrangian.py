"""PPO-Lagrangian (constrained RL baseline).

    max_pi J_r(pi) - kappa (J_c(pi) - d),   kappa <- [kappa + eta (J_c - d)]_+

Per-interval cost c_t = 1[min voltage < V_min] + 10 * max(0, V_min - V_min,t).
The policy is a diagonal Gaussian over pre-sigmoid actions; the environment
receives sigmoid(raw) in [0, 1]^d. One update per training episode.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

HIDDEN = (256, 128, 64)


def _mlp(i, o):
    layers, d = [], i
    for h in HIDDEN:
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    def __init__(self, s_dim: int, a_dim: int):
        super().__init__()
        self.mu = _mlp(s_dim, a_dim)
        self.log_std = nn.Parameter(torch.full((a_dim,), -0.5))

    def dist(self, s):
        return torch.distributions.Normal(self.mu(s), self.log_std.exp())


class Critic(nn.Module):
    def __init__(self, s_dim: int):
        super().__init__()
        self.net = _mlp(s_dim, 1)

    def forward(self, s):
        return self.net(s).squeeze(-1)


def gae(r: np.ndarray, v: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Single-episode GAE; the episode ends after the last step (v_{T} = 0)."""
    adv = np.zeros_like(r)
    last = 0.0
    for t in reversed(range(len(r))):
        v_next = v[t + 1] if t + 1 < len(r) else 0.0
        delta = r[t] + gamma * v_next - v[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + v


class OnPolicyBase:
    def __init__(self, cfg: dict, s_dim: int, a_dim: int):
        p = cfg["training"]["ppo"]
        self.gamma, self.lam, self.clip, self.epochs = p["gamma"], p["gae_lambda"], p["clip"], p["epochs"]
        self.pi = GaussianPolicy(s_dim, a_dim)
        self.vr, self.vc = Critic(s_dim), Critic(s_dim)
        self.traj = []

    @torch.no_grad()
    def act(self, s: np.ndarray, explore: bool):
        d = self.pi.dist(torch.as_tensor(s, dtype=torch.float32))
        raw = d.sample() if explore else d.mean
        return torch.sigmoid(raw).numpy(), raw.numpy(), float(d.log_prob(raw).sum())

    def store(self, s, raw, logp, r, c):
        self.traj.append((s, raw, logp, r, c))

    def _batch(self):
        s = torch.as_tensor(np.array([t[0] for t in self.traj]), dtype=torch.float32)
        raw = torch.as_tensor(np.array([t[1] for t in self.traj]), dtype=torch.float32)
        logp = torch.as_tensor([t[2] for t in self.traj], dtype=torch.float32)
        r = np.array([t[3] for t in self.traj], np.float32)
        c = np.array([t[4] for t in self.traj], np.float32)
        return s, raw, logp, r, c

    def state_dict(self) -> dict:
        return {"pi": self.pi.state_dict(), "vr": self.vr.state_dict(), "vc": self.vc.state_dict(),
                **({"kappa": self.kappa} if hasattr(self, "kappa") else {})}

    def load_state_dict(self, sd: dict) -> None:
        self.pi.load_state_dict(sd["pi"])
        self.vr.load_state_dict(sd["vr"])
        self.vc.load_state_dict(sd["vc"])
        if "kappa" in sd:
            self.kappa = sd["kappa"]


class PPOLagrangian(OnPolicyBase):
    def __init__(self, cfg: dict, s_dim: int, a_dim: int, cost_limit: float = 0.0,
                 lr_dual: float = 0.05):
        super().__init__(cfg, s_dim, a_dim)
        self.opt = torch.optim.Adam([*self.pi.parameters(), *self.vr.parameters(),
                                     *self.vc.parameters()], lr=cfg["training"]["ppo"]["lr"])
        self.kappa, self.d, self.lr_dual = 0.0, cost_limit, lr_dual

    def update(self) -> None:
        if not self.traj:
            return
        s, raw, logp_old, r, c = self._batch()
        with torch.no_grad():
            vr, vc = self.vr(s).numpy(), self.vc(s).numpy()
        adv_r, ret_r = gae(r, vr, self.gamma, self.lam)
        adv_c, ret_c = gae(c, vc, self.gamma, self.lam)
        self.kappa = max(0.0, self.kappa + self.lr_dual * (float(c.sum()) - self.d))
        adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
        adv_c = (adv_c - adv_c.mean()) / (adv_c.std() + 1e-8)
        adv = torch.as_tensor((adv_r - self.kappa * adv_c) / (1.0 + self.kappa))
        ret_r, ret_c = torch.as_tensor(ret_r), torch.as_tensor(ret_c)
        for _ in range(self.epochs):
            ratio = torch.exp(self.pi.dist(s).log_prob(raw).sum(-1) - logp_old)
            l_pi = -torch.min(ratio * adv, ratio.clamp(1 - self.clip, 1 + self.clip) * adv).mean()
            loss = l_pi + 0.5 * nn.functional.mse_loss(self.vr(s), ret_r) \
                + 0.5 * nn.functional.mse_loss(self.vc(s), ret_c)
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.pi.parameters(), 0.5)
            self.opt.step()
        self.traj.clear()
