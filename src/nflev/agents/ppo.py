"""PPO (Schulman et al., 2017) with a Beta policy on [0, 1]^d (Level 1 DSO).

Each Beta distribution has concentration parameters softplus(.) + 1, so the
density is unimodal and actions stay inside the bounded price box without any
squashing transform. Deterministic evaluation uses the Beta mean.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta

HIDDEN = (256, 128, 64)


def _body(s_dim):
    layers, d = [], s_dim
    for h in HIDDEN:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    return nn.Sequential(*layers), d


class BetaPolicy(nn.Module):
    def __init__(self, s_dim: int, a_dim: int):
        super().__init__()
        self.body, d = _body(s_dim)
        self.alpha = nn.Linear(d, a_dim)
        self.beta = nn.Linear(d, a_dim)

    def dist(self, s):
        z = self.body(s)
        return Beta(nn.functional.softplus(self.alpha(z)) + 1.0,
                    nn.functional.softplus(self.beta(z)) + 1.0)


class ValueNet(nn.Module):
    def __init__(self, s_dim: int):
        super().__init__()
        self.body, d = _body(s_dim)
        self.head = nn.Linear(d, 1)

    def forward(self, s):
        return self.head(self.body(s)).squeeze(-1)


class PPOAgent:
    def __init__(self, cfg: dict, s_dim: int, a_dim: int = 2):
        p = cfg["training"]["ppo"]
        self.gamma, self.lam, self.clip = p["gamma"], p["gae_lambda"], p["clip"]
        self.epochs, self.ent = p["epochs"], p["entropy"]
        self.pi = BetaPolicy(s_dim, a_dim)
        self.v = ValueNet(s_dim)
        self.opt = torch.optim.Adam([*self.pi.parameters(), *self.v.parameters()], lr=p["lr"])
        self.buf = []

    @torch.no_grad()
    def act(self, s: np.ndarray, explore: bool) -> tuple[np.ndarray, float]:
        d = self.pi.dist(torch.as_tensor(s, dtype=torch.float32))
        a = d.sample() if explore else d.mean
        a = a.clamp(1e-4, 1 - 1e-4)
        return a.numpy(), float(d.log_prob(a).sum())

    def store(self, s, a, logp, r, done):
        self.buf.append((s, a, logp, r, done))

    def update(self) -> None:
        if len(self.buf) < 2:
            self.buf.clear()
            return
        s = torch.as_tensor(np.array([b[0] for b in self.buf]), dtype=torch.float32)
        a = torch.as_tensor(np.array([b[1] for b in self.buf]), dtype=torch.float32)
        logp_old = torch.as_tensor([b[2] for b in self.buf], dtype=torch.float32)
        r = np.array([b[3] for b in self.buf], np.float32)
        done = np.array([b[4] for b in self.buf], np.float32)
        with torch.no_grad():
            v = self.v(s).numpy()
        adv = np.zeros_like(r)
        last = 0.0
        for t in reversed(range(len(r))):
            v_next = 0.0 if (done[t] or t + 1 == len(r)) else v[t + 1]
            delta = r[t] + self.gamma * v_next - v[t]
            last = delta + self.gamma * self.lam * (0.0 if done[t] else last)
            adv[t] = last
        ret = torch.as_tensor(adv + v)
        adv_t = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8))
        for _ in range(self.epochs):
            d = self.pi.dist(s)
            ratio = torch.exp(d.log_prob(a).sum(-1) - logp_old)
            l_pi = -torch.min(ratio * adv_t, ratio.clamp(1 - self.clip, 1 + self.clip) * adv_t).mean()
            loss = l_pi + 0.5 * nn.functional.mse_loss(self.v(s), ret) - self.ent * d.entropy().sum(-1).mean()
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([*self.pi.parameters(), *self.v.parameters()], 0.5)
            self.opt.step()
        self.buf.clear()

    def state_dict(self) -> dict:
        return {"pi": self.pi.state_dict(), "v": self.v.state_dict()}

    def load_state_dict(self, sd: dict) -> None:
        self.pi.load_state_dict(sd["pi"])
        self.v.load_state_dict(sd["v"])
