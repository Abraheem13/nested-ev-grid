"""Run one policy on one episode."""
from __future__ import annotations


def run_policy_episode(env, policy, spec) -> dict:
    env.reset(spec)
    policy.reset(env)
    while not env.done:
        policy.act(env)
        env.run_interval()
    return env.episode_metrics()
