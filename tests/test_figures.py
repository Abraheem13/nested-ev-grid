"""Tests for the figures of the paper: the layout check finds overlapping text,
and every figure is drawn without any."""
import pathlib
import sys

import matplotlib
import pytest
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nflev.eval import analysis                              # noqa: E402
from nflev.eval.diagrams import (fig_architecture, fig_l3_loop,  # noqa: E402
                                 fig_pipeline, layout_problems)

CFG = yaml.safe_load(open(ROOT / "configs" / "base.yaml"))


def test_layout_check_finds_overlaps():
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1], [0.5, 0.5], color="black")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    ax.text(0.5, 0.2, "clear of the line", ha="center")
    assert layout_problems(fig) == []
    ax.text(0.5, 0.5, "on the line", ha="center", va="center")
    ax.text(0.5, 0.2, "on the other text", ha="center")
    probs = layout_problems(fig)
    assert any("'on the line' touches a line" in p for p in probs)
    assert any("overlaps text 'on the other text'" in p for p in probs)
    plt.close(fig)


def test_layout_check_ignores_data_outside_the_axes():
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([-5, 1], [0.5, 0.5], color="black")             # left part is clipped away
    ax.text(-0.5, 0.5, "beside the axes", transform=ax.transData, clip_on=False)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    assert layout_problems(fig) == []
    plt.close(fig)


def test_diagrams_have_no_overlapping_text(tmp_path):
    fig_architecture(tmp_path)                              # each raises if any text overlaps
    fig_l3_loop(tmp_path)
    fig_pipeline(CFG, tmp_path, 5, 3)
    assert {p.name for p in tmp_path.iterdir()} == {"fig_architecture.pdf", "fig_l3_loop.pdf", "fig_pipeline.pdf"}


@pytest.mark.skipif(not (ROOT / "artifacts" / "profiles").exists(), reason="needs the stored day profiles")
def test_profile_figure_has_no_overlapping_text(tmp_path):
    analysis.fig_profile(analysis.load_profiles(ROOT / "artifacts"), tmp_path)
    assert (tmp_path / "fig_profile.pdf").exists()
