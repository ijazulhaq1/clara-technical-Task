import json
from pathlib import Path

import matplotlib.pyplot as plt

from evaluation.scenarios import generate_stream, SCENARIOS
from main import run_stream
from src.config import load_catalogue, load_policy


def _panel(ax, events, records, title):
    t = [e["timestamp"] for e in events]
    ax.plot(t, [1.0 - e["response_quality"] for e in events], label="negative response quality")
    ax.plot(t, [e["participation_imbalance"] for e in events], label="participation imbalance")
    ax.plot(t, [e["goal_convergence"] for e in events], label="goal convergence")
    ax.plot(t, [e["state_confidence"] for e in events], label="upstream confidence", alpha=0.65)

    # Mark decisions produced by the runtime. No ground-truth fields are used here.
    trigger_records = [r for r in records if r.get("triggered")]
    for r in trigger_records:
        x = r["timestamp"]
        ax.axvline(x, linestyle="--", linewidth=1.2)
        ax.annotate(
            r.get("action", "decision"),
            xy=(x, 0.96), xycoords=("data", "axes fraction"),
            xytext=(4, -4), textcoords="offset points", fontsize=8, rotation=90,
            va="top",
        )

    ax.set_title(title, loc="left")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("signal")
    ax.grid(alpha=0.15)


def make_timeline(output_path="outputs/timeline.png", seed=7):
    catalogue = load_catalogue()
    policy = load_policy()
    fig, axes = plt.subplots(len(SCENARIOS), 1, figsize=(14, 12), sharex=True)

    for ax, scenario in zip(axes, SCENARIOS):
        events = list(generate_stream(seed, scenario))
        records = run_stream(events, catalogue, policy)
        _panel(ax, events, records, f"G{seed:03d} — {scenario.replace('_', ' ')}")

    axes[-1].set_xlabel("session time (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=4, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(make_timeline())
