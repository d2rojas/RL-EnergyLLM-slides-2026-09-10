#!/usr/bin/env python3
"""Regenerate choice_evolution.png and f3_pareto.png for the Sep-10 deck.

Data source: the eval outputs pulled from the Nautilus PVC (energy-ckpt-pvc,
/ckpt/eval/<config>/{turns.csv,aggregate.json}). Pass the local copy of that
eval/ directory as argv[1] (default: the session scratchpad pull).

Energy convention: aggregate's mean_j_per_episode_ref (measured, board-calibrated
to 4.0 J/token) when present and > 0; otherwise tokens/episode x 4.0 J/token
(the hardware-free currency) -- the two agree by construction of the calibration.
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVAL = Path(sys.argv[1] if len(sys.argv) > 1 else
            "/private/tmp/claude-501/-Users-lzanda-Library-CloudStorage-OneDrive-UCSanDiego-"
            "Docs-UCSD-RL-Energy-clean/685aa2be-993d-4f96-85e0-375cc01b520b/scratchpad/evalpull/eval")
FIGS = Path(__file__).resolve().parent.parent / "figures"
REF_JPT = 4.0
MAXT = 30

LEVELS = ["none", "low", "mid", "high"]
LCOLOR = {"none": "#999999", "low": "#1baf7a", "mid": "#eb6834", "high": "#e34948"}
LLABEL = {"none": "no tag", "low": "low", "mid": "mid", "high": "high"}


def turns(tag):
    with open(EVAL / tag / "turns.csv") as f:
        return list(csv.DictReader(f))


def agg(tag):
    return json.load(open(EVAL / tag / "aggregate.json"))


# ---------------------------------------------------------------- choice evolution
def choice_evolution():
    panels = [("base", "base (untrained)"), ("A2_step10", "A2 @ step 10"), ("A2_step60", "A2 @ step 60")]
    fig, axes = plt.subplots(2, 3, figsize=(20.8, 9.6), sharex=True)
    for col, (tag, title) in enumerate(panels):
        rows = turns(tag)
        share = {lv: [] for lv in LEVELS}
        toks, inval = [], []
        for t in range(MAXT):
            rt = [r for r in rows if int(r["turn"]) == t]
            n = max(1, len(rt))
            for lv in LEVELS:
                share[lv].append(100 * sum(1 for r in rt if (r["effort"] or "none") == lv) / n)
            toks.append(sum(float(r["gen_tok"]) for r in rt) / n)
            inval.append(100 * sum(1 for r in rt if r["valid"] == "0") / n)
        ax = axes[0][col]
        ax.stackplot(range(MAXT), [share[lv] for lv in LEVELS],
                     colors=[LCOLOR[lv] for lv in LEVELS],
                     labels=[LLABEL[lv] for lv in LEVELS])
        ax.set_title(title, fontsize=15)
        ax.set_ylim(0, 100)
        if col == 0:
            ax.set_ylabel("% of turns", fontsize=13)
        if col == 2:
            ax.legend(loc="center right", fontsize=10)
        ax2 = axes[1][col]
        ax2.plot(range(MAXT), toks, color="#222222", lw=2.2, label="tokens/turn")
        ax2.plot(range(MAXT), inval, color="#e34948", lw=1.8, ls="--", label="% invalid actions")
        ax2.set_ylim(0, 160)
        mean_inv = sum(inval) / len(inval)
        ax2.annotate(f"invalid actions: {mean_inv:.1f}% of turns",
                     xy=(0.5, 0.30), xycoords="axes fraction", ha="center",
                     fontsize=12, color="#e34948")
        ax2.set_xlabel("turn within the episode", fontsize=13)
        if col == 0:
            ax2.set_ylabel("tokens / % invalid", fontsize=13)
            ax2.legend(fontsize=10)
        for a in (ax, ax2):
            a.grid(alpha=0.25)
    fig.suptitle(
        "Choice per turn (134 games each): base chooses erratically (scattered levels, broken format, long)\n"
        "$\\rightarrow$ training converges to consistent but UNCONDITIONAL choice (always mid, short). "
        "Missing: state-conditional choice --- runs D/F and the decision test target exactly that.",
        fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGS / "choice_evolution.png", dpi=100)
    print("wrote choice_evolution.png")


# ---------------------------------------------------------------- pareto
def point(tag):
    a = agg(tag)
    succ = 100 * a["success_rate"]
    n = a.get("n_episodes") or 134
    se = (succ * (100 - succ) / n) ** 0.5
    kj = a.get("mean_j_per_episode_ref", 0) / 1000
    if kj <= 0:
        rows = turns(tag)
        per_ep = defaultdict(float)
        for r in rows:
            per_ep[r["task"]] += float(r["gen_tok"])
        kj = REF_JPT * sum(per_ep.values()) / len(per_ep) / 1000
    return kj, succ, se


def pareto():
    fig, ax = plt.subplots(figsize=(13, 8))
    series = [
        ("A2", "#e34948", "s", [f"A2_step{s}" for s in (10, 20, 30, 40, 50, 60)]),
        ("B2 ($\\lambda$=0)", "#2a78d6", "o", [f"B2_step{s}" for s in (10, 20, 30, 40, 50, 60)]),
    ]
    for name, color, marker, tags in series:
        xs, ys = [], []
        for tag in tags:
            kj, succ, se = point(tag)
            xs.append(kj); ys.append(succ)
            ax.errorbar(kj, succ, yerr=se, color=color, marker=marker, ms=9,
                        capsize=3, lw=0, elinewidth=1.2)
            ax.annotate(tag.split("step")[1], (kj, succ), textcoords="offset points",
                        xytext=(7, 4), fontsize=10, color=color)
        ax.plot(xs, ys, color=color, alpha=0.35, lw=1.4, label=f"{name} steps 10$\\to$60")
    for tag, label in [("base", "base free"), ("base_low", "b·low"),
                       ("base_mid", "b·mid"), ("base_high", "b·high")]:
        kj, succ, se = point(tag)
        ax.errorbar(kj, succ, yerr=se, color="#555555", marker="o", mfc="none", ms=10,
                    capsize=3, lw=0, elinewidth=1.2)
        ax.annotate(label, (kj, succ), textcoords="offset points", xytext=(7, 5),
                    fontsize=10, color="#555555")
    ax.set_xlabel("kJ per episode at 4.0 J/token (board-calibrated)", fontsize=13)
    ax.set_ylabel("success %, 134 OOD games ($\\pm$1 s.e.)", fontsize=13)
    ax.set_title("F3: the full A2 ($\\lambda$=0.5) and B2 ($\\lambda$=0) series vs the untrained base\n"
                 "the two series interleave within $\\pm$1 s.e. at every step --- "
                 "the $\\lambda$ term never separates them",
                 fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "f3_pareto.png", dpi=100)
    print("wrote f3_pareto.png")


if __name__ == "__main__":
    choice_evolution()
    pareto()
