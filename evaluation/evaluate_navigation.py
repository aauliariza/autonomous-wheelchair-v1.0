#!/usr/bin/env python3
"""Navigation-command and safety-critical evaluation (spec sections AL, AM).

SAFETY ASYMMETRY — the central idea of this script
--------------------------------------------------
Command accuracy alone is the wrong headline for a wheelchair, because its
errors are not symmetric:

* An **unsafe** error says FORWARD when the truth is STOP/TURN. The chair drives
  into an obstacle. This can injure the occupant.
* An **over-cautious** error says STOP when the truth is FORWARD. The chair
  stops unnecessarily. This is annoying, not dangerous.

A model with 95% accuracy whose 5% of errors are all unsafe is far worse than one
with 85% accuracy whose errors are all over-cautious. ``unsafe_command_rate`` is
therefore the primary reported figure and accuracy is secondary (spec AM).

INPUT
-----
A CSV of ground-truth commands per frame, with columns ``frame_id`` and
``gt_command``, joined against the pipeline's own per-frame log (spec section AI).
Without ground-truth labels no navigation accuracy can be computed, and this
script says NOT MEASURED rather than inventing a number.

Usage:
    python evaluation/evaluate_navigation.py --predictions outputs/logs/run.csv \
        --ground-truth data/nav_gt.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io import save_json  # noqa: E402
from utils.logger import get_logger  # noqa: E402

LOG = get_logger("evaluate_navigation")

COMMANDS = ("FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP", "EMERGENCY_STOP")
# Commands that keep the chair moving; predicting one of these when the truth
# was a halt is the unsafe direction.
MOVING = {"FORWARD", "TURN_LEFT", "TURN_RIGHT"}
HALTING = {"STOP", "EMERGENCY_STOP"}


def read_command_csv(path: Path, command_column: str, frame_column: str = "frame_id") -> dict[str, str]:
    """Read ``{frame_id: command}`` from a CSV, taking the first row per frame."""
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path.resolve()}")

    out: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or command_column not in reader.fieldnames:
            raise ValueError(f"Column '{command_column}' not found in {path}. Available: {reader.fieldnames}")
        for row in reader:
            frame = str(row.get(frame_column, "")).strip()
            cmd = str(row.get(command_column, "")).strip().upper()
            # Per-frame logs emit one row per obstacle; the command is constant
            # within a frame, so the first occurrence is authoritative.
            if frame and cmd and frame not in out:
                out[frame] = cmd
    return out


def evaluate(pred: dict[str, str], truth: dict[str, str]) -> dict[str, Any]:
    """Compute command, per-class and safety-critical metrics."""
    frames = sorted(set(pred) & set(truth), key=lambda x: (len(x), x))
    if not frames:
        return {"note": "NOT MEASURED - no overlapping frame_id between predictions and ground truth"}

    correct = 0
    unsafe = 0
    over_cautious = 0
    per_class: dict[str, Counter] = {c: Counter() for c in COMMANDS}
    confusion: Counter = Counter()

    for f in frames:
        p, t = pred[f], truth[f]
        confusion[(t, p)] += 1
        if p == t:
            correct += 1
            per_class.setdefault(t, Counter())["tp"] += 1
        else:
            per_class.setdefault(t, Counter())["fn"] += 1
            per_class.setdefault(p, Counter())["fp"] += 1
            if t in HALTING and p in MOVING:
                unsafe += 1  # should have stopped, kept moving
            elif t in MOVING and p in HALTING:
                over_cautious += 1  # should have moved, stopped
            elif t in MOVING and p in MOVING:
                unsafe += 1  # wrong direction is also a collision risk

    n = len(frames)
    results: dict[str, Any] = {
        "num_frames": n,
        "command_accuracy": correct / n,
        # --- safety-critical (spec section AM), reported first ---
        "unsafe_command_rate": unsafe / n,
        "over_cautious_rate": over_cautious / n,
        "safety_distance_violation_rate": unsafe / n,
        "false_safe_rate": unsafe / n,  # said "safe to move" when it was not
        "false_block_rate": over_cautious / n,
    }

    for cmd in COMMANDS:
        c = per_class.get(cmd, Counter())
        support = c["tp"] + c["fn"]
        results[f"{cmd.lower()}_accuracy"] = (c["tp"] / support) if support else None
        results[f"{cmd.lower()}_support"] = support

    results["confusion"] = {f"{t}->{p}": v for (t, p), v in sorted(confusion.items())}
    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Evaluate navigation command accuracy and safety.")
    ap.add_argument("--predictions", type=Path, required=True, help="Per-frame log CSV from the pipeline")
    ap.add_argument("--ground-truth", type=Path, required=True, help="CSV with frame_id and gt_command")
    ap.add_argument("--pred-column", default="command")
    ap.add_argument("--gt-column", default="gt_command")
    ap.add_argument("--output", type=Path, default=Path("outputs/evaluation/navigation_metrics.json"))
    args = ap.parse_args(argv)

    try:
        pred = read_command_csv(args.predictions, args.pred_column)
        truth = read_command_csv(args.ground_truth, args.gt_column)
        LOG.info("Loaded %d predicted and %d ground-truth frames", len(pred), len(truth))

        results = evaluate(pred, truth)

        print()
        if "note" in results:
            print(results["note"])
        else:
            print(f"frames evaluated           : {results['num_frames']}")
            print(f"command accuracy           : {results['command_accuracy']:.4f}")
            print()
            print("SAFETY-CRITICAL (spec AM) — these outrank accuracy:")
            print(f"  unsafe command rate      : {results['unsafe_command_rate']:.4f}   <- primary metric")
            print(f"  over-cautious rate       : {results['over_cautious_rate']:.4f}   (annoying, not dangerous)")
            print()
            print("per-command accuracy:")
            for cmd in COMMANDS:
                acc = results.get(f"{cmd.lower()}_accuracy")
                sup = results.get(f"{cmd.lower()}_support", 0)
                shown = f"{acc:.4f}" if acc is not None else "NOT MEASURED (no support)"
                print(f"  {cmd:<16} {shown:>26}  (n={sup})")

        save_json(
            {"predictions": str(args.predictions), "ground_truth": str(args.ground_truth), **results}, args.output
        )
        LOG.info("Results written to %s", args.output)
    except (FileNotFoundError, ValueError) as e:
        LOG.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
