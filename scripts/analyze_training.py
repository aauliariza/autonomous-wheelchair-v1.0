#!/usr/bin/env python3
"""Diagnose a depth training run from its Ultralytics ``results.csv``.

Answers three questions the raw CSV makes you compute by hand:

1. Which epoch actually produced ``best.pt``, and what did it score?
2. Is the run overfitting? Reported as the val/train loss ratio over time — a
   ratio that grows while train loss keeps falling is the signature.
3. Did the metric plateau, and how many epochs were spent after the best one?

Every number is read from the CSV. Nothing is estimated or interpolated.

Usage:
    python scripts/analyze_training.py --results outputs/experiments/<run>/results.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

# Ultralytics depth columns. delta1 is "higher is better", the rest are not.
METRIC = "metrics/delta1"
LOWER_IS_BETTER = ("metrics/abs_rel", "metrics/rmse", "metrics/silog")
TRAIN_LOSS, VAL_LOSS = "train/dlog_loss", "val/dlog_loss"


def load(path: Path) -> list[dict[str, Any]]:
    """Read results.csv, keeping only rows with a parseable epoch."""
    if not path.is_file():
        raise FileNotFoundError(
            f"results.csv not found: {path.resolve()}\n"
            f"  Recovery: point --results at outputs/experiments/<run>/results.csv"
        )
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            clean = {k.strip(): v for k, v in row.items() if k}
            try:
                clean["epoch"] = int(float(clean["epoch"]))
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(clean)
    if not rows:
        raise ValueError(f"No usable rows in {path}. Is this an Ultralytics results.csv?")
    return rows


def num(row: dict[str, Any], key: str) -> float | None:
    """Return a float cell, or None when the column is absent or blank."""
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute best epoch, overfitting trend and post-best waste."""
    scored = [r for r in rows if num(r, METRIC) is not None]
    if not scored:
        raise ValueError(f"Column '{METRIC}' missing; this does not look like a depth run.")
    best = max(scored, key=lambda r: num(r, METRIC))

    ratios = [
        (r["epoch"], num(r, VAL_LOSS) / num(r, TRAIN_LOSS))
        for r in rows
        if num(r, TRAIN_LOSS) not in (None, 0.0) and num(r, VAL_LOSS) is not None
    ]
    val_losses = [(r["epoch"], num(r, VAL_LOSS)) for r in rows if num(r, VAL_LOSS) is not None]
    best_val = min(val_losses, key=lambda t: t[1]) if val_losses else None

    return {
        "epochs": len(rows),
        "best": best,
        "ratios": ratios,
        "best_val": best_val,
        "wasted": rows[-1]["epoch"] - best["epoch"],
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Diagnose a depth training run.")
    ap.add_argument("--results", type=Path, required=True, help="Path to results.csv")
    args = ap.parse_args(argv)

    try:
        rows = load(args.results)
        a = analyze(rows)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1

    best = a["best"]
    print(f"epochs recorded : {a['epochs']}")
    print(f"best epoch      : {best['epoch']}  ({METRIC} = {num(best, METRIC):.5f})")
    for k in LOWER_IS_BETTER:
        v = num(best, k)
        if v is not None:
            print(f"  {k:20s}: {v:.5f}")
    print(f"epochs after best: {a['wasted']}  (early-stopping patience was spent here)")

    if a["best_val"]:
        e, v = a["best_val"]
        print(f"lowest val loss : {v:.5f} at epoch {e}")

    if a["ratios"]:
        print("\nval/train loss ratio — a RISING ratio while train loss falls means overfitting:")
        picks = [a["ratios"][0], *[r for r in a["ratios"] if r[0] == best["epoch"]], a["ratios"][-1]]
        for e, r in picks:
            print(f"  epoch {e:>4}: {r:.2f}x")
        first, last = a["ratios"][0][1], a["ratios"][-1][1]
        if last > first * 1.25:
            print(f"  VERDICT: overfitting — the gap widened {first:.2f}x -> {last:.2f}x.")
            print("           Regularize (freeze early layers, photometric augmentation,")
            print("           higher weight decay) rather than training longer.")
        else:
            print(f"  VERDICT: no strong overfitting signal ({first:.2f}x -> {last:.2f}x).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
