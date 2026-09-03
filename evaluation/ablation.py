#!/usr/bin/env python3
"""Aggregate experiments into the ablation table (spec sections AP, BJ).

Collects every experiment's ``metrics.json`` and emits a single comparison table
in Markdown and CSV.

NO FABRICATION (spec section BK)
--------------------------------
Any cell whose value was not actually measured is printed as ``NOT MEASURED``.
The table never interpolates, estimates, or carries a number over from a
different configuration. A half-empty table is a truthful report of what has been
run; a full table of invented numbers is not.

Usage:
    python evaluation/ablation.py --experiments outputs/experiments --output docs/ablation
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.logger import get_logger  # noqa: E402

LOG = get_logger("ablation")

NOT_MEASURED = "NOT MEASURED"

# Table layout: (column header, dotted lookup path, float precision)
COLUMNS: list[tuple[str, str, int | None]] = [
    ("Experiment", "experiment_tag", None),
    ("Model", "model_name", None),
    ("KD terms", "kd_terms_str", None),
    ("AbsRel", "metric.abs_rel", 4),
    ("SqRel", "metric.sq_rel", 4),
    ("RMSE", "metric.rmse", 4),
    ("RMSElog", "metric.rmse_log", 4),
    ("delta1", "metric.delta1", 4),
    ("delta2", "metric.delta2", 4),
    ("delta3", "metric.delta3", 4),
    ("SILog", "metric.silog", 3),
    ("Params", "complexity.parameters", 0),
    ("GFLOPs", "complexity.gflops", 3),
    ("Latency ms", "latency.mean_ms", 2),
    ("FPS", "latency.fps", 2),
    ("Obst MAE", "distance.obstacle_distance_mae", 4),
    ("Obst RMSE", "distance.obstacle_distance_rmse", 4),
    ("±10%", "distance.percentage_within_10_percent", 1),
    ("±20%", "distance.percentage_within_20_percent", 1),
    ("±30%", "distance.percentage_within_30_percent", 1),
]


def dig(data: dict[str, Any], path: str) -> Any:
    """Look up a dotted path, returning None when any segment is missing."""
    node: Any = data
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def fmt(value: Any, precision: int | None) -> str:
    """Format a value, or NOT MEASURED when it is absent."""
    if value is None:
        return NOT_MEASURED
    if isinstance(value, str):
        return value
    if precision is None:
        return str(value)
    if precision == 0:
        return f"{int(value):,}"
    return f"{float(value):.{precision}f}"


def collect(experiments_dir: Path, extra_files: list[Path]) -> list[dict[str, Any]]:
    """Gather every experiment record found under ``experiments_dir``."""
    rows: list[dict[str, Any]] = []
    candidates = sorted(experiments_dir.rglob("metrics.json")) if experiments_dir.exists() else []
    candidates += [p for p in extra_files if p.exists()]

    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            LOG.warning("Skipping unreadable %s (%s)", path, e)
            continue

        record: dict[str, Any] = dict(data)
        record["_source"] = str(path)
        record.setdefault("experiment_tag", data.get("experiment_tag") or path.parent.name)
        record.setdefault("model_name", data.get("role", path.parent.name))

        terms = data.get("kd_terms")
        record["kd_terms_str"] = ", ".join(terms) if terms else ("none (baseline)" if data.get("role") else None)

        # Merge sibling evaluation artefacts written next to the run.
        for sibling, key in (("depth_metrics.json", "metric"), ("distance_metrics.json", "distance"),
                             ("latency_benchmark.json", "latency")):
            f = path.parent / sibling
            if f.exists():
                try:
                    payload = json.loads(f.read_text(encoding="utf-8"))
                    record[key] = payload.get(key, payload) if key == "metric" else payload
                except (json.JSONDecodeError, OSError):
                    pass

        rows.append(record)

    return rows


def to_markdown(rows: list[dict[str, Any]]) -> str:
    """Render the ablation table as Markdown."""
    header = "| " + " | ".join(c[0] for c in COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in COLUMNS) + "|"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(fmt(dig(r, p), prec) for _, p, prec in COLUMNS) + " |")

    body = "\n".join(lines)
    note = (
        "\n\n"
        f"`{NOT_MEASURED}` marks values that were not produced by an actual run. "
        "No cell is estimated, interpolated, or copied from another configuration "
        "(spec section BK).\n\n"
        "Depth metrics are MODE 1 (metric, `align=none`). Aligned figures are in each run's "
        "`depth_metrics.json` under `aligned`; they are NOT comparable to metric-scale accuracy.\n"
    )
    return body + note


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Build the ablation comparison table.")
    ap.add_argument("--experiments", type=Path, default=Path("outputs/experiments"))
    ap.add_argument("--include", type=Path, nargs="*", default=[], help="Extra metrics.json files")
    ap.add_argument("--output", type=Path, default=Path("docs/ablation"))
    args = ap.parse_args(argv)

    rows = collect(args.experiments, list(args.include))

    if not rows:
        LOG.warning("No experiment records found under %s.", args.experiments)
        LOG.warning("Run the training scripts first; every table cell would otherwise be %s.", NOT_MEASURED)

    # Order by the ablation tags A-F so the table reads in experiment order.
    order = {t: i for i, t in enumerate("ABCDEF")}
    rows.sort(key=lambda r: order.get(str(r.get("experiment_tag", ""))[:1], 99))

    md = to_markdown(rows)
    print()
    print(md)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    md_path = args.output.with_suffix(".md")
    md_path.write_text("# Ablation Study\n\n" + md, encoding="utf-8")

    csv_path = args.output.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([c[0] for c in COLUMNS])
        for r in rows:
            w.writerow([fmt(dig(r, p), prec) for _, p, prec in COLUMNS])

    LOG.info("Ablation table written to %s and %s (%d row(s))", md_path, csv_path, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
