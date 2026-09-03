"""Configuration and artifact IO (spec sections AR, AJ, BI).

Every experiment parameter lives in YAML; nothing is hard-coded in source.
``load_config`` supports single-level ``_base_`` inheritance so that, for example,
``distillation.yaml`` can inherit the student's training defaults and override
only the KD terms.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when a configuration file is missing, malformed, or inconsistent."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict, with an actionable error if it is missing."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(
            f"Config file not found: {path.resolve()}\n"
            f"  Searched relative to: {Path.cwd()}\n"
            f"  Recovery: copy one of the templates in configs/ or pass --config with a valid path."
        )
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Malformed YAML in {path}: {e}\n  Recovery: validate indentation and quoting.") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(data).__name__} in {path}.")
    return data


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving a single ``_base_`` inheritance chain.

    ``_base_`` is resolved relative to the child config's directory. The child's
    keys win over the base's, merged recursively.
    """
    path = Path(path)
    cfg = load_yaml(path)
    base_ref = cfg.pop("_base_", None)
    if base_ref:
        base_path = (path.parent / base_ref).resolve()
        base_cfg = load_config(base_path)
        cfg = _deep_update(base_cfg, cfg)
    return cfg


def _parse_scalar(raw: str) -> Any:
    """Coerce a CLI override string to a Python scalar.

    ``yaml.safe_load`` follows YAML 1.1, which does NOT recognise unsigned
    exponent floats such as ``1e-3`` (it requires ``1.0e-3``) and would silently
    hand back the string ``"1e-3"`` — turning a learning-rate override into a
    type error deep inside the optimizer. Scientific notation is therefore
    checked explicitly before falling back to the YAML parser.
    """
    text = raw.strip()
    if re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+", text):
        return float(text)
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return raw


def merge_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``key.subkey=value`` CLI overrides onto a config dict.

    Values are parsed as YAML scalars, so ``lr=1e-3``, ``deterministic=true`` and
    ``lambdas.gt=1.0`` all coerce to the right Python type.
    """
    if not overrides:
        return cfg
    out = dict(cfg)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Malformed override '{item}'. Expected 'key=value' or 'section.key=value'.")
        key, raw = item.split("=", 1)
        value = _parse_scalar(raw)
        node = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = value
    return out


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable SHA-256 (first 12 hex chars) of a config, for experiment provenance.

    Keys are sorted so logically identical configs hash identically regardless of
    the order they were written in.
    """
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def save_yaml(data: dict[str, Any], path: str | Path) -> Path:
    """Write a dict to YAML, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
    return path


def save_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    """Write an object to JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    return path


def load_json(path: str | Path) -> Any:
    """Read a JSON file."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"JSON file not found: {path.resolve()}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
