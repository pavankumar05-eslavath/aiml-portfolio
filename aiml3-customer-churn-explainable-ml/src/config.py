"""Configuration loading and path resolution.

All paths in configs/config.yaml are relative to the project root, which is
derived from this file's location. Nothing in the codebase hardcodes an absolute
path, so the project runs from any checkout directory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class Config:
    """Parsed configuration with helpers for path resolution."""

    raw: dict[str, Any] = field(repr=False)
    path: Path = DEFAULT_CONFIG

    # -- dict-style access -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    # -- convenience accessors --------------------------------------------
    @property
    def seed(self) -> int:
        return int(self.raw["project"]["seed"])

    @property
    def target(self) -> str:
        return str(self.raw["data"]["target"])

    @property
    def positive_label(self) -> str:
        return str(self.raw["data"]["positive_label"])

    @property
    def id_column(self) -> str:
        return str(self.raw["data"]["id_column"])

    def resolve(self, *keys: str) -> Path:
        """Resolve a dotted config value into an absolute path.

        >>> cfg.resolve("data", "raw_file")   # -> <root>/data/raw/telco_churn.csv
        """
        node: Any = self.raw
        for k in keys:
            node = node[k]
        return (PROJECT_ROOT / str(node)).resolve()

    def ensure_dirs(self) -> None:
        """Create the output directories a run writes into."""
        for keys in (("paths", "models_dir"), ("paths", "reports_dir"),
                     ("paths", "figures_dir"), ("data", "processed_dir")):
            self.resolve(*keys).mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from YAML."""
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    with p.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, path=p)
